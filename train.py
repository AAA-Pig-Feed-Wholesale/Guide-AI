# train_and_infer.py
import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model
from utils import LLMDataset, ID2LABEL
import numpy as np
import torch
import torch.nn as nn  # ✅ 确保这一行存在
from transformers import AutoModelForCausalLM  # ✅ 替换 AutoModel
from tqdm import tqdm



# -----------------------------
# 配置
# -----------------------------
MODEL_NAME = "Qwen/Qwen-1_8B-Chat"  #
OUTPUT_DIR = "models/fold_"
PREDICTION_DIR = "predictions/"
DATA_DIR = "data/"
NUM_FOLDS = 5
MAX_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # 模拟 batch_size = 4 * 4 = 16
LEARNING_RATE = 3e-4
EPOCHS = 3

os.makedirs(PREDICTION_DIR, exist_ok=True)

# 强制使用 PyTorch
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# -----------------------------
# 加载 tokenizer（只需一次）
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True  # ⚠️ 必须添加
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 对 Qwen-7B 有效
    # 或者显式设置：
    # tokenizer.pad_token = '<|endoftext|>'
tokenizer.padding_side = "left"  # Qwen 推荐 left-padding（因为是 decoder-only）


# -----------------------------
# 自定义 Qwen 分类模型
# -----------------------------
class QwenForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels=3, id2label=None, label2id=None):
        super().__init__()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        # ✅ 关键：trust_remote_code=True
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        hidden_size = self.qwen.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = sequence_lengths.clamp(min=0)
        pooled_output = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits}

# -----------------------------
# 5折交叉验证主循环
# -----------------------------
fold_preds = []  # 存储每折对 test 的预测概率

for fold in range(NUM_FOLDS):
    print(f"\n🚀 开始训练 Fold {fold}")

    # 1. 加载数据
    print("📊 正在加载数据...")
    train_df = pd.read_csv(f"{DATA_DIR}/fold{fold}_train.csv")
    val_df = pd.read_csv(f"{DATA_DIR}/fold{fold}_val.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test_clean.csv")

    # 2. 构造输入文本
    print("📝 正在构造输入文本...")
    def make_input(row):
        return f"Prompt: {row['prompt']}\nResponse A: {row['response_a']}\nResponse B: {row['response_b']}\nQuestion: Which response is better? Answer:"

    train_texts = train_df.apply(make_input, axis=1).tolist()
    val_texts = val_df.apply(make_input, axis=1).tolist()
    test_texts = test_df.apply(make_input, axis=1).tolist()

    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()

    # 3. 创建 dataset
    print("💾 正在创建数据集...")
    train_dataset = LLMDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = LLMDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # 4. 加载模型（分类任务）
    print("🧠 正在加载模型...")
    model = QwenForSequenceClassification(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()}  # 假设 LABEL2ID 未定义
    )

    # 5. 添加 LoRA
    print("🔧 正在配置LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的模块名
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",  # 注意：虽然做分类，但 Qwen 是 causal LM
        modules_to_save=["classifier"]  # ✅ 关键！必须保存分类头
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 应显示少量可训练参数

    # 6. 训练参数
    print("⚙️ 正在设置训练参数...")
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}{fold}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",  # 不上传wandb等
        fp16=True,  # 使用混合精度
        remove_unused_columns=False,
        warmup_ratio=0.1,
        seed=42,
    )

    # 7. 定义评估指标
    print("📈 正在定义评估指标...")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = predictions.argmax(-1)
        macro_f1 = f1_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "macro_f1": macro_f1}

    # 8. 创建 Trainer
    print("🎯 正在创建Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 9. 开始训练
    print("🔥 正在开始训练...")
    with tqdm(total=training_args.num_train_epochs, desc="Epochs") as pbar:
        for epoch in range(training_args.num_train_epochs):
            # 执行训练
            trainer.train()
            pbar.update(1)

    # 10. 验证集预测（可选：看每折性能）
    print("🔍 正在进行验证集预测...")
    val_preds = trainer.predict(val_dataset)
    val_pred_labels = val_preds.predictions.argmax(-1)
    val_true_labels = val_preds.label_ids
    macro_f1 = f1_score(val_true_labels, val_pred_labels, average='macro')
    print(f"Fold {fold} 验证集 Macro F1: {macro_f1:.4f}")

    # 11. 对 test 集预测（输出概率）
    print("🧪 正在进行测试集预测...")
    test_dataset = LLMDataset(test_texts, ["A"] * len(test_texts), tokenizer, MAX_LENGTH)  # label占位
    test_preds = trainer.predict(test_dataset)
    test_probs = torch.nn.functional.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()  # (N, 3)
    fold_preds.append(test_probs)

    # 12. 释放显存
    print("🧹 正在清理显存...")
    del model, trainer
    torch.cuda.empty_cache()


# -----------------------------
# 第四步：集成预测（5折平均）
# -----------------------------
# 所有 fold 的预测概率平均
final_probs = np.mean(fold_preds, axis=0)  # (N, 3)
final_preds = final_probs.argmax(axis=1)  # 取最大概率
final_labels = [ID2LABEL[i] for i in final_preds]

# -----------------------------
# 第五步：生成提交文件
# -----------------------------
submission = pd.DataFrame({
    'id': test_df['id'],
    'label': final_labels
})
submission.to_csv(f"{PREDICTION_DIR}/submission_ensemble.csv", index=False)
print("✅ 提交文件生成完毕！")
print(submission['label'].value_counts())