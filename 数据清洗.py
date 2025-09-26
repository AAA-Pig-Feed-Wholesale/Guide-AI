import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取原始数据
train_df = pd.read_csv('llm-classification-finetuning/train.csv')
test_df = pd.read_csv('llm-classification-finetuning/test.csv')

# -----------------------------
# 2. 数据清洗（示例）
# -----------------------------

# 去除完全重复的行
train_df = train_df.drop_duplicates()

# 去除关键字段为空的样本
train_df = train_df.dropna(subset=['prompt', 'response_a', 'response_b'])

# 简单去噪：去除空白字符过多的文本（可选）
train_df = train_df[train_df['prompt'].str.strip().str.len() > 5]
train_df = train_df[train_df['response_a'].str.strip().str.len() > 5]
train_df = train_df[train_df['response_b'].str.strip().str.len() > 5]

# 重置索引
train_df = train_df.reset_index(drop=True)

# 添加label列（根据winner_*列确定获胜者）
train_df['label'] = train_df.apply(lambda row: 0 if row['winner_model_a'] == 1 else (1 if row['winner_model_b'] == 1 else 2), axis=1)

# 对 test.csv 也做同样清洗（但不要删label，因为test本来就没label）
test_df = test_df.drop_duplicates()
test_df = test_df.dropna(subset=['prompt', 'response_a', 'response_b'])
test_df = test_df.reset_index(drop=True)

# -----------------------------
# 3. 划分训练集和验证集（8:1:1 中的 8:1 → 训练:验证）
# -----------------------------
# 注意：原始train.csv是57,477条，test.csv是独立的测试集
# 所以我们只从train_df中划分出val集，test_df保持不变

train_part, val_part = train_test_split(
    train_df,
    test_size=0.1,               # 10%作为验证集
    random_state=42,
    stratify=train_df['label']   # 保证各类别比例一致
)

# -----------------------------
# 4. 输出所有清洗后的数据
# -----------------------------

# 输出全部清洗后的训练数据（可用于交叉验证）
train_df.to_csv('data/full_clean.csv', index=False)

# 输出训练集
train_part.to_csv('data/train_clean.csv', index=False)

# 输出验证集
val_part.to_csv('data/val_clean.csv', index=False)

# 输出清洗后的测试集（用于最终预测）
test_df.to_csv('data/test_clean.csv', index=False)

print("✅ 数据清洗完成，已输出到 /data/ 目录！")
print(f"训练集大小: {len(train_part)}")
print(f"验证集大小: {len(val_part)}")
print(f"测试集大小: {len(test_df)}")


from sklearn.model_selection import StratifiedKFold

# 创建5折分割器
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]

    fold_train.to_csv(f'data/fold{fold}_train.csv', index=False)
    fold_val.to_csv(f'data/fold{fold}_val.csv', index=False)

print("✅ 5折交叉验证数据已保存！")