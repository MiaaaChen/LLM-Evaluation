import pandas as pd

# 設定輸入 CSV 檔案
input_csv = "label_category.csv"
output_csv = "llm_category_count.csv"

# 讀取 CSV 檔案
df = pd.read_csv(input_csv)

# 確保 'ID' 和 'label' 欄位存在
if "ID" not in df.columns or "label" not in df.columns:
    raise ValueError("CSV 檔案缺少 'ID' 或 'label' 欄位")

# 解析 LLM 名稱
df["LLM"] = df["ID"].apply(lambda x: x.split("_")[0])

# option 1. 計算 True/False 數量
# llm_counts = df.groupby(["LLM", "label"]).size().unstack(fill_value=0)

# option 2. 計算 Likert 數量
# df["label"] = df["label"].apply(lambda x: str(x))  # 確保所有值都是字串
# llm_counts = df.groupby(["LLM", "label"]).size().unstack(fill_value=0)

# # option 3. 計算 Category 數量
llm_counts = df.groupby(["LLM", "label"]).size().unstack(fill_value=0)

# 儲存結果到 CSV
llm_counts.to_csv(output_csv)
print(f"✅ 統計結果已儲存至 {output_csv}")
