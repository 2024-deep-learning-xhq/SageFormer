import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "extract_result.csv"

df = pd.read_csv(OUTPUT_PATH)

# 创建透视表
pivot_table = df.pivot(index="d_model", columns="n_heads", values="mse")

# 创建热力图
plt.figure(figsize=(5, 4))
sns.heatmap(
    pivot_table,
    annot=True,  # 显示数值
    fmt=".3f",  # 数值格式化为3位小数
    cmap="Blues_r",  # 使用反转的配色方案（值越小颜色越深）
    square=True,  # 确保每个单元格是正方形
    cbar_kws={"label": "MSE"},
)

# plt.title("MSE Performance across Different Model Configurations")
plt.xlabel("Number of Heads")
plt.ylabel("Dimension of Model")

plt.tight_layout()
plt.savefig("extract_result.png")
plt.savefig("extract_result.eps")
