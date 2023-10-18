import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv(r'C:\Users\HUAWEI\Desktop\data.csv')

# 创建一个图形和一个轴
fig, ax1 = plt.subplots()

# 绘制 epoch - iou 折线图
ax1.plot(df['epoch'], df['iou'], color='blue', label='IOU')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('IOU', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建第二个 y 轴
ax2 = ax1.twinx()
ax2.plot(df['epoch'], df['loss'], color='red', label='Loss')
ax2.set_ylabel('Loss', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# 添加图例
fig.tight_layout()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

# 显示图表
plt.show()
