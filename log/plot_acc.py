import matplotlib.pyplot as plt
import pandas as pd
import os
import re  # 导入正则表达式模块

directory = './FMNIST/lr=0.001'  # CSV文件目录
plt.figure(figsize=(10, 6))  # 设置图形大小

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path)

        # 使用正则表达式来捕获max_norm和sigma的值
        max_norm_match = re.search(r'max_norm_([0-9]+(?:\.[0-9]+)?)', filename)
        sigma_match = re.search(r'sigma_([0-9]+(?:\.[0-9]+)?)', filename)
        algorithm_match = re.search(r'algorithm_(.+)\.csv', filename)

        # 提取max_norm和sigma的值
        max_norm_value = max_norm_match.group(1) if max_norm_match else "Unknown"
        sigma_value = sigma_match.group(1) if sigma_match else "Unknown"
        algorithm_name = algorithm_match.group(1) if algorithm_match else "Unknown"

        # 构建简化的文件名以用作图例标签
        simplified_filename = f"C={max_norm_value}, Sigma={sigma_value}, Algorithm {algorithm_name}"

        # 绘制图形
        plt.plot(data['epoch'], data['test_accuracy'], label=simplified_filename)


plt.xlabel('Epoch')  # 设置X轴标签
plt.ylabel('Test Accuracy')  # 设置Y轴标签
plt.title('Test Accuracy over Epochs(FMNIST)')  # 设置图形标题

# 调整图例位置和大小
plt.legend(
    # fontsize='x-small',  # 更小的字体大小
    # loc='lower right',  # 放置在右下角，通常数据集中度较低的区域
    # framealpha=0.5,  # 图例背景半透明
    # fancybox=True  # 无边框背景
)

# 设置X轴为对数尺度
# plt.xscale('log')

# 保存为PNG文件，设置dpi为300
plt.savefig('./FMNIST/lr=0.001/test_accuracy_FMNIST.png', dpi=300, bbox_inches='tight')

# 展示图形
plt.show()
