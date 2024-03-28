import pandas as pd
import matplotlib.pyplot as plt

# 定义一个函数用来读取数据并绘制图形
def plot_from_csv(csv_path, label):
    data = pd.read_csv(csv_path)
    plt.plot(data['round'], data['epsilon'], label=label)

# 曲线的名字
labels = ['N=5', 'N=10', 'N=20', 'N=50', 'N=100']

# 读取文件并绘制曲线
plot_from_csv('./log/num_clients_5_q_for_batch_size_0.01_sigma_1.0_delta_1e-5_algorithm_fed_avg_with_dp_with_shuffler.csv', labels[0])
plot_from_csv('./log/num_clients_10_q_for_batch_size_0.01_sigma_1.0_delta_1e-5_algorithm_fed_avg_with_dp_with_shuffler.csv', labels[1])
plot_from_csv('./log/num_clients_20_q_for_batch_size_0.01_sigma_1.0_delta_1e-5_algorithm_fed_avg_with_dp_with_shuffler.csv', labels[2])
plot_from_csv('./log/num_clients_50_q_for_batch_size_0.01_sigma_1.0_delta_1e-5_algorithm_fed_avg_with_dp_with_shuffler.csv', labels[3])
plot_from_csv('./log/num_clients_100_q_for_batch_size_0.01_sigma_1.0_delta_1e-5_algorithm_fed_avg_with_dp_with_shuffler.csv', labels[4])

# 添加图例
plt.legend()

# 设置对数尺度的x轴
# plt.xscale('log')

# 设置坐标轴标签和标题
plt.xlabel('T')
plt.ylabel('ε')
plt.title('Privacy loss with δ = 1.1 x 1e-5')

# 显示图形
plt.show()

plt.savefig('./log/privacy_loss_comparsion(q=0.01, sigma=1.0).png', dpi=300)
