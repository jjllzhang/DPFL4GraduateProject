import matplotlib.pyplot as plt
import csv
import os
from FL.utils.train_helper import load_config

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def reconstruct_filename_from_config(config):
    # 移除不需要记录在文件名中的配置项
    keys_to_remove = ['device', 'epochs', 'momentum', 'seed', 'save_dir', 'start_round', 'test_batch_size', 'iters', 'learning_rate', 'num_clients', 'alpha', 'max_norm', 'dataset']
    for key in keys_to_remove:
        config.pop(key, None)
    
    # 生成文件名
    config_items = [f"{key}_{value}" for key, value in config.items()]
    filename = f"{'_'.join(config_items)}.csv"
    filepath = os.path.join('./log/', filename)
    
    return filepath

# 假设这是当初写入文件时使用的配置字典
CONFIG_PATH = "../config.yml"
config_used_for_logging =  load_config(CONFIG_PATH)


# 使用相同的配置重建文件名
filepath = reconstruct_filename_from_config(config_used_for_logging.copy())

# 初始化两个空列表来存储数据
rounds = []
epsilons = []
opt_orders = []

# 读取CSV文件
with open(filepath, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        rounds.append(int(row['round']))
        epsilons.append(float(row['epsilon']))

# 绘制 epsilon 随 round 变化的图像
plt.figure(figsize=(8, 4))
plt.plot(rounds, epsilons, marker='o', linestyle='-', color='b', label=r'$\epsilon$', linewidth=1, markersize=5)
plt.title(r'Epsilon Changes Over Rounds')
plt.xlabel(r'Round')
plt.ylabel(r'Epsilon')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('./log/epsilon_changes.pdf', format='pdf', dpi=1200)