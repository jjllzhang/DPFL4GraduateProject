from FL.fed_avg.fed_avg_with_dp import fed_avg_with_dp
from data.utils.get_data import load_dataset
from models.get_model import get_model
import torch


if __name__=="__main__":
    train_data, test_data = load_dataset('MNIST')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model('DPSGD', 'MNIST', device)
    batch_size=256
    learning_rate = 0.002
    numEpoch = 1       #客户端本地下降次数
    number_of_clients=10
    momentum=0.9
    iters=100
    alpha=0.05 #狄立克雷的异质参数
    seed=1   #狄立克雷的随机种子
    q_for_batch_size=0.01   #基于该数据采样率组建每个客户端的batchsize
    max_norm=0.1
    sigma=1.1
    delta=1e-5
    fed_avg_with_dp(train_data, test_data, batch_size, number_of_clients, learning_rate, momentum, numEpoch, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)