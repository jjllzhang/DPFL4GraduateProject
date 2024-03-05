import argparse
import torch
from FL.fed_avg.fed_avg import fed_avg
from FL.fed_avg.fed_avg_with_dp import fed_avg_with_dp
from FL.fed_avg.fed_avg_with_dp_perlayer import fed_avg_with_dp_perlayer
from data.utils.get_data import load_dataset
from models.get_model import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行带有不同参数的机器学习模型 (Run a machine learning model with different parameters)')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'FMNIST'], help='数据集名称 (Dataset name)')
    parser.add_argument('--test_batch_size', type=int, default=256, help='测试批量大小 (Test batch size)')
    parser.add_argument('--lr', type=float, default=0.002, help='学习率 (Learning rate)')
    parser.add_argument('--epochs', type=int, default=1, help='训练周期数 (Number of epochs)')
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量 (Number of clients)')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量 (Momentum)')
    parser.add_argument('--iters', type=int, default=100, help='迭代次数 (Number of iterations)')
    parser.add_argument('--alpha', type=float, default=0.05, help='狄立克雷分布的参数 (Dirichlet distribution alpha)')
    parser.add_argument('--seed', type=int, default=1, help='随机种子 (Random seed)')
    parser.add_argument('--q_for_batch_size', type=float, default=0.01, help='数据采样率 (Data sampling rate)')
    parser.add_argument('--max_norm', type=float, default=0.1, help='最大范数 (Max norm)')
    parser.add_argument('--sigma', type=float, default=1.1, help='隐私保护参数σ (Privacy parameter sigma)')
    parser.add_argument('--delta', type=float, default=1e-5, help='隐私保护参数δ (Privacy parameter delta)')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备，cpu或cuda (Training device, cpu or cuda)')
    parser.add_argument('--algorithm', type=str, default='fed_avg', choices=['fed_avg', 'fed_avg_with_dp', 'fed_avg_with_dp_perlayer'], help='使用的算法 (Algorithm used)')

    args = parser.parse_args()

    train_data, test_data = load_dataset(args.dataset)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = get_model(args.dataset, device)
    test_batch_size = args.test_batch_size
    learning_rate = args.lr
    num_epoch = args.epochs
    number_of_clients = args.num_clients
    momentum = args.momentum
    iters = args.iters
    alpha = args.alpha
    seed = args.seed
    q_for_batch_size = args.q_for_batch_size
    max_norm = args.max_norm
    sigma = args.sigma
    delta = args.delta

    if args.algorithm == 'fed_avg':
        fed_avg(train_data, test_data, test_batch_size, number_of_clients, learning_rate, momentum, num_epoch, iters, alpha, seed, q_for_batch_size, model, device)
    elif args.algorithm == 'fed_avg_with_dp':
        fed_avg_with_dp(train_data, test_data, test_batch_size, number_of_clients, learning_rate, momentum, num_epoch, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)
    elif args.algorithm == 'fed_avg_with_dp_perlayer':
        fed_avg_with_dp_perlayer(train_data, test_data, test_batch_size, number_of_clients, learning_rate, momentum, num_epoch, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)
    else:
        raise NotImplementedError