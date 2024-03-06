import torch
import yaml
from FL.fed_avg.fed_avg import fed_avg
from FL.fed_avg.fed_avg_with_dp import fed_avg_with_dp
from FL.fed_avg.fed_avg_with_dp_perlayer import fed_avg_with_dp_perlayer
from data.utils.get_data import load_dataset
from models.get_model import get_model


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config_path = "config.yml"
    config = load_config(config_path)

    dataset = config['dataset']
    test_batch_size = config['test_batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    num_clients = config['num_clients']
    momentum = config['momentum']
    iters = config['iters']
    alpha = config['alpha']
    seed = config['seed']
    q_for_batch_size = config['q_for_batch_size']
    max_norm = config['max_norm']
    sigma = config['sigma']
    delta = config['delta']
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    algorithm = config['algorithm']

    train_data, test_data = load_dataset(dataset)
    model = get_model(dataset, device)

    if algorithm == 'fed_avg':
        test_loss_and_acc_record = fed_avg(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, model, device)
    elif algorithm == 'fed_avg_with_dp':
        test_loss_and_acc_record = fed_avg_with_dp(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)
    elif algorithm == 'fed_avg_with_dp_perlayer':
        test_loss_and_acc_record = fed_avg_with_dp_perlayer(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")