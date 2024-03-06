import torch
import yaml
import os
import csv
from FL.fed_avg.fed_avg import fed_avg
from FL.fed_avg.fed_avg_with_dp import fed_avg_with_dp
from FL.fed_avg.fed_avg_with_dp_perlayer import fed_avg_with_dp_perlayer
from data.utils.get_data import load_dataset
from models.get_model import get_model


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


import os
import csv

def log_results(config, test_loss_list, test_acc_list):
    config_modified = config.copy()
    keys_to_remove = ['device', 'epochs', 'momentum', 'seed']  # 将要移除的键存储在列表中
    for key in keys_to_remove:
        config_modified.pop(key, None)  # 移除指定的键，如果不存在则忽略

    log_dir = './log/'
    os.makedirs(log_dir, exist_ok=True)

    config_items = [f"{key}_{value}" for key, value in config_modified.items()]
    filename = f"{'_'.join(config_items)}.csv"
    filepath = os.path.join(log_dir, filename)

    file_exists = os.path.isfile(filepath)

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'test_loss', 'test_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for epoch, (loss, acc) in enumerate(zip(test_loss_list, test_acc_list), start=1):
            writer.writerow({'epoch': epoch, 'test_loss': loss, 'test_accuracy': acc})


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
    test_loss_list = None
    test_acc_list = None

    if algorithm == 'fed_avg':
        test_loss_list, test_acc_list = fed_avg(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, model, device)
    elif algorithm == 'fed_avg_with_dp':
        test_loss_list, test_acc_list = fed_avg_with_dp(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)
    elif algorithm == 'fed_avg_with_dp_perlayer':
        test_loss_list, test_acc_list = fed_avg_with_dp_perlayer(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")
    
    if test_loss_list is not None and test_acc_list is not None:
        log_results(config, test_loss_list, test_acc_list)