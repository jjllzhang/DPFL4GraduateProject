import os
import torch
from FL.fed_avg.fed_avg import fed_avg
from FL.fed_avg.fed_avg_with_dp import fed_avg_with_dp
from FL.fed_avg.fed_avg_with_dp_auto import fed_avg_with_dp_auto
from FL.fed_avg.fed_avg_with_dp_perlayer import fed_avg_with_dp_perlayer
from algorithms.train_with_DPSGD import train_with_DPSGD
from data.utils.get_data import load_dataset
from models.get_model import get_model
from FL.utils.train_helper import load_config
from FL.utils.log_helper import log_results


if __name__ == "__main__":
    CONFIG_PATH = "config.yml"
    config = load_config(CONFIG_PATH)

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
    delta = float(config['delta'])
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    algorithm = config['algorithm']
    start_round = config['start_round']
    
    if config['algorithm'] == 'DPSGD':
        parameters_string = f"dataset_{dataset}_lr_{learning_rate}_q_{q_for_batch_size}_max_norm_{max_norm}_sigma_{sigma}_delta_{delta}"
    else:
        parameters_string = f"dataset_{dataset}_lr_{learning_rate}_clients_{num_clients}_q_{q_for_batch_size}_max_norm_{max_norm}_sigma_{sigma}_delta_{delta}"
    save_dir = os.path.join('./saved_states', parameters_string)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    train_data, test_data = load_dataset(dataset)
    model = get_model(dataset, device)
    test_loss_list = None
    test_acc_list = None

    if algorithm == 'fed_avg':
        test_loss_list, test_acc_list = fed_avg(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, model, device, start_round, save_dir)
    elif algorithm == 'fed_avg_with_dp':
        test_loss_list, test_acc_list = fed_avg_with_dp(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device, start_round, save_dir)
    elif algorithm == 'fed_avg_with_dp_perlayer':
        test_loss_list, test_acc_list = fed_avg_with_dp_perlayer(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device, start_round, save_dir)
    elif algorithm == 'fed_avg_with_dp_auto':
        test_loss_list, test_acc_list = fed_avg_with_dp_auto(train_data, test_data, test_batch_size, num_clients, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device, start_round, save_dir)
    elif algorithm == 'DPSGD':
        test_loss_list, test_acc_list = train_with_DPSGD(train_data, test_data, test_batch_size, learning_rate, momentum, epochs, iters, alpha, seed, q_for_batch_size, max_norm, sigma, delta, model, device, start_round, save_dir)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")
    
    if test_loss_list is not None and test_acc_list is not None:
        test_loss_list = [round(loss.item(), 2) for loss in test_loss_list]
        test_acc_list = [round(acc, 2) for acc in test_acc_list]
        log_results(config, test_loss_list, test_acc_list)