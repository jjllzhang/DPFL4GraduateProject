from FL.utils.train_helper import load_model_and_optimizers, save_model_and_optimizers, shuffle_weights
from data.utils.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL.utils.create_client import create_clients_with_dp
from FL.utils.update_model import send_center_model_to_clients, set_center_model_with_weights
from FL.utils.local_train import local_clients_train_with_dp_one_epoch
from privacy_analysis.compute_rdp import compute_rdp
from privacy_analysis.rdp_convert_dp import compute_eps
from train_and_validation.validation import validation
import os
import torch
import math


def fed_avg_with_dp(train_data, test_data, test_batchsize, num_of_clients, lr, momentum, num_epoch, iters, alpha, seed, q, max_norm, sigma, delta, model, device, start_round=0, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'saved_states')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    clients_data_list, weight_of_each_client, batchsize_of_each_client = fed_dataset_NonIID_Dirichlet(train_data, num_of_clients, alpha, seed, q)
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_clients_with_dp(num_of_clients, lr, model, momentum, max_norm, sigma, batchsize_of_each_client)
    center_model = model.to(device)
    
    load_path = f"{save_dir}/model_optimizers_state_round_{start_round}_fed_avg_with_dp.pt"
    if start_round > 0 and os.path.exists(load_path):
        load_model_and_optimizers(center_model, clients_optimizer_list, load_path)

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batchsize, shuffle=False)
    # orders = (list(range(2,64)) + [128, 256, 512])
    # rdp = compute_rdp(q, sigma, start_round * num_epoch * math.floor(1/q), orders)

    print("Start Federated Learning with DP")
    test_acc_list = []
    test_loss_list = []
    for i in range(iters):
        current_round = i + start_round
        print(f"Round: {current_round + 1}")
        clients_model_list = send_center_model_to_clients(center_model, clients_model_list)
        local_clients_train_with_dp_one_epoch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device)
        
        # rdp += compute_rdp(q, sigma, num_epoch * math.floor(1/q), orders)
        # epsilon, best_alpha = compute_eps(orders, rdp, delta)
        # print(f"Iteration: {current_round + 1}, Epsilon: {epsilon:.4f}, Best Alpha: {best_alpha}")
        
        weight_of_each_client = shuffle_weights(weight_of_each_client)
        center_model = set_center_model_with_weights(center_model, clients_model_list, weight_of_each_client)
        test_loss, test_acc = validation(center_model, test_data_loader, device)
        print(f"Iteration: {current_round + 1}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f} %")
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        
        if (current_round + 1) % 100 == 0:
            save_path = f"{save_dir}/model_optimizers_state_round_{current_round + 1}_fed_avg_with_dp.pt"
            save_model_and_optimizers(center_model, clients_optimizer_list, save_path)

    record = [test_loss_list, test_acc_list]
    return record
