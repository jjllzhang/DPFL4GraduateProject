from data.utils.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL.utils.create_client import create_clients
from FL.utils.update_model import send_center_model_to_clients, set_center_model_with_weights
from FL.utils.local_train import local_clients_train_without_dp_one_batch, local_clients_train_without_dp_one_epoch
from train_and_validation.validation import validation
from data.utils.get_data import load_dataset
from models.get_model import get_model
import torch

def fed_avg(train_data, test_data, test_batchsize, num_of_clients, lr, momentum, num_epoch, iters, alpha, seed, q, model, device):
    clients_data_list, weight_of_each_client, batchsize_of_each_client = fed_dataset_NonIID_Dirichlet(train_data, num_of_clients, alpha, seed, q)
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_clients(num_of_clients, lr, model)
    center_model = model
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batchsize, shuffle=False)
    print("Start Federated Learning without DP")
    test_acc_list = []
    test_loss_list = []
    for i in range(iters):
        print("round: ", i+1)
        clients_model_list = send_center_model_to_clients(center_model, clients_model_list)
        local_clients_train_without_dp_one_epoch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device)
        center_model = set_center_model_with_weights(center_model, clients_model_list, weight_of_each_client)
        test_loss, test_acc = validation(center_model, test_data_loader, device)
        print(f"Iteration: {i+1}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f} %")
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

    record = [test_loss_list, test_acc_list]
    return record

