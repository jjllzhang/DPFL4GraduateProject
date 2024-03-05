import torch
import copy

from utils.dp_optimizer import DPAdam_Optimizer

def create_clients(num_of_clients, lr, model):
    clients_list = []
    clients_optimizer_list = []
    clients_criterion_list = []
    for i in range(num_of_clients):
        clients_list.append(copy.deepcopy(model))
        clients_criterion_list.append(torch.nn.CrossEntropyLoss())
        clients_optimizer_list.append(torch.optim.SGD(clients_list[i].parameters(), lr=lr))

    return clients_list, clients_optimizer_list, clients_criterion_list

def create_clients_with_dp(num_of_clients, lr, model, momentum, max_norm, sigma, batch_size_of_each_clients):
    clients_list = []
    clients_optimizer_list = []
    clients_criterion_list = []
    for i in range(num_of_clients):
        clients_list.append(copy.deepcopy(model))
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=max_norm,
            noise_multiplier=sigma,
            minibatch_size=batch_size_of_each_clients[i],
            microbatch_size=1,
            params=clients_list[i].parameters(),
            lr=lr,
        )
        clients_optimizer_list.append(optimizer)
        clients_criterion_list.append(torch.nn.CrossEntropyLoss())

    return clients_list, clients_optimizer_list, clients_criterion_list