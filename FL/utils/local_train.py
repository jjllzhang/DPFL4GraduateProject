import torch
import math

from train_and_validation.train import train
from train_and_validation.train_with_dp import train_with_dp
from data.utils.sampling import get_data_loaders_uniform_without_replace

def local_clients_train_without_dp_one_epoch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device):
    for i in range(num_of_clients):
        batch_size = math.floor(len(clients_data_list[i]) * q)
        train_dataloader = torch.utils.data.DataLoader(clients_data_list[i], batch_size=batch_size, shuffle=False, drop_last = True)
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]
        criterion = clients_criterion_list[i]

        # print("Client:", i)
        
        for epoch in range(num_epoch):
            train_loss, train_accuracy = train(model, train_dataloader, optimizer, device)
            # print(f'Epoch: {epoch + 1} |'f' Train Loss: {train_loss:.4f} |'f' Train Accuracy: {train_accuracy:.2f}%')


def local_clients_train_without_dp_one_batch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device):
    for i in range(num_of_clients):
        batch_size = math.floor(len(clients_data_list[i]) * q)
        minibatch_size = batch_size
        microbatch_size = 1
        iterations = 1
        minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations)
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]
        criterion = clients_criterion_list[i]

        # print("Client:", i)
        
        for epoch in range(num_epoch):
            train_dataloader = minibatch_loader(clients_data_list[i])
            train_loss, train_accuracy = train(model, train_dataloader, optimizer, device)
            # print(f'Epoch: {epoch + 1} |'f' Train Loss: {train_loss:.4f} |'f' Train Accuracy: {train_accuracy:.2f}%')

def local_clients_train_with_dp_one_epoch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device):
    for i in range(num_of_clients):
        batch_size = math.floor(len(clients_data_list[i]) * q)
        train_dataloader = torch.utils.data.DataLoader(clients_data_list[i], batch_size=batch_size, shuffle=False, drop_last = True)
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]
        criterion = clients_criterion_list[i]

        # print("Client:", i)

        for epoch in range(num_epoch):
            train_loss, train_accuracy = train_with_dp(model, train_dataloader, optimizer, device)
            # print(f'Epoch: {epoch + 1} |'f' Train Loss: {train_loss:.4f} |'f' Train Accuracy: {train_accuracy:.2f}%')

def local_clients_train_with_dp_one_batch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device):
    for i in range(num_of_clients):
        batch_size = math.floor(len(clients_data_list[i]) * q)
        minibatch_size = batch_size
        microbatch_size = 1
        iterations = 1
        minibatch_loader, microbatch_loader = get_data_loaders_uniform_without_replace(minibatch_size, microbatch_size, iterations)
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]
        criterion = clients_criterion_list[i]

        # print("Client:", i)
        
        for epoch in range(num_epoch):
            train_dataloader = minibatch_loader(clients_data_list[i])
            train_loss, train_accuracy = train_with_dp(model, train_dataloader, optimizer, device)
            # print(f'Epoch: {epoch + 1} |'f' Train Loss: {train_loss:.4f} |'f' Train Accuracy: {train_accuracy:.2f}%')