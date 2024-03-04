import torch

def send_center_model_to_clients(center_model, clients_model_list):
    with torch.no_grad():
        for client_model in clients_model_list:
            client_model.load_state_dict(center_model.state_dict(), strict=True)

    return clients_model_list

def set_center_model_with_weights(center_model, clients_model_list, weights):
    with torch.no_grad():
        for client_model, weight in zip(clients_model_list, weights):
            for center_param, client_param in zip(center_model.parameters(), client_model.parameters()):
                center_param.data.add_(client_param.data * weight)

    return center_model

def set_center_mode_averagelly(center_model, clients_mode_list):
    set_center_model_with_weights(center_model, clients_mode_list, [1/len(clients_mode_list) for i in range(len(clients_mode_list))])
    return center_model