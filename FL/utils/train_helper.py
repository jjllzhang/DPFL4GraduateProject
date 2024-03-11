import yaml
import torch
import random

def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        loaded_config = yaml.safe_load(file)
    return loaded_config

def save_model_and_optimizers(model, optimizers, save_path):
    state = {
        'model_state': model.state_dict(),
        'optimizers_state': [optimizer.state_dict() for optimizer in optimizers]
    }
    torch.save(state, save_path)

def load_model_and_optimizers(model, optimizers, load_path):
    state = torch.load(load_path)
    model.load_state_dict(state['model_state'])
    for optimizer, state_dict in zip(optimizers, state['optimizers_state']):
        optimizer.load_state_dict(state_dict)


def shuffle_weights(weight_of_each_client):
    shuffled_weight_of_each_client = weight_of_each_client[:]
    random.shuffle(shuffled_weight_of_each_client)
    return shuffled_weight_of_each_client

