import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

def train_with_dp(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()

        microbatch_size = 1
        num_microbatch_size = data.size(0) // microbatch_size
        for i in range(num_microbatch_size):
            start_index = i * microbatch_size
            end_index = start_index + microbatch_size
            X_microbatch = data[start_index:end_index]
            y_microbatch = target[start_index:end_index]
            optimizer.zero_microbatch_grad()

            output = model(X_microbatch)
            loss = F.cross_entropy(output, y_microbatch)
            loss.backward()

            optimizer.microbatch_step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            correct += (predicted == y_microbatch).sum().item()
            total += y_microbatch.size(0)

        optimizer.step_dp()

    train_loss /= total
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy

def train_with_dp_perlayer(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()

        microbatch_size = 1
        num_microbatch_size = data.size(0) // microbatch_size
        for i in range(num_microbatch_size):
            start_index = i * microbatch_size
            end_index = start_index + microbatch_size
            X_microbatch = data[start_index:end_index]
            y_microbatch = target[start_index:end_index]
            optimizer.zero_microbatch_grad()

            output = model(X_microbatch)
            loss = F.cross_entropy(output, y_microbatch)
            loss.backward()

            optimizer.microbatch_step_perlayer()

            train_loss += loss.item()
            _, predicted = output.max(1)
            correct += (predicted == y_microbatch).sum().item()
            total += y_microbatch.size(0)

        optimizer.step_dp_perlayer()

    train_loss /= total
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy


def train_with_dp_auto(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()

        microbatch_size = 1
        num_microbatch_size = data.size(0) // microbatch_size
        for i in range(num_microbatch_size):
            start_index = i * microbatch_size
            end_index = start_index + microbatch_size
            X_microbatch = data[start_index:end_index]
            y_microbatch = target[start_index:end_index]
            optimizer.zero_microbatch_grad()

            output = model(X_microbatch)
            loss = F.cross_entropy(output, y_microbatch)
            loss.backward()

            optimizer.microbatch_step_auto()

            train_loss += loss.item()
            _, predicted = output.max(1)
            correct += (predicted == y_microbatch).sum().item()
            total += y_microbatch.size(0)

        optimizer.step_dp_auto()

    train_loss /= total
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy