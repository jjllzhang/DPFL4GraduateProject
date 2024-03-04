import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

def train_with_dp(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_accuracy = 0
    correct = 0
    total = 0

    for id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_accum_grad()

        for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):
            optimizer.zero_microbatch_grad()
            output = model(torch.unsqueeze(X_microbatch, 0))

            if len(output.shape) == 2:
                output = torch.squeeze(output, 0)
            
            loss = F.cross_entropy(output, y_microbatch)
            loss.backward()
            optimizer.microbatch_step()

            # train_loss += loss.item()
            # _, predicted = torch.max(output.data, 0)
            # correct += (predicted == y_microbatch).sum().item()
            # total += y_microbatch.size(0)

        optimizer.step_dp()

    # train_loss /= total
    # train_accuracy = 100. * correct / total

    return train_loss, train_accuracy

