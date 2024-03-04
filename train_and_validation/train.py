import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.to(torch.float32))
        loss = F.cross_entropy(output, target.to(torch.long))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc
