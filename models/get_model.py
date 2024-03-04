from models.CIFAR10 import CIFAR10Net
from models.MNIST import MNISTNet

def get_model(algorithm, dataset_name, device):
    if algorithm == 'DPSGD':
        if dataset_name == 'MNIST':
            model = MNISTNet().to(device)
        elif dataset_name == 'CIFAR-10':
            model = CIFAR10Net().to(device)
        else:
            raise ValueError("Unsupported dataset name.")
        
    else:
        pass

    return model