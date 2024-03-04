from algorithms.DPSGD import DPSGD
from data.utils.get_data import load_dataset
from models.get_model import get_model
from utils.dp_optimizer import get_dp_optimizer
from datetime import datetime
import torch
import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="DPSGD",choices=['DPSGD', 'DPSGD-TS', 'DPSGD-HF', 'DPSUR', 'DPAGD'])
    parser.add_argument('--dataset_name', type=str, default="MNIST",choices=['MNIST', 'FMNIST', 'CIFAR-10', 'IMDB'])
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--use_scattering', action="store_true")
    parser.add_argument('--input_norm', default=None, choices=["GroupNorm", "BN"])
    parser.add_argument('--bn_noise_multiplier', type=float, default=8)
    parser.add_argument('--num_groups', type=int, default=27)

    parser.add_argument('--sigma_t', type=float, default=1.23)
    parser.add_argument('--C_t', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--sigma_v', type=float, default=1.0)
    parser.add_argument('--C_v', type=float, default=0.001)
    parser.add_argument('--bs_valid', type=int, default=256)
    parser.add_argument('--beta', type=float, default=-1.0)

    parser.add_argument('--MIA', type=bool, default=False)

    parser.add_argument('--device', type=str, default='cpu',choices=['cpu', 'cuda'])


    args = parser.parse_args()

    algorithm=args.algorithm
    dataset_name=args.dataset_name
    lr=args.lr
    momentum=args.momentum

    use_scattering=args.use_scattering
    input_norm=args.input_norm
    bn_noise_multiplier=args.bn_noise_multiplier
    num_groups=args.num_groups

    sigma_t=args.sigma_t
    C_t=args.C_t
    epsilon=args.epsilon
    delta=args.delta
    batch_size=args.batch_size

    sigma_v=args.sigma_v
    bs_valid=args.bs_valid
    C_v=args.C_v
    beta=args.beta

    MIA=args.MIA

    device=args.device

    if MIA:
        pass
    else:
        train_data, test_data = load_dataset(dataset_name)
        model=get_model(algorithm,dataset_name,device)
        optimizer = get_dp_optimizer(lr, momentum, C_t, sigma_t, batch_size, model)

        if algorithm == 'DPSGD':
            test_acc,last_iter,best_acc,best_iter,trained_model,iter_list=DPSGD(train_data, test_data, model,optimizer, batch_size, epsilon, delta,sigma_t,device)
        else:
            pass

    if MIA:
        pass
    else:
        File_Path_Csv = os.getcwd() + f"/result/Without_MIA/{algorithm}/{dataset_name}/{epsilon}/"
        if not os.path.exists(File_Path_Csv):
            os.makedirs(File_Path_Csv)
        result_path = f'{File_Path_Csv}/{str(sigma_t)}_{str(lr)}_{str(batch_size)}_{str(sigma_v)}_{str(bs_valid)}.csv'
        pd.DataFrame([best_acc, int(best_iter), test_acc, int(last_iter)]).to_csv(result_path, index=False,
                                                                                  header=False)
        torch.save(iter_list, f"{File_Path_Csv}/iterList.pth")


if __name__=="__main__":

    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    main()
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start time: ", start_time)
    print("end time: ", end_time)