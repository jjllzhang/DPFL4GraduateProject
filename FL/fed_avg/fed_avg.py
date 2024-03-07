from FL.utils.train_helper import load_model_and_optimizers, save_model_and_optimizers
from data.utils.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL.utils.create_client import create_clients
from FL.utils.update_model import send_center_model_to_clients, set_center_model_with_weights
from FL.utils.local_train import local_clients_train_without_dp_one_epoch
from train_and_validation.validation import validation
import os
import torch


def fed_avg(train_data, test_data, test_batchsize, num_of_clients, lr, momentum, num_epoch, iters, alpha, seed, q, model, device, start_round=0, save_dir=None):
    if save_dir is None:
        # 设置默认的save_dir为当前工作目录下的saved_states文件夹
        save_dir = os.path.join(os.getcwd(), 'saved_states')
    # 检查save_dir目录是否存在，如果不存在，则创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    clients_data_list, weight_of_each_client, batchsize_of_each_client = fed_dataset_NonIID_Dirichlet(train_data, num_of_clients, alpha, seed, q)
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_clients(num_of_clients, lr, model)
    center_model = model.to(device)
    
    # 根据start_round自动生成加载路径，并检查文件是否存在
    load_path = f"{save_dir}/model_optimizers_state_round_{start_round}_fed_avg.pt"
    if start_round > 0 and os.path.exists(load_path):
        load_model_and_optimizers(center_model, clients_optimizer_list, load_path)

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batchsize, shuffle=False)
    print("Start Federated Learning without DP")
    test_acc_list = []
    test_loss_list = []
    for i in range(iters):
        current_round = i + start_round  # 真正的所在轮数
        print(f"Round: {current_round + 1}")
        clients_model_list = send_center_model_to_clients(center_model, clients_model_list)
        local_clients_train_without_dp_one_epoch(num_of_clients, clients_data_list, clients_model_list, clients_criterion_list, clients_optimizer_list, num_epoch, q, device)
        center_model = set_center_model_with_weights(center_model, clients_model_list, weight_of_each_client)
        test_loss, test_acc = validation(center_model, test_data_loader, device)
        print(f"Iteration: {current_round + 1}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f} %")
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        
        # 每训练100次后保存模型和所有客户端优化器的状态
        if (current_round + 1) % 100 == 0:
            save_path = f"{save_dir}/model_optimizers_state_round_{current_round + 1}_fed_avg.pt"
            save_model_and_optimizers(center_model, clients_optimizer_list, save_path)

    record = [test_loss_list, test_acc_list]
    return record


