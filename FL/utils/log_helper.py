import os
import csv

def log_results(config, test_loss_list, test_acc_list):
    # 保留原始配置的副本以进行修改
    config_modified = config.copy()
    
    # 移除不需要记录在文件名中的配置项
    keys_to_remove = ['device', 'epochs', 'momentum', 'seed', 'save_dir', 'start_round', 'test_batch_size']
    for key in keys_to_remove:
        config_modified.pop(key, None)
    
    # 确保日志目录存在
    log_dir = './log/'
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成文件名
    config_items = [f"{key}_{value}" for key, value in config_modified.items()]
    filename = f"{'_'.join(config_items)}.csv"
    filepath = os.path.join(log_dir, filename)
    
    # 检查文件是否存在，以确定是否需要写入表头
    file_exists = os.path.isfile(filepath)
    
    # 使用追加模式打开文件，以便在文件存在时添加内容而不是覆盖
    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'test_loss', 'test_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 根据start_round调整epoch的起始值
        start_epoch = config['start_round'] + 1
        
        for epoch, (loss, acc) in enumerate(zip(test_loss_list, test_acc_list), start=start_epoch):
            writer.writerow({'epoch': epoch, 'test_loss': loss, 'test_accuracy': acc})
