from FL.utils.train_helper import load_config
from privacy_analysis.compute_dp_sgd import apply_dp_sgd_analysis
from FL.utils.log_helper import log_epsilon_opt_order_delta
from privacy_analysis.shuffle import ShuffleDP_FMT_ver

def cal_eps(epochs, q, sigma, delta, orders):
    """
    Calculate epsilon and optimal order for each epoch based on the given parameters.

    Args:
        epochs (int): The number of training epochs.
        q (float): The sampling probability for each record.
        sigma (float): The standard deviation multiplier of the Gaussian noise added during training.
        delta (float): The privacy parameter delta, which bounds the privacy loss.
        orders (list): The list of orders for each moment.

    Returns:
    - eps_list: A list of epsilon values calculated for each epoch
    - opt_order_list: A list of optimal order values calculated for each epoch
    - delta_list: A list of delta values calculated for each epoch
    """
    eps_list = []
    opt_oder_list = []
    delta_list = []
    for i in range(epochs):
        epsilon, opt_order = apply_dp_sgd_analysis(q, sigma, i + 1, orders, delta)
        eps_list.append(epsilon)
        opt_oder_list.append(opt_order)
        delta_list.append(delta)
    
    return eps_list, opt_oder_list, delta_list


def cal_eps_DPSGD(epochs, q, sigma, delta, orders):
    """
    Calculate the privacy budget (epsilon) for Differentially Private Stochastic Gradient Descent (DPSGD).

    Args:
        epochs (int): The number of training epochs.
        q (float): The sampling probability for each record.
        sigma (float): The standard deviation multiplier of the Gaussian noise added during training.
        delta (float): The privacy parameter delta, which bounds the privacy loss.
        orders (list): The list of orders for each moment.

    Returns:
    - eps_list: A list of epsilon values calculated for each epoch
    - opt_order_list: A list of optimal order values calculated for each epoch  
    - delta_list: A list of delta values calculated for each epoch
    """
    return cal_eps(epochs, q, sigma, delta, orders)


def cal_eps_FL_with_dp_per_client(epochs, q, sigma, delta, orders):
    """
    Calculate the privacy budget (epsilon) for Federated Learning with Differential Privacy (FL-DP) for each client.

    Args:
        epochs (int): The number of training epochs.
        q (float): The sampling probability for each record.
        sigma (float): The standard deviation multiplier of the Gaussian noise added during training.
        delta (float): The privacy parameter delta, which bounds the privacy loss.
        orders (list): The list of orders for each moment.

    Returns:
    - eps_list: A list of epsilon values calculated for each epoch
    - opt_order_list: A list of optimal order values calculated for each epoch
    - delta_list: A list of delta values calculated for each epoch
     """
    return cal_eps(epochs, q, sigma, delta, orders)


def cal_eps_FL_with_dp_perlayer_clip_per_client(epochs, q, sigma, delta, orders):
    """
    Calculate the privacy budget (epsilon) for Federated Learning with Differential Privacy perlayer clipping (FL-DP-perlayer-clip) for each client.

    Args:
        epochs (int): The number of training epochs.
        q (float): The sampling probability for each record.
        sigma (float): The standard deviation multiplier of the Gaussian noise added during training.
        delta (float): The privacy parameter delta, which bounds the privacy loss.
        orders (list): The list of orders for each moment.

    Returns:
    - eps_list: A list of epsilon values calculated for each epoch
    - opt_order_list: A list of optimal order values calculated for each epoch
    - delta_list: A list of delta values calculated for each epoch
    """
    return cal_eps(epochs, q, sigma, delta, orders)

def cal_eps_FL_with_dp_with_shuffler(epochs, q, sigma, delta, num_of_clients, orders):
    """
    Calculate the privacy budget (epsilon) for Federated Learning with Differential Privacy with shuffler (FL-DP-shuffler) for each client.

    Args:
        epochs (int): The number of training epochs.
        q (float): The sampling probability for each record.
        sigma (float): The standard deviation multiplier of the Gaussian noise added during training.
        delta (float): The privacy parameter delta, which bounds the privacy loss.
        orders (list): The list of orders for each moment.

    Returns:
    - eps_list: A list of epsilon values calculated for each epoch
    - opt_order_list: A list of optimal order values calculated for each epoch
    - delta_list: A list of delta values calculated for each epoch
    """
    tmp_delta = delta / num_of_clients / 10
    eps_list, opt_order_list, delta_list = cal_eps(epochs, q, sigma, tmp_delta, orders)
    new_eps_list = []
    new_delta_list = []
    for old_eps in eps_list:
        s = ShuffleDP_FMT_ver(eps0=old_eps, delta0=tmp_delta, delta=delta, num_of_clients=num_of_clients)
        new_eps = s.calculate_eps_ver2()
        new_delta = s.calculate_delta_ver2()
        new_eps_list.append(new_eps)
        new_delta_list.append(new_delta)
    return new_eps_list, opt_order_list, new_delta_list

if __name__ == '__main__':
    CONFIG_PATH = "../config.yml"
    config = load_config(CONFIG_PATH)

    orders = list(range(2, 64)) + [128, 256, 512]
    epochs = config['iters']
    q = config['q_for_batch_size']
    sigma = config['sigma']
    delta = float(config['delta'])
    num_of_clients = config['num_clients']

    if config['algorithm'] == 'DPSGD':
        eps_list, opt_order_list, delta_list = cal_eps_DPSGD(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp':
        eps_list, opt_order_list, delta_list = cal_eps_FL_with_dp_per_client(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp_perlayer':
        eps_list, opt_order_list, delta_list = cal_eps_FL_with_dp_perlayer_clip_per_client(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp_auto':
        eps_list, opt_order_list, delta_list = cal_eps_FL_with_dp_per_client(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp_with_shuffler':
        eps_list, opt_order_list, delta_list = cal_eps_FL_with_dp_with_shuffler(epochs, q, sigma, delta, num_of_clients, orders)
    else:
        raise ValueError("Unsupported algorithm name.")
    
    log_epsilon_opt_order_delta(config, eps_list, opt_order_list, delta_list)