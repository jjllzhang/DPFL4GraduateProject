from FL.utils.train_helper import load_config
from privacy_analysis.compute_dp_sgd import apply_dp_sgd_analysis
from FL.utils.log_helper import log_epsilon_opt_order

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
    """
    eps_list = []
    opt_oder_list = []
    for i in range(epochs):
        epsilon, opt_order = apply_dp_sgd_analysis(q, sigma, i + 1, orders, delta)
        eps_list.append(epsilon)
        opt_oder_list.append(opt_order)
    
    return eps_list, opt_oder_list


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
    """
    return cal_eps(epochs, q, sigma, delta, orders)

if __name__ == '__main__':
    CONFIG_PATH = "../config.yml"
    config = load_config(CONFIG_PATH)

    orders = list(range(2, 64)) + [128, 256, 512]
    epochs = config['iters']
    q = config['q_for_batch_size']
    sigma = config['sigma']
    delta = float(config['delta'])

    if config['algorithm'] == 'DPSGD':
        eps_list, opt_order_list = cal_eps_DPSGD(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp':
        eps_list, opt_order_list = cal_eps_FL_with_dp_per_client(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp_perlayer':
        eps_list, opt_order_list = cal_eps_FL_with_dp_perlayer_clip_per_client(epochs, q, sigma, delta, orders)
    elif config['algorithm'] == 'fed_avg_with_dp_auto':
        eps_list, opt_order_list = cal_eps_FL_with_dp_per_client(epochs, q, sigma, delta, orders)
    else:
        raise ValueError("Unsupported algorithm name.")
    
    log_epsilon_opt_order(config, eps_list, opt_order_list)