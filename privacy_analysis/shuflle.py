import math
from scipy.special import comb
from privacy_analysis.rdp_convert_dp import compute_eps


## Using the second calculation version of ShuffleDP_FMT_ver to calculate privacy budget can get the tightest bound for shufflerDP
## to get similar final delta bound, notice to adjust delta0 and delta values

class ShufflerDP:
    def __init__(self, eps0=1, delta0=1e-5, delta=1e-5, num_of_clients=100) -> None:
        self.eps0 = eps0
        self.delta0 = delta0
        self.delta = delta
        self.num_of_clients = num_of_clients

    def calculate_eps(self):
        raise NotImplemented
    
    def calculate_delta(self):
        raise NotImplemented

    def check_eps0(self):
        raise NotImplemented
    

class ShuffleDP_FMT_ver(ShufflerDP):
    """
    This version of privacy  budget calculation is based on the paper "Feldman et al. - 2021 - Hiding Among the Clones A Simple and Nearly Optim.pdf"
    """
    def __init__(self, eps0=1, delta0=1e-5, delta=1e-5, num_of_clients=100) -> None:
        super().__init__(eps0, delta0, delta, num_of_clients)


    def check_eps0(self):
        bound = math.log(self.num_of_clients / (16 * math.log(2 / self.delta)))
        if self.eps0 >= bound:
            return False
        else:
            return True
        
    
    def calculate_eps(self):
        if  not self.check_eps0:
            raise ValueError("eps0 is too large")
        tmp1 = 8 * math.sqrt(math.exp(self.eps0) * math.log(4 / self.delta)) / math.sqrt(self.num_of_clients)
        tmp2 = 8 * math.exp(self.eps0) / self.num_of_clients
        tmp3 = (math.exp(self.eps0) - 1) / (math.exp(self.eps0) + 1)
        upper_bound  = math.log(1 + tmp3 * (tmp1 + tmp2))
        return upper_bound
    

    def calculate_eps_ver2(self):
        if not self.check_eps0:
            raise ValueError("eps0 is too large")
        
        return math.exp(self.eps0 / 2) / math.sqrt(self.num_of_clients)
    
    def calculate_delta(self):
        eps = self.calculate_eps()
        tmp1 = math.exp(eps) + 1
        tmp2 = 1 + math.exp(-self.eps0) / 2
        shuffled_delta = self.delta + tmp1 * tmp2 * self.num_of_clients * self.delta0
        return shuffled_delta
    
    def calculate_delta_ver2(self):
        return self.delta + self.num_of_clients * self.delta0
    
        
    def adjust_delta_ver2(self, expected_delta):
        self.delta = expected_delta - self.num_of_clients * self.delta0
        if self.delta <= 0:
            raise ValueError("expected delta is too small")
        return self.delta
        
        
class ShuffleDP_GDDSK_ver():
    def __init__(self, alpha=10, eps0=1, delta=1e-5, num_of_clients=100) -> None:
        self.alpha = alpha
        self.eps0 = eps0
        self.delta = delta
        self.num_of_clients = num_of_clients

    def check_alpha(self):
        left = math.pow(self.alpha, 4) * math.exp(5 * self.eps0)
        right = self.num_of_clients / 9
        if left >= right:
            return False
        else:
            return True


    def calculate_renyi_eps_ver1(self):
        if  not isinstance(self.alpha, int):
            raise ValueError("alpha is not an integer")
        
        if not self.check_alpha:
            raise ValueError("alpha is too large")
        
        tmp1 = math.exp(self.eps0) - 1
        tmp2 = comb(self.alpha, 2) * 4 * tmp1 * tmp1 / self.num_of_clients
        upper_bound = math.log(1 + tmp2) / (self.alpha - 1)
        return upper_bound

    def calculate_n_bar(self):
        n_bar = math.floor((self.num_of_clients - 1) / 2 / math.exp(self.eps0)) + 1
        return n_bar
    
    def calculate_renyi_eps_ver2(self):
        tmp1 = math.exp(self.eps0) - 1
        tmp2 = tmp1 ** 2 / self.calculate_n_bar()
        # tmp3 = math.exp(self.alpha ** 2 * tmp2)
        tmp4 = math.exp(self.alpha * self.eps0 - (self.num_of_clients - 1) / 8 / math.exp(self.eps0))
        upper_bound = math.log(math.exp(self.alpha ** 2 * tmp2) + tmp4) / (self.alpha - 1)
        return upper_bound
    
    def calculate_renyi_eps_RDP(self):
        tmp1 = math.exp(self.eps0) - 1
        tmp2 = math.exp(4 * self.eps0) * tmp1 * tmp1 / self.num_of_clients
        upper_bound = self.alpha * 2 * tmp2
        return upper_bound


def test():
    s = ShuffleDP_FMT_ver(eps0=2, delta0=1e-7, delta=1e-7, num_of_clients=10)
    print(s.calculate_eps())
    print(s.calculate_delta())
    print(s.calculate_eps_ver2())
    print(s.calculate_delta_ver2())
    print("Test RDP")
    s2 = ShuffleDP_GDDSK_ver(alpha=10, eps0=1.5, delta=1e-5, num_of_clients=10)
    print(compute_eps(orders=[10], rdp=[s2.calculate_renyi_eps_ver1()], delta=1e-5))
    print(compute_eps(orders=[10], rdp=[s2.calculate_renyi_eps_ver2()], delta=1e-5))
    print(compute_eps(orders=[10], rdp=[s2.calculate_renyi_eps_RDP()], delta=1e-5))

test()