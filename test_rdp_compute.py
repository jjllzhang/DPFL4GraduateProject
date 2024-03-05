

from privacy_analysis.compute_rdp import compute_rdp
from privacy_analysis.rdp_convert_dp import compute_eps

rdp = 0.0
q = 0.01
sigma = 1.1
orders = (list(range(2,64)) + [128, 256, 512])
for i in range(100):
    rdp += compute_rdp(q, sigma, 100, orders)
    epsilon, best_alpha = compute_eps(orders, rdp, 1e-5)
    print("Iteration: ", i, "epsilon:", epsilon, "best_alpha:", best_alpha)
