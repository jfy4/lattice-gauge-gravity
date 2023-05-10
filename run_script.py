import os
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["OMP_NUM_THREADS"] = "4"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--L', required=True, type=int, help='number of spatial lattice sites')
parser.add_argument('--kappa', type=float, default=0.0)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--omega', type=float, default=1.0)
parser.add_argument('--zeta', type=float, default=1.0)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--K', type=float, default=2.0)
parser.add_argument('--mpi', default="1.1.1.1", type=str, help='mpi ranks passed to gpt')
globals().update(vars(parser.parse_args()))

from metropolis import *


if __name__ == "__main__":
    # make the levi tensors
    #levi = make_levi()
    #levi3 = three_levi()

    # initialize lattice
    lattice = Simulation(L)

    lattice.run(path="/wclustre/lqcd_gpt/gauge_configs/", kappa=kappa, lam=lam, alpha=alpha, beta=beta, gamma=gamma, omega=omega, zeta=zeta, eta=eta, K=K, measurement_rate=1)
    
            
            
