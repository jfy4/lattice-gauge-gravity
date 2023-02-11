#!/judah/miniconda3/bin/python

import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
from metropolis import *


if __name__ == "__main__":

    # initialize lattice
    
    # parameters
    # kappa = 1.
    # lam = 1.
    # K = 1.
    # omega = 1.
    # alpha = 1.
    L = 4

    # make the levi tensors
    levi = make_levi()
    levi3 = three_levi()

    lattice = Simulation(L)
    # lattice.load_config("./k1.0_lam1.0_a1.0_K1.0_L4/fields_k1.0_lam1.0_a1.0_K1.0_L4_swp1169.hdf5")
    lattice.run(omega=1., measurement_rate=1)
    
            
            