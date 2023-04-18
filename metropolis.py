#!/miniconda3/bin/python

import os
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["OMP_NUM_THREADS"] = "4"
import gpt as g
import itertools as it
import numpy as np
import copy
import h5py


class Simulation:

    def __init__(self, L):
        """
        Initialize a Metropolis simulation.
        """
        self.L = L # symmetric lattice
        self.grid = g.grid([self.L]*4, g.double) # make the lattice
        self.link_acpt = [0]*100
        self.tet_acpt = [0]*100
        self.load = False
        g.message(self.grid)
        # self.rng = g.random("seed string") # initialize random seed
        self.rng = g.random("0") # initialize random seed

        # make the tetrads
        self.e = [[self.rng.normal(g.real(self.grid)) for a in range(4)] for mu in range(4)]
        # make the Us
        self.make_Us()
        # make the checkerboard mask
        self.make_initial_mask()


    def load_config(self, fields_path, swp_number):
        """Load saved gauge and tetrad fields."""
        self.load = True
        fields = h5py.File(fields_path, 'r')
        swp = swp_number

        self.kappa = fields.attrs['kappa']
        self.lam = fields.attrs['lambda']
        self.alpha = fields.attrs['alpha']
        self.K = fields.attrs['K']
        self.omega = fields.attrs['omega']
        self.zeta = fields.attrs['zeta']
        self.eta = fields.attrs['eta']
        self.swp_count = swp
        self.rng = g.random(str(self.swp_count)) # initialize random seed
        assert self.L == fields.attrs['L']
        
        for mu in range(4):
            self.U[mu][:] = fields['sweep'][str(swp)]['gauge'][str(mu)][:]
            for a in range(4):
                self.e[mu][a][:] = fields['sweep'][str(swp)]['tetrad'][str(mu)][str(a)][:] + 0j
        g.message(f"Loaded config. Sweep count = {self.swp_count}, L = {self.L}, kappa = {self.kappa}, lambda = {self.lam}, alpha = {self.alpha}, K = {self.K}, omega = {self.omega}, zeta = {self.zeta}, eta = {self.eta}")
        fields.close()
        

    def save_config(self, path):
        """ Save field configurations."""
        kappa = str(np.round(self.kappa, 8))
        lam = str(np.round(self.lam, 8))
        alpha = str(np.round(self.alpha, 8))
        K = str(np.round(self.K, 8))
        omega = str(np.round(self.omega, 8))
        zeta = str(np.round(self.zeta, 8))
        eta = str(np.round(self.eta, 8))
        # current_path = ("./k" + kappa + "_lam" + lam
        #                 + "_a" + alpha + "_K" + K + "_o" + omega +
        #                 "_z" + zeta + "_e" + eta +
        #                 "_L" + str(self.L) + "/")
        # try:
        #     os.mkdir(current_path)
        # except FileExistsError:
        #     pass
        f = h5py.File(path + "fields_k" + kappa + "_lam" + lam
                      + "_a" + alpha + "_K" + K + "_o" + omega +
                      "_z" + zeta + "_e" + eta + "_L" + str(self.L)
                      + ".hdf5", 'a')
        if self.swp_count == 0:
            f.attrs['kappa'] = self.kappa
            f.attrs['lambda'] = self.lam
            f.attrs['alpha'] = self.alpha
            f.attrs['K'] = self.K
            f.attrs['omega'] = self.omega
            f.attrs['zeta'] = self.zeta
            f.attrs['eta'] = self.eta
            # f.attrs['sweep'] = self.swp_count
            f.attrs['L'] = self.L
        for mu in range(4):
            f.create_dataset("sweep/" + str(self.swp_count) + "/gauge/" + str(mu), data=self.U[mu][:])
            for a in range(4):
                f.create_dataset("sweep/" + str(self.swp_count) + "/tetrad/" + str(mu) + "/" + str(a), data=np.real(self.e[mu][a][:]))
        R, dete = self.compute_obs()
        f.create_dataset("sweep/" + str(self.swp_count) + "/obs/R", data=np.real(R[:]))
        f.create_dataset("sweep/" + str(self.swp_count) + "/obs/dete", data=np.real(dete[:]))
        f.close()


            
    def inverse_tetrad(self, ):
        """ compute the inverse tetrad."""
        edet = det(self.e)
        einv = [[g.real(self.grid) for a in range(4)] for mu in range(4)]
        inverse_dete = g.component.inv(edet)
        for a in range(4):
            for mu in range(4):
                einv[a][mu][:] = 0
        for idx_a, val_a in levi.items():
            for idx_mu, val_mu in levi.items():
                einv[idx_a[0]][idx_mu[0]] += (1/6. * self.e[idx_mu[1]][idx_a[1]]
                                              * self.e[idx_mu[2]][idx_a[2]] * self.e[idx_mu[3]][idx_a[3]]
                                              * val_a * val_mu) * inverse_dete
        return einv

                
    def make_initial_mask(self,):
        """
        Makes a checkerboard mask such that every other site in
        every direction is revealed.  Note this is not the bi-partite
        split.
        """
        self.starting_ones =  g.real(self.grid)
        self.starting_ones[:] = 0
        nonzero_indices = range(0, self.L, 2)
        for i in it.product(nonzero_indices, repeat=4):
            id0, id1, id2, id3 = i
            self.starting_ones[id0, id1, id2, id3] = 1

    def random_shift(self, scale=1.0):
        """ Create random numbers from the normal distribution."""
        return self.rng.normal(g.real(self.grid), sigma=scale)
    
    def make_eslash(self,):
        """ Make the slashed es."""
        eslash = [g.mspin(self.grid) for mu in range(4)]
        for mu in range(4):
            eslash[mu][:] = 0
        for mu in range(4):
            for a in range(4):
                eslash[mu] += g.gamma[a].tensor() * self.e[mu][a]
        return eslash

    def make_einvslash(self,):
        """ Make the slashed version of the inverse tetrads."""
        eslash = [g.mspin(self.grid) for mu in range(4)]
        einv = self.inverse_tetrad()
        for mu in range(4):
            eslash[mu][:] = 0
        for mu in range(4):
            for a in range(4):
                eslash[mu] += g.gamma[a].tensor() * einv[a][mu]
        return eslash

    def make_ginv(self,):
        """ make the inverse metric."""
        einvslash = self.make_einvslash()
        ginv = [[g.real(self.grid) for mu in range(4)] for nu in range(4)]            
        for mu, nu in it.product(range(4), repeat=2):
            ginv[mu][nu] = g.trace(einvslash[mu] * einvslash[nu]) / 4
        return ginv

    def random_links(self, scale=1.0):
        """ Make a lattice of random link variables."""
        Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
                  [a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
        lnV = g.mspin(self.grid) 
        lnV[:] = 0
        for a in range(0, 4):
            for b in range(0, 4):
                lnV += Ji2[a][b] * self.rng.normal(g.complex(self.grid), sigma=scale)
        V = g.mspin(self.grid)
        V = g.matrix.exp(lnV)
        del lnV, Ji2
        return V

    def build_Bmunu_squared(self,):
        """Compute Bmunu squared."""
        ricci = self.make_ricci()
        ginv = self.make_ginv()
        temp1 = [[g.real(self.grid) for mu in range(4)] for nu in range(4)]
        temp2 = [[g.real(self.grid) for mu in range(4)] for nu in range(4)]
        riccisq = g.real(self.grid)
        riccisq[:] = 0
        riccitwist = g.real(self.grid)
        riccitwist[:] = 0
        for mu, nu in it.product(range(4), repeat=2):
            temp1[mu][nu][:] = 0
            temp2[mu][nu][:] = 0
        for mu, nu, sig in it.product(range(4), repeat=3):
            temp1[sig][nu] += ricci[mu][nu] * ginv[mu][sig]
        for mu, nu, sig in it.product(range(4), repeat=3):
            temp2[mu][sig] += temp1[mu][nu] * ginv[nu][sig]
        for mu, nu in it.product(range(4), repeat=2):
            riccisq += ricci[mu][nu] * temp2[mu][nu]
            riccitwist += ricci[mu][nu] * temp2[nu][mu]
        Bsmallsq = 2 * riccisq - 2*riccitwist
        return Bsmallsq

    def symmetric_clover(self, U, mu, nu):
        """ Create the symmetric clover, H, from the notes."""
        assert mu != nu
        # v = staple_up + staple_down
        v = g.eval(
            g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu])
            + g.cshift(g.adj(g.cshift(U[nu], mu, 1)) * g.adj(U[mu]) * U[nu], nu, -1)
        )
        
        F = g.eval(U[mu] * v + g.cshift(v * U[mu], mu, -1))
        F @= 0.125 * (F + g.adj(F))
        return F
            
    def make_ricci(self,):
        """ Make the Ricci curvature tensor."""
        eslash = self.make_eslash()
        Ricci = [[g.real(self.grid) for mu in range(4)] for nu in range(4)]
        for mu, nu in it.product(range(4), repeat=2):
            Ricci[mu][nu][:] = 0
        einvslash = self.make_einvslash()
        for sig, mu, nu in it.product(range(4), repeat=3):
            if sig == mu:
                continue
            Gsigmu = g.qcd.gauge.field_strength(self.U, sig, mu)
            Ricci[mu][nu] += -1 * g.trace(Gsigmu * einvslash[sig] * eslash[nu]) / 8
        return Ricci

    def check_R(self,):
        Rcheck = g.real(self.grid)
        Rcheck[:] = 0
        Rcheck2 = g.real(self.grid)
        Rcheck2[:] = 0
        Rcheck3 = g.real(self.grid)
        Rcheck3[:] = 0
        R = g.real(self.grid)
        R[:] = 0
        Riemann, Riemann_up = self.make_riemann()
        eslash = self.make_eslash()
        ginv = self.make_ginv()
        metric = [[g.real(self.grid) for mu in range(4)] for nu in range(4)]
        for mu, nu in it.product(range(4), repeat=2):
            metric[mu][nu] = g.trace(eslash[mu] * eslash[nu]) / 4
        for mu,nu,rho,sig in it.product(range(4), repeat=4):
            Rcheck += Riemann_up[mu][nu][rho][sig] * metric[mu][rho] * metric[nu][sig]
            Rcheck2 += Riemann[mu][nu][rho][sig] * ginv[mu][rho] * ginv[nu][sig]
        ricci = self.make_ricci()
        for mu, nu in it.product(range(4), repeat=2):
            Rcheck3 += ricci[mu][nu] * ginv[mu][nu]
        print("Rcheck", Rcheck[0,0,0,0])
        print("Rcheck2", Rcheck2[0,0,0,0])
        print("Rcheck3", Rcheck3[0,0,0,0])
        for idx, val in levi.items():
            mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
            Gmunu = g.qcd.gauge.field_strength(self.U, mu, nu)
            R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
        R /= 16
        dete = det(self.e)
        R *= g.component.inv(dete)
        print("real R", R[0,0,0,0])
        assert False
            
            
    def make_riemann(self,):
        """ Make the Riemann curvature tensor."""
        eslash = self.make_eslash()
        einvslash = self.make_einvslash()
        Riemann = [[[[g.real(self.grid) for mu in range(4)] for nu in range(4)]
                    for rho in range(4)] for sig in range(4)]
        Riemann_up = [[[[g.real(self.grid) for mu in range(4)] for nu in range(4)]
                    for rho in range(4)] for sig in range(4)]
        ginv = self.make_ginv()
        G_up = [[g.lattice(self.U[0]) for mu in range(4)] for nu in range(4)]
        temp1 = [[g.lattice(self.U[0]) for mu in range(4)] for nu in range(4)]
        for mu, nu in it.product(range(4), repeat=2):
            G_up[mu][nu][:] = 0
            temp1[mu][nu][:] = 0
        for mu, nu, sig in it.product(range(4), repeat=3):
            if sig == nu:
                continue
            temp1[mu][nu] += ginv[mu][sig] * g.qcd.gauge.field_strength(self.U, sig, nu)
        for mu, nu, sig in it.product(range(4), repeat=3):
            G_up[mu][nu] +=  temp1[mu][sig] * ginv[sig][nu]
        for mu,nu,rho,sig in it.product(range(4), repeat=4):
            Riemann[sig][mu][rho][nu][:] = 0
            Riemann_up[sig][mu][rho][nu][:] = 0
            if sig == mu:
                continue
            if rho == nu:
                continue
            Gsigmu = g.qcd.gauge.field_strength(self.U, sig, mu)
            Riemann[sig][mu][rho][nu] @= (-1 * g.trace(Gsigmu * eslash[rho] * eslash[nu]) / 8)
            Riemann_up[sig][mu][rho][nu] @= (-1 * g.trace(G_up[sig][mu] * einvslash[rho] * einvslash[nu]) / 8)
        return (Riemann, Riemann_up)


    def build_Bmunurhosig_squared(self,):
        riemann, riemann_up = self.make_riemann()
        BigBsquared = g.real(self.grid)
        BigBsquared[:] = 0
        for mu, nu, rho, sig in it.product(range(4), repeat=4):
            if (mu == nu):
                continue
            if (rho == sig):
                continue
            BigBsquared += (3 * riemann[mu][nu][rho][sig] * (riemann_up[mu][nu][rho][sig]
                                                             +
                                                             2 * riemann_up[mu][rho][sig][nu]))
        return BigBsquared
        
    
    def compute_action(self,):
        """ Compute the gravity action site-wise."""
        R = g.real(self.grid)
        R[:] = 0
        Rsq = g.real(self.grid)
        Rsq[:] = 0
        vol = g.real(self.grid)
        vol[:] = 0
        eslash = self.make_eslash()
        wilson = g.real(self.grid)
        wilson[:] = 0
        smallB = self.build_Bmunu_squared()
        bigB = self.build_Bmunurhosig_squared()
        for mu, nu in it.product(range(4), repeat=2):
            if mu == nu:
                continue
            Hmunu = self.symmetric_clover(self.U, mu, nu)
            wilson += g.trace(g.identity(Hmunu) - Hmunu)
        for idx, val in levi.items():
            mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
            Gmunu = g.qcd.gauge.field_strength(self.U, mu, nu)
            R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
            vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
        Rsq += R * R # g.component.pow(2)(R)
        dete = det(self.e)
        absdete = g.component.abs(dete)
        wilson *= absdete
        smallB *= absdete
        bigB *= absdete
        meas = g.component.log(absdete)
        action = (sign(dete) * ((self.lam / 96) * vol
                                -(self.kappa / 32) * R
                                + (self.alpha * Rsq * g.component.inv(dete) / 256))
                  - (self.K * meas)
                  + (self.omega * wilson)
                  + (self.eta * bigB)
                  + (self.zeta * smallB)
                  )
        del R, Rsq, vol, eslash, dete, meas, wilson, Hmunu, bigB, smallB
        return action

    def compute_obs(self,):
        """ Compute the R and dete."""
        R = g.real(self.grid)
        R[:] = 0
        eslash = self.make_eslash()
        for idx, val in levi.items():
            mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
            Gmunu = g.qcd.gauge.field_strength(self.U, mu, nu)
            R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
        R /= 16
        dete = det(self.e)
        R *= g.eval(g.component.inv(dete))
        dete = g.eval(dete)
        return (R, dete)
        
    
    
    def staple(self, mu):
        Emu = g.mspin(self.grid)
        Emu[:] = 0
        Emutilde = g.mspin(self.grid)
        Emutilde[:] = 0
        eslash = self.make_eslash()
        sign_x_plus_mu = g.cshift(sign(det(self.e)), mu, 1)
        # print(eslash[mu])
        # assert False
        for (nu,rho,sig), val in levi3[mu].items():
            # print(nu,rho,sig, val)
            sign_x_minus_nu = g.cshift(sign(det(self.e)), nu, -1)
            sign_x_plus_nu = g.cshift(sign(det(self.e)), nu, 1)
            
            e_rho_x_plus_mu = g.cshift(eslash[rho], mu, 1)
            e_sig_x_plus_mu = g.cshift(eslash[sig], mu, 1)
            e_rho_x_plus_nu = g.cshift(eslash[rho], nu, 1)
            e_sig_x_plus_nu = g.cshift(eslash[sig], nu, 1)
            e_rho_x_minus_nu = g.cshift(eslash[rho], nu, -1)
            e_sig_x_minus_nu = g.cshift(eslash[sig], nu, -1)
            
            U_nu_x_plus_mu = g.cshift(self.U[nu], mu, 1)
            U_nu_x_minus_nu = g.cshift(self.U[nu], nu, -1)
            U_mu_x_plus_nu = g.cshift(self.U[mu], nu, 1)
            
            one = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
                         g.adj(self.U[nu]) * eslash[rho] * eslash[sig] *
                         g.gamma[5]) * sign(det(self.e))
            two = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                         g.adj(g.cshift(self.U[mu], nu, -1)) * U_nu_x_minus_nu *
                         eslash[rho] * eslash[sig] * g.gamma[5]) * sign(det(self.e))
            three = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
                           g.gamma[5] * U_nu_x_plus_mu *
                           g.adj(U_mu_x_plus_nu) * g.adj(self.U[nu])) * sign_x_plus_mu
            four = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
                          g.gamma[5] * g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                          g.adj(g.cshift(self.U[mu], nu, -1)) *
                          U_nu_x_minus_nu) * sign_x_plus_mu
            five = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
                          e_rho_x_plus_nu * e_sig_x_plus_nu *
                          g.gamma[5] * g.adj(self.U[nu])) * sign_x_plus_nu
            six = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                         g.adj(g.cshift(self.U[mu], nu, -1)) *
                         e_rho_x_minus_nu * e_sig_x_minus_nu *
                         g.gamma[5] * U_nu_x_minus_nu) * sign_x_minus_nu
            seven = g.eval(U_nu_x_plus_mu * g.cshift(e_rho_x_plus_mu, nu, 1) *
                           g.cshift(e_sig_x_plus_mu, nu, 1) * g.gamma[5] *
                           g.adj(U_mu_x_plus_nu) *
                           g.adj(self.U[nu])) * g.cshift(sign_x_plus_mu, nu, 1)
            eight = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                           g.cshift(e_rho_x_plus_mu, nu, -1) *
                           g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
                           g.adj(g.cshift(self.U[mu], nu, -1)) *
                           U_nu_x_minus_nu) * g.cshift(sign_x_plus_mu, nu, -1)
            Emu += 0.125 * val * (one - two + three - four + five - six + seven - eight)
            # Emutilde part
            # one = g.eval(eslash[rho] * eslash[sig] * g.gamma[5] *
            #        links[nu] * U_mu_x_plus_nu * g.adj(U_nu_x_plus_mu)) * sign(det(e))
            # two = g.eval(eslash[rho] * eslash[sig] * g.gamma[5] *
            #        g.adj(U_nu_x_minus_nu) * g.cshift(links[mu], nu, -1) *
            #        g.cshift(U_nu_x_plus_mu, nu, -1)) * sign(det(e))
            # three = g.eval(links[nu] * U_mu_x_plus_nu * g.adj(U_nu_x_plus_mu) *
            #          e_rho_x_plus_mu * e_sig_x_plus_mu
            #          * g.gamma[5]) * sign_x_plus_mu
            # four = g.eval(g.adj(U_nu_x_minus_nu) * g.cshift(links[mu], nu, -1) *
            #         g.cshift(U_nu_x_plus_mu, nu, -1) * e_rho_x_plus_mu *
            #         e_sig_x_plus_mu * g.gamma[5]) * sign_x_plus_mu
            # five = g.eval(links[nu] * e_rho_x_plus_nu * e_sig_x_plus_nu
            #         * g.gamma[5] * U_mu_x_plus_nu *
            #         g.adj(U_nu_x_plus_mu)) * sign_x_plus_nu
            # six = g.eval(g.adj(U_nu_x_minus_nu) * e_rho_x_minus_nu *
            #        e_sig_x_minus_nu * g.gamma[5] * g.cshift(links[mu], nu, -1) *
            #        g.cshift(U_nu_x_plus_mu, nu, -1)) * sign_x_minus_nu
            # seven = g.eval(links[nu] * U_mu_x_plus_nu * g.cshift(e_rho_x_plus_nu, mu, 1) *
            #          g.cshift(e_sig_x_plus_nu, mu, 1) * g.gamma[5] *
            #          g.adj(U_nu_x_plus_mu)) * g.cshift(sign_x_plus_mu, nu, 1)
            # eight = g.eval(g.adj(U_nu_x_minus_nu) * g.cshift(links[mu], nu, -1) *
            #          g.cshift(e_rho_x_plus_mu, nu, -1) *
            #          g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
            #          g.cshift(U_nu_x_plus_mu, nu, -1)) * g.cshift(sign_x_plus_mu, nu, -1)
            # Emutilde += 0.125 * val * (- one + two - three + four - five + six - seven + eight)
        # return (Emu, Emutilde)
        return Emu

    def eenv(self, eslash, mu):
        Vmu = g.mspin(self.grid)
        Vmu[:] = 0
        Wmu = g.mspin(self.grid)
        Wmu[:] = 0
        for (nu, rho, sig), val in levi3[mu].items():
            Vmu += eslash[nu] * eslash[rho] * eslash[sig] * g.gamma[5] * val
            Wmu += (eslash[nu] * g.gamma[5] *
                    g.qcd.gauge.field_strength(self.U, rho, sig) * val)
        return (self.lam / 96)*Vmu - (self.kappa / 32)*Wmu
        
            
    def compute_link_action(self, mu):
        R = g.real(self.grid)
        R[:] = 0
        # E, Etil = staple(links, e, mu)
        E = self.staple(mu)
        R = g.trace(self.U[mu] * E)
        return (-self.kappa / 32) * R


    def compute_tet_action(self, mu):
        V = g.real(self.grid)
        V[:] = 0
        eslash = self.make_eslash()
        F = self.eenv(eslash, mu)
        V = sign(det(self.e)) * g.trace(eslash[mu] * F)
        return V

    def compute_total_action(self,):
        want = g.real(self.grid)
        want[:] = 0
        for mu in range(4):
            B = self.compute_tet_action(mu)
            want += B
        return want

    def make_Us(self,):
        """ Make random link variables in SU(2)xSU(2)."""
        # Mike's links
        # make log U
        lnU = [g.mspin(self.grid) for mu in range(4)]
        for i in range(4):
            lnU[i][:] = 0
        # gamma commutators
        Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() -
                  g.gamma[b].tensor()*g.gamma[a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
        omega = [ [ [ self.rng.normal(g.complex(self.grid)) for b in range(0,4)]
                    for a in range(0,4) ] for mu in range(0, 4) ]
        for mu in range(0, 4):
            for a in range(0, 4):
                for b in range(0, 4):
                    lnU[mu] += Ji2[a][b]*omega[mu][a][b]
        # the Us
        self.U = [ g.mspin(self.grid) for mu in range(0,4) ]
        for mu in range(0,4):
            self.U[mu] = g.matrix.exp(lnU[mu])
        del lnU, Ji2, omega

    

    def update_links(self,):
        """ Metropolis update for the link variables."""
        for mu in range(4):
            action = self.compute_action()
            # action = self.compute_link_action(mu)
            # V = g.lattice(links[mu])
            V_eye = g.identity(self.U[mu])
            # g.message(V_eye)
            V = self.random_links(scale=self.Uinc)
            # g.message(V)
            V = g.where(self.mask, V, V_eye)
            lo = self.U[mu]
            # links_prime = links.copy() # copy links?
            # g.message(links_prime)
            lp = g.eval(V * lo)
            self.U[mu] = g.eval(V * lo)
            # links_prime[mu] = g.eval(V * links[mu])
            action_prime = self.compute_action()
            # action_prime = self.compute_link_action(mu)
            prob = g.component.exp(action - action_prime)
            # g.message(prob)
            rn = g.lattice(prob)
            self.rng.uniform_real(rn)
            accept = rn < prob
            accept *= self.mask
            self.link_acpt.pop()
            acpt_amount = np.real(np.sum(accept[:]) / np.sum(self.starting_ones[:]))
            self.link_acpt.insert(0, acpt_amount)
            self.U[mu] @= g.where(accept, lp, lo)
        del lp, lo, V, V_eye, action, action_prime, prob, rn, accept
            # print(links[mu][0,0,0,0], lo[0,0,0,0], lp[0,0,0,0])
            # print('==================')
        
        
    def update_tetrads(self,):
        """ Metropolis update for the tetrad variables."""
        for mu in range(4):
            for a in range(4):
                action = self.compute_action()
                # action = self.compute_tet_action(mu)
                ii_eye = g.lattice(self.e[mu][a])
                ii_eye[:] = 0
                ii = self.random_shift(scale=self.einc)
                ii = g.where(self.mask, ii, ii_eye)
                # print(ii[:])
                # assert False
                eo = self.e[mu][a]
                # dete = det(self.e)
                # print(eo[0,0,0,0])
                ep = g.eval(ii + eo)
                # print(ep[0,0,0,0])
                # print(ep, e[mu][a])
                # assert False
                # e_prime = 1
                self.e[mu][a] = g.eval(ii + eo)
                # detep = det(self.e)
                # print(eo[0,0,0,0], e[mu][a][0,0,0,0])
                # print(e[mu][a][0,0,0,0], e_prime[mu][a][0,0,0,0])
                action_prime = self.compute_action()
                # action_prime = self.compute_tet_action(mu)
                # print(np.sum(g.eval(action_prime)[:]), np.sum(g.eval(action)[:]))
                # meas = g.component.pow(self.K)(g.component.abs(detep) * g.component.inv(g.component.abs(dete)))
                # prob = g.eval(g.component.exp(action - action_prime) * meas)
                prob = g.eval(g.component.exp(action - action_prime))
                rn = g.lattice(prob)
                self.rng.uniform_real(rn)
                accept = rn < prob
                accept *= self.mask
                self.tet_acpt.pop()
                acpt_amount = np.real(np.sum(accept[:]) / np.sum(self.starting_ones[:]))
                self.tet_acpt.insert(0, acpt_amount)
                self.e[mu][a] @= g.where(accept, ep, eo)
        del action, ii_eye, ii, eo, ep, action_prime, prob, rn, accept
                # print(e[mu][a][0,0,0,0], eo[0,0,0,0], ep[0,0,0,0])
                # print(np.sum(g.eval(compute_tet_action(links, e, mu))[:]))
                # print('==================')


            
            
    def update_fields(self,):
        """ Update the links and the tetrads."""
        self.update_links()
        self.update_tetrads()

    def run(self, path="./", kappa=1., lam=1., alpha=1., K=1., omega=1., zeta=1., eta=1., measurement_rate=20, uacpt_rate=0.6, eacpt_rate=0.6):
        """ Runs the Metropolis algorithm."""
        self.Uinc = 0.4
        self.einc = 0.4
        self.du_step = 0.01
        self.de_step = 0.01
        self.target_u_acpt = uacpt_rate
        self.target_e_acpt = eacpt_rate
        self.meas_rate = measurement_rate

        if self.load:
            pass
        else:
            self.swp_count = 0
            self.kappa = np.float64(kappa)
            self.lam = np.float64(lam)
            self.K = np.float64(K)
            self.alpha = np.float64(alpha)
            self.omega = np.float64(omega)
            self.zeta = np.float64(zeta)
            self.eta = np.float64(eta)
            g.message(f"Sweep count = {self.swp_count}, L = {self.L}, kappa = {self.kappa}, lambda = {self.lam}, alpha = {self.alpha}, K = {self.K}, omega = {self.omega}, zeta = {self.zeta}, eta = {self.eta}")
            # self.save_config(path)
        # self.check_R()
        while True:
            self.sweep()
            self.swp_count += 1
            if (self.swp_count % self.meas_rate == 0):
                # self.save_config(path)
                pass

    def sweep(self,):
        """ Performs a single sweep of the lattice for the links and tetrads."""
        plaq = g.qcd.gauge.plaquette(self.U)
        R_2x1 = g.qcd.gauge.rectangle(self.U, 2, 1)
        the_det = np.real(np.mean(det(self.e)[:]))
        act = np.real(np.sum(g.eval(self.compute_action())[:]) / self.L**4)
        link_acceptance = np.real(np.mean(self.link_acpt))
        tet_acceptance = np.real(np.mean(self.tet_acpt))
        if abs(link_acceptance - self.target_u_acpt) < 0.02:
            pass
        elif link_acceptance < self.target_u_acpt:
            self.Uinc -= self.du_step
        else:
            self.Uinc += self.du_step
            
        if abs(tet_acceptance - self.target_e_acpt) < 0.02:
            pass
        elif tet_acceptance < self.target_e_acpt:
            self.einc -= self.de_step
        else:
            self.einc += self.de_step
        g.message(f"Metropolis {self.swp_count} has det = {the_det}, P = {plaq}, R_2x1 = {R_2x1}, act = {act}")
        g.message(f"Metropolis {self.swp_count} has link acceptance = {link_acceptance}, and tetrad acceptance = {tet_acceptance}")
        g.message(f"Metropolis {self.swp_count} has link step = {self.Uinc}, and tetrad step = {self.einc}")
        # self.check = g.real(self.grid)
        # self.check[:] = 0
        for coord in it.product(range(2), repeat=4):
            shift0, shift1, shift2, shift3 = coord
            self.mask = g.cshift(g.cshift(g.cshift(g.cshift(self.starting_ones, 0, shift0), 1, shift1), 2, shift2), 3, shift3)
            self.update_fields()
            # self.check += self.mask
            # print(np.sum(self.check[:]), 4**4)
            # assert False



        



def det(e):
    """ Computes the determinant of the tetrad."""
    want = g.lattice(e[0][0])
    want[:] = 0
    for idx, val in levi.items():
        want += (e[0][idx[0]] * e[1][idx[1]] *
                     e[2][idx[2]] * e[3][idx[3]] *
                     val)
    return want

def sign(x):
    """ Computes the sign of a lattice object."""
    y = g.lattice(x)
    y[:] = 0
    plus = g.lattice(x)
    plus[:] = 1
    minu = g.lattice(x)
    minu[:] = -1
    check = x > y
    the_sign = g.where(check, plus, minu)
    return the_sign



def make_levi():
    """ Makes the 4d Levi-Civita tensor."""
    arr = dict()
    for i,j,k,l in it.product(range(4), repeat=4):
        prod = (i-j)*(i-k)*(i-l)*(j-k)*(j-l)*(k-l)
        if prod == 0:
            continue
        else:
            if prod > 0:
                arr[(i,j,k,l)] = 1
            else:
                arr[(i,j,k,l)] = -1
    return arr

def three_levi():
    """ Makes the 4d Levi-Civita tensor with one index fixed.""" 
    arr = {i:dict() for i in range(4)}
    for i,j,k,l in it.product(range(4), repeat=4):
        prod = (i-j)*(i-k)*(i-l)*(j-k)*(j-l)*(k-l)
        if prod == 0:
            continue
        else:
            if prod > 0:
                arr[i][(j,k,l)] = 1
            else:
                arr[i][(j,k,l)] = -1
    return arr

levi = make_levi()
levi3 = three_levi()


