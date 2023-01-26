#!/miniconda3/bin/python

import gpt as g
import itertools as it
import numpy as np
import copy


class Simulation:

    def __init__(self, L):
        self.L = L # symmetric lattice
        self.grid = g.grid([self.L]*4, g.double) # make the lattice
        g.message(self.grid)
        self.rng = g.random("seed string") # initialize random seed

        # make the tetrads
        self.e = [[self.rng.normal(g.real(self.grid)) for a in range(4)] for mu in range(4)]
        self.make_Us() # creates the Us
        self.make_initial_mask()
        
    def make_initial_mask(self,):
        self.starting_ones =  g.real(self.grid)
        self.starting_ones[:] = 0
        # print(self.starting_ones[0,0,0,0])
        # assert False
        nonzero_indices = range(0, self.L, 2)
        # idx_filler = list()
        for i in it.product(nonzero_indices, repeat=4):
            id0, id1, id2, id3 = i
            self.starting_ones[id0, id1, id2, id3] = 1
        # idx_filler = np.array(idx_filler)
        # assert False
        # self.starting_ones[idx_filler] = 0
        # print(g.cshift(self.starting_ones, 0, 1))
        # assert False

    def random_shift(self, scale=1.0):
        return self.rng.normal(g.real(self.grid), sigma=scale)
    
    def make_eslash(self,):
        eslash = [g.mspin(self.grid) for mu in range(4)]
        for mu in range(4):
            eslash[mu][:] = 0
        for mu in range(4):
            for a in range(4):
                eslash[mu] += g.gamma[a].tensor() * self.e[mu][a]
        return eslash

    def random_links(self, scale=1.0):
        Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
                  [a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
        lnV = g.mspin(self.grid) 
        lnV[:] = 0
        for a in range(0, 4):
            for b in range(0, 4):
                lnV += Ji2[a][b] * self.rng.normal(g.complex(self.grid), sigma=scale)
        V = g.mspin(self.grid)
        V = g.matrix.exp(lnV)
        return V

    def compute_action(self,):
        R = g.real(self.grid)
        R[:] = 0
        Rsq = g.real(self.grid)
        Rsq[:] = 0
        vol = g.real(self.grid)
        vol[:] = 0
        eslash = self.make_eslash()
        for idx, val in levi.items():
            mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
            Gmunu = g.qcd.gauge.field_strength(self.U, mu, nu)
            R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
            vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
        Rsq += R * R # g.component.pow(2)(R)
        dete = det(self.e)
        action = (sign(dete) * ((-1) * (self.kappa / 16) * R
                                + (self.lam / 96) * vol
                                + self.alpha * Rsq * g.component.inv(dete) / 64))
        return action

    
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
        return (self.lam / 96)*Vmu - (self.kappa / 16)*Wmu
        
            
    def compute_link_action(self, mu):
        R = g.real(self.grid)
        R[:] = 0
        # E, Etil = staple(links, e, mu)
        E = self.staple(mu)
        R = g.trace(self.U[mu] * E)
        return (-self.kappa / 16) * R


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
        # Mike's links
        # make log U
        lnU = [g.mspin(self.grid) for mu in range(4)]
        for i in range(4):
            lnU[i][:] = 0
        # gamma commutators
        Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
                  [a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
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

    

    def update_links(self,):
        for mu in range(4):
            action = self.compute_action()
            # action = self.compute_link_action(mu)
            # V = g.lattice(links[mu])
            V_eye = g.identity(self.U[mu])
            # g.message(V_eye)
            V = self.random_links(scale=0.1)
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
            self.U[mu] @= g.where(accept, lp, lo)
            # print(links[mu][0,0,0,0], lo[0,0,0,0], lp[0,0,0,0])
            # print('==================')
        
        
    def update_tetrads(self,):
        for mu in range(4):
            for a in range(4):
                action = self.compute_action()
                # action = self.compute_tet_action(mu)
                ii_eye = g.identity(self.e[mu][a])
                ii = self.random_shift(scale=1.)
                ii = g.where(self.mask, ii, ii_eye)
                eo = self.e[mu][a]
                dete = det(self.e)
                # print(eo[0,0,0,0])
                ep = g.eval(ii + eo)
                # print(ep[0,0,0,0])
                # print(ep, e[mu][a])
                # assert False
                # e_prime = 1
                self.e[mu][a] = g.eval(ii + eo)
                detep = det(self.e)
                # print(eo[0,0,0,0], e[mu][a][0,0,0,0])
                # print(e[mu][a][0,0,0,0], e_prime[mu][a][0,0,0,0])
                action_prime = self.compute_action()
                # action_prime = self.compute_tet_action(mu)
                # print(np.sum(g.eval(action_prime)[:]), np.sum(g.eval(action)[:]))
                meas = g.component.pow(self.K)(g.component.abs(detep) * g.component.inv(g.component.abs(dete)))
                prob = g.eval(g.component.exp(action - action_prime) * meas)
                rn = g.lattice(prob)
                self.rng.uniform_real(rn)
                accept = rn < prob
                accept *= self.mask
                self.e[mu][a] @= g.where(accept, ep, eo)
                # print(e[mu][a][0,0,0,0], eo[0,0,0,0], ep[0,0,0,0])
                # print(np.sum(g.eval(compute_tet_action(links, e, mu))[:]))
                # print('==================')
                
            
            
    def update_fields(self,):
        self.update_links()
        self.update_tetrads()

    def run(self, nswps, kappa, lam, alpha, K, crosscheck=False):
        self.crosscheck = crosscheck
        self.kappa = kappa
        self.lam = lam
        self.K = K
        self.alpha = alpha
        # need to do this masking for even odd update
        grid_eo = self.grid.checkerboarded(g.redblack)
        self.mask_rb = g.complex(grid_eo)
        self.mask_rb[:] = 1
        self.mask = g.complex(self.grid)

        self.measurements = list()
        for swp in range(nswps):
            # print(swp)
            self.sweep(swp)

    def sweep(self, swp):
        plaq = g.qcd.gauge.plaquette(self.U)
        R_2x1 = g.qcd.gauge.rectangle(self.U, 2, 1)
        the_det = np.real(np.mean(det(self.e)[:]))
        act = np.real(np.sum(g.eval(self.compute_action())[:]) / self.L**4)
        self.measurements.append([plaq, R_2x1,the_det,act])
        # act2 = np.sum(g.eval(compute_action_check(U, e))[:])
        # # act2 = g.eval(compute_action_check(U, e))[0,0,0,0]
        # act1 = g.real(grid)
        # act1[:] = 0
        # for mu in range(4):
        #     act1 += g.eval(compute_link_action(U, e, mu))
        # act1 = np.sum(act1[:])
        g.message(f"Metropolis {swp} has det = {the_det}, P = {plaq}, R_2x1 = {R_2x1}, act = {act}")
        if self.crosscheck:
            check_act = np.real(np.sum(g.eval(self.compute_action())[:]) / self.L**4)
            g.message(f"Cross check action = {check_act}")
        # act2 = np.mean(g.eval(compute_action_check(U, e))[:])
        # act1 = g.real(grid)
        # act1[:] = 0
        # for mu in range(4):
        #     act1 += g.eval(compute_tet_action(U, e, mu))
        # act1 = np.mean(act1[:])
        # g.message(f"action1 = {act1}, action2 = {act2}")
        for coord in it.product(range(2), repeat=4):
            shift0, shift1, shift2, shift3 = coord
            self.mask = g.cshift(g.cshift(g.cshift(g.cshift(self.starting_ones, 0, shift0), 1, shift1), 2, shift2), 3, shift3)
            self.update_fields()
            # assert False
        # for cb in [g.even, g.odd]:
            # self.mask[:] = 0
            # self.mask_rb.checkerboard(cb)
            # print(self.mask_rb)
            # g.set_checkerboard(self.mask, self.mask_rb)
            # print(self.mask)
            # assert False
            # self.update_fields()



        



def det(e):
    want = g.lattice(e[0][0])
    want[:] = 0
    for idx, val in levi.items():
        want += (e[0][idx[0]] * e[1][idx[1]] *
                     e[2][idx[2]] * e[3][idx[3]] *
                     val)
    return want

def sign(x):
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



if __name__ == "__main__":

    # initialize lattice
    
    # parameters
    kappa = 1.
    lam = 1.
    K = 0
    alpha = 1.
    L = 4
    nswps = 1000
    # alpha = 1
    # beta = 1

    # make the levi tensors
    levi = make_levi()
    levi3 = three_levi()

    lattice = Simulation(L)
    lattice.run(nswps, kappa, lam, alpha, K, crosscheck=False)
    
    
    np.save("measure_nswps" + str(nswps) + "_K" + str(K) +
            "_kappa" + str(kappa) + "_lam" + str(lam) +
            "_alpha" + str(alpha) + ".npy", lattice.measurements)
        
            
