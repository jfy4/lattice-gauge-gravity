#!/miniconda3/bin/python

import gpt as g
import itertools as it
import numpy as np
import sys, os
import copy
from gpt.core.group import differentiable_functional


def make_eslash(e):
    eslash = [g.mspin(grid) for mu in range(4)]
    for mu in range(4):
        eslash[mu][:] = 0
    for mu in range(4):
        for a in range(4):
            eslash[mu] += g.gamma[a].tensor() * e[mu][a]
    return eslash





class Gbase(differentiable_functional):
    def Ugradient(self, U, e, dU):
        # Eq. (1.3) and Appendix A of https://link.springer.com/content/pdf/10.1007/JHEP08(2010)071.pdf
        # S(Umu) = -2/g^2 Re trace(Umu * staple)
        # dS(Umu) = lim_{eps->0} Ta ( S(e^{eps Ta} Umu) - S(Umu) ) / eps  with  \Tr[T_a T_b]=-1/2 \delta_{ab}
        # dS(Umu) = -2/g^2 T_a Re trace(T_a * Umu * staple)
        #         = -2/g^2 T_a 1/2 trace(T_a * Umu * staple + adj(staple) * adj(Umu) * adj(Ta))
        #         = -2/g^2 T_a 1/2 trace(T_a * (Umu * staple - adj(staple) * adj(Umu)))
        #         = -2/g^2 T_a 1/2 trace(T_a * (Umu * staple - adj(Umu*staple)))
        #         = -2/g^2 T_a trace(T_a * r0)    with r0 = 1/2(Umu * staple - adj(Umu*staple))
        # r0 = c_a T_a + imaginary_diagonal   with A^dag = -A
        # trace(T_a * r0) = -1/2 c_a
        # dS(Umu) = 1/g^2 tracelss_anti_hermitian(Umu * staple)
        # define staple here as adjoint
        dS = []
        for Umu in dU:
            mu = U.index(Umu)
            dSdU_mu = self.staple(U, e, mu)
            # NOTE why is there an adj in (qcd version of) U * staple?
            # dSdU_mu @= g.qcd.gauge.project.traceless_anti_hermitian(g(Umu * g.adj(dSdU_mu))) * (
            #     1.0 / 8.0 / 1j
            # )
            # non-adjoint version here
            dSdU_mu @= g.qcd.gauge.project.traceless_anti_hermitian(g(Umu * dSdU_mu)) * (
                1.0 / 8.0 / 1j
            )
            dSdU_mu.otype = Umu.otype.cartesian()
            dS.append(dSdU_mu)
        return dS

    def egradient(self, U, e):
        # Eq. (1.3) and Appendix A of https://link.springer.com/content/pdf/10.1007/JHEP08(2010)071.pdf
        # S(Umu) = -2/g^2 Re trace(Umu * staple)
        # dS(Umu) = lim_{eps->0} Ta ( S(e^{eps Ta} Umu) - S(Umu) ) / eps  with  \Tr[T_a T_b]=-1/2 \delta_{ab}
        # dS(Umu) = -2/g^2 T_a Re trace(T_a * Umu * staple)
        #         = -2/g^2 T_a 1/2 trace(T_a * Umu * staple + adj(staple) * adj(Umu) * adj(Ta))
        #         = -2/g^2 T_a 1/2 trace(T_a * (Umu * staple - adj(staple) * adj(Umu)))
        #         = -2/g^2 T_a 1/2 trace(T_a * (Umu * staple - adj(Umu*staple)))
        #         = -2/g^2 T_a trace(T_a * r0)    with r0 = 1/2(Umu * staple - adj(Umu*staple))
        # r0 = c_a T_a + imaginary_diagonal   with A^dag = -A
        # trace(T_a * r0) = -1/2 c_a
        # dS(Umu) = 1/g^2 tracelss_anti_hermitian(Umu * staple)
        # define staple here as adjoint
        dS = []
        eslash = make_eslash(e)
        de = eslash
        for emu in de:
            mu = eslash.index(emu)
            dSde_mu = self.eenv(U, e, mu)
            # NOTE why is there an adj in (qcd version of) U * staple?
            one = g.qcd.gauge.project.traceless_anti_hermitian(g(emu * dSde_mu)) * (
                -1.0 / 1j
            )
            two = g.qcd.gauge.project.traceless_hermitian(g(emu * dSde_mu))
            three = g.identity(g.mspin(grid)) * g.trace(g(emu * dSde_mu))
            dSde_mu @= one + two + three
            dSde_mu.otype = emu.otype.cartesian()
            dS.append(dSde_mu)
        return dS

def random_algebra_element(lnU, scale=1.0):
    # lnU = [g.mspin(grid) for mu in range(4)]
    for mu in range(4):
        lnU[mu][:] = 0
    # gamma commutators
    Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
                [a].tensor())/4 for b in range(0,4) ] for a in range(0,4) ]
    omega = [ [ [ rng.normal(g.complex(grid), sigma=scale) for b in range(0,4)]
                for a in range(0,4) ] for mu in range(0, 4) ]
    for mu in range(0, 4):
        for a in range(0, 4):
            for b in range(0, 4):
                lnU[mu] += Ji2[a][b]*omega[mu][a][b]
    # V = g.mspin(grid)
    # V = g.matrix.exp(lnV)
    # return lnU

# def random_links(scale=1.0):
#     Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
#                 [a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
#     lnV = g.mspin(grid) 
#     lnV[:] = 0
#     for a in range(0, 4):
#         for b in range(0, 4):
#             lnV += Ji2[a][b] * rng.normal(g.complex(grid), sigma=scale)
#     V = g.mspin(grid)
#     V = g.matrix.exp(lnV)
#     return V

# class Gtetrad(Gbase):
#     def __init__(self, lam):
#         self.lam = lam
#         # self.lam = lam
#         # self.alpha = alpha
#         self.__name__ = f"gravity({lam})"

#     def __call__(self, U, e):



class Ggauge(Gbase):
    def __init__(self, kappa, lam):
        self.kappa = kappa
        self.lam = lam
        # self.alpha = alpha
        self.__name__ = f"gravity({kappa},{lam})"

    def __call__(self, U, e):
        # Let beta = 2 ndim_repr / g^2
        #
        # S(U) = -beta sum_{mu>nu} Re[Tr[P_{mu,nu}]]/ndim_repr        (only U-dependent part)
        #      = -2/g^2 sum_{mu>nu} Re[Tr[P_{mu,nu}]]
        #      = -1/g^2 sum_{mu,nu} Re[Tr[P_{mu,nu}]]
        #      = -2/g^2 sum_{mu,nu} Re[Tr[staple_{mu,nu}^dag U_mu]]
        #
        # since   P_{mu,nu} = staple_{mu,nu}^dag U_mu + staple_{mu,nu} U_mu^dag = 2 Re[staple^dag * U]
        # Nd = len(U)
        # vol = U[0].grid.gsites
        # return self.beta * (1.0 - g.qcd.gauge.plaquette(U)) * (Nd - 1) * Nd * vol / 2.0

        # NOTE "R" here is really det(e) * R
        R = g.real(grid)
        R[:] = 0
        vol = g.real(grid)
        vol[:] = 0
        eslash = make_eslash(e)
        for idx, val in levi.items():
            mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
            Gmunu = g.qcd.gauge.field_strength(U, mu, nu)
            R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
            vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
        # Rsq = R * R # g.component.pow(2)(R)
        dete = det(e)
        # dete = vol # is this dete?
        action = (sign(dete) * ((-1) * (self.kappa / 16) * R
                                        + (self.lam / 96) * vol))
                                # + self.alpha * Rsq * g.component.inv(dete) / 8))
        # action = (sign(dete) * ((-1) * (self.kappa / 16) * R))
        return np.real(np.sum(g.eval(action)[:]))

    # def staple(self, U, mu):
    #     st = g.lattice(U[0])
    #     st[:] = 0
    #     Nd = len(U)
    #     for nu in range(Nd):
    #         if mu != nu:
    #             st += g.qcd.gauge.staple(U, mu, nu)
    #     scale = self.beta / U[0].otype.shape[0]
    #     return g(scale * st)

    def eenv(self, U, e, mu):
        eslash = make_eslash(e)
        Vmu = g.mspin(grid)
        Vmu[:] = 0
        Wmu = g.mspin(grid)
        Wmu[:] = 0
        for (nu, rho, sig), val in levi3[mu].items():
            Vmu += eslash[nu] * eslash[rho] * eslash[sig] * g.gamma[5] * val
            Wmu += (eslash[nu] * g.gamma[5] *
                    g.qcd.gauge.field_strength(U, rho, sig) * val)
        return (self.lam / 96)*Vmu - (self.kappa / 16)*Wmu

    def staple(self, U, e, mu):
        Emu = g.lattice(U[0])
        Emu[:] = 0
        # Emutilde = g.mspin(self.grid)
        # Emutilde[:] = 0
        eslash = make_eslash(e)
        sign_x_plus_mu = g.cshift(sign(det(e)), mu, 1)
        # print(eslash[mu])
        # assert False
        for (nu,rho,sig), val in levi3[mu].items():
            # print(nu,rho,sig, val)
            sign_x_minus_nu = g.cshift(sign(det(e)), nu, -1)
            sign_x_plus_nu = g.cshift(sign(det(e)), nu, 1)
            
            e_rho_x_plus_mu = g.cshift(eslash[rho], mu, 1)
            e_sig_x_plus_mu = g.cshift(eslash[sig], mu, 1)
            e_rho_x_plus_nu = g.cshift(eslash[rho], nu, 1)
            e_sig_x_plus_nu = g.cshift(eslash[sig], nu, 1)
            e_rho_x_minus_nu = g.cshift(eslash[rho], nu, -1)
            e_sig_x_minus_nu = g.cshift(eslash[sig], nu, -1)
            
            U_nu_x_plus_mu = g.cshift(U[nu], mu, 1)
            U_nu_x_minus_nu = g.cshift(U[nu], nu, -1)
            U_mu_x_plus_nu = g.cshift(U[mu], nu, 1)
            
            one = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
                         g.adj(U[nu]) * eslash[rho] * eslash[sig] *
                         g.gamma[5]) * sign(det(e))
            two = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                         g.adj(g.cshift(U[mu], nu, -1)) * U_nu_x_minus_nu *
                         eslash[rho] * eslash[sig] * g.gamma[5]) * sign(det(e))
            three = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
                           g.gamma[5] * U_nu_x_plus_mu *
                           g.adj(U_mu_x_plus_nu) * g.adj(U[nu])) * sign_x_plus_mu
            four = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
                          g.gamma[5] * g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                          g.adj(g.cshift(U[mu], nu, -1)) *
                          U_nu_x_minus_nu) * sign_x_plus_mu
            five = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
                          e_rho_x_plus_nu * e_sig_x_plus_nu *
                          g.gamma[5] * g.adj(U[nu])) * sign_x_plus_nu
            six = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                         g.adj(g.cshift(U[mu], nu, -1)) *
                         e_rho_x_minus_nu * e_sig_x_minus_nu *
                         g.gamma[5] * U_nu_x_minus_nu) * sign_x_minus_nu
            seven = g.eval(U_nu_x_plus_mu * g.cshift(e_rho_x_plus_mu, nu, 1) *
                           g.cshift(e_sig_x_plus_mu, nu, 1) * g.gamma[5] *
                           g.adj(U_mu_x_plus_nu) *
                           g.adj(U[nu])) * g.cshift(sign_x_plus_mu, nu, 1)
            eight = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                           g.cshift(e_rho_x_plus_mu, nu, -1) *
                           g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
                           g.adj(g.cshift(U[mu], nu, -1)) *
                           U_nu_x_minus_nu) * g.cshift(sign_x_plus_mu, nu, -1)
            # I'm shifting kappa inside here
            Emu += 0.125 * val * (one - two + three - four + five - six + seven - eight) * self.kappa
        return g(Emu)

def make_Us():
    # Mike's links
    # make log U
    lnU = [g.mspin(grid) for mu in range(4)]
    for i in range(4):
        lnU[i][:] = 0
    # gamma commutators
    Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
                [a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
    omega = [ [ [ rng.normal(g.complex(grid)) for b in range(0,4)]
                for a in range(0,4) ] for mu in range(0, 4) ]
    for mu in range(0, 4):
        for a in range(0, 4):
            for b in range(0, 4):
                lnU[mu] += Ji2[a][b]*omega[mu][a][b]
    # the Us
    U = [ g.mspin(grid) for mu in range(0,4) ]
    for mu in range(0,4):
        U[mu] = g.matrix.exp(lnU[mu])
    return U


# class Simulation:

#     def __init__(self, L, kappa, lamb, alpha):
#         self.L = L # symmetric lattice
#         self.grid = g.grid([self.L]*4, g.double) # make the lattice
#         g.message(self.grid)
#         self.rng = g.random("seed string") # initialize random seed
#         self.kappa, self.lam, self.alpha = kappa, lamb, alpha # the couplings

#         # make the tetrads
#         self.e = [[self.rng.normal(g.real(self.grid)) for a in range(4)] for mu in range(4)]
#         self.make_Us() # creates the Us
        

#     def random_shift(self, scale=1.0):
#         return self.rng.normal(g.real(self.grid), sigma=scale)
    
    

#     def random_links(self, scale=1.0):
#         Ji2 = [ [(g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma
#                   [a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
#         lnV = g.mspin(self.grid) 
#         lnV[:] = 0
#         for a in range(0, 4):
#             for b in range(0, 4):
#                 lnV += Ji2[a][b] * self.rng.normal(g.complex(self.grid), sigma=scale)
#         V = g.mspin(self.grid)
#         V = g.matrix.exp(lnV)
#         return V

#     def compute_action_check(self,):
#         R = g.real(self.grid)
#         R[:] = 0
#         vol = g.real(self.grid)
#         vol[:] = 0
#         eslash = self.make_eslash()
#         for idx, val in levi.items():
#             mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
#             Gmunu = g.qcd.gauge.field_strength(self.U, mu, nu)
#             R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
#             vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
#         Rsq = R * R # g.component.pow(2)(R)
#         dete = det(self.e)
#         action = (sign(dete) * ((-1) * (self.kappa / 16) * R
#                                 + (self.lam / 96) * vol
#                                 + self.alpha * Rsq * g.component.inv(dete) / 8))
#         # action = R * sign(det(e))
#         return action

    
#     def staple(self, mu):
#         Emu = g.mspin(self.grid)
#         Emu[:] = 0
#         # Emutilde = g.mspin(self.grid)
#         # Emutilde[:] = 0
#         eslash = self.make_eslash()
#         sign_x_plus_mu = g.cshift(sign(det(self.e)), mu, 1)
#         # print(eslash[mu])
#         # assert False
#         for (nu,rho,sig), val in levi3[mu].items():
#             # print(nu,rho,sig, val)
#             sign_x_minus_nu = g.cshift(sign(det(self.e)), nu, -1)
#             sign_x_plus_nu = g.cshift(sign(det(self.e)), nu, 1)
            
#             e_rho_x_plus_mu = g.cshift(eslash[rho], mu, 1)
#             e_sig_x_plus_mu = g.cshift(eslash[sig], mu, 1)
#             e_rho_x_plus_nu = g.cshift(eslash[rho], nu, 1)
#             e_sig_x_plus_nu = g.cshift(eslash[sig], nu, 1)
#             e_rho_x_minus_nu = g.cshift(eslash[rho], nu, -1)
#             e_sig_x_minus_nu = g.cshift(eslash[sig], nu, -1)
            
#             U_nu_x_plus_mu = g.cshift(self.U[nu], mu, 1)
#             U_nu_x_minus_nu = g.cshift(self.U[nu], nu, -1)
#             U_mu_x_plus_nu = g.cshift(self.U[mu], nu, 1)
            
#             one = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
#                          g.adj(self.U[nu]) * eslash[rho] * eslash[sig] *
#                          g.gamma[5]) * sign(det(self.e))
#             two = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
#                          g.adj(g.cshift(self.U[mu], nu, -1)) * U_nu_x_minus_nu *
#                          eslash[rho] * eslash[sig] * g.gamma[5]) * sign(det(self.e))
#             three = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
#                            g.gamma[5] * U_nu_x_plus_mu *
#                            g.adj(U_mu_x_plus_nu) * g.adj(self.U[nu])) * sign_x_plus_mu
#             four = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
#                           g.gamma[5] * g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
#                           g.adj(g.cshift(self.U[mu], nu, -1)) *
#                           U_nu_x_minus_nu) * sign_x_plus_mu
#             five = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
#                           e_rho_x_plus_nu * e_sig_x_plus_nu *
#                           g.gamma[5] * g.adj(self.U[nu])) * sign_x_plus_nu
#             six = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
#                          g.adj(g.cshift(self.U[mu], nu, -1)) *
#                          e_rho_x_minus_nu * e_sig_x_minus_nu *
#                          g.gamma[5] * U_nu_x_minus_nu) * sign_x_minus_nu
#             seven = g.eval(U_nu_x_plus_mu * g.cshift(e_rho_x_plus_mu, nu, 1) *
#                            g.cshift(e_sig_x_plus_mu, nu, 1) * g.gamma[5] *
#                            g.adj(U_mu_x_plus_nu) *
#                            g.adj(self.U[nu])) * g.cshift(sign_x_plus_mu, nu, 1)
#             eight = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
#                            g.cshift(e_rho_x_plus_mu, nu, -1) *
#                            g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
#                            g.adj(g.cshift(self.U[mu], nu, -1)) *
#                            U_nu_x_minus_nu) * g.cshift(sign_x_plus_mu, nu, -1)
#             Emu += 0.125 * val * (one - two + three - four + five - six + seven - eight)
#             # Emutilde part
#             # one = g.eval(eslash[rho] * eslash[sig] * g.gamma[5] *
#             #        links[nu] * U_mu_x_plus_nu * g.adj(U_nu_x_plus_mu)) * sign(det(e))
#             # two = g.eval(eslash[rho] * eslash[sig] * g.gamma[5] *
#             #        g.adj(U_nu_x_minus_nu) * g.cshift(links[mu], nu, -1) *
#             #        g.cshift(U_nu_x_plus_mu, nu, -1)) * sign(det(e))
#             # three = g.eval(links[nu] * U_mu_x_plus_nu * g.adj(U_nu_x_plus_mu) *
#             #          e_rho_x_plus_mu * e_sig_x_plus_mu
#             #          * g.gamma[5]) * sign_x_plus_mu
#             # four = g.eval(g.adj(U_nu_x_minus_nu) * g.cshift(links[mu], nu, -1) *
#             #         g.cshift(U_nu_x_plus_mu, nu, -1) * e_rho_x_plus_mu *
#             #         e_sig_x_plus_mu * g.gamma[5]) * sign_x_plus_mu
#             # five = g.eval(links[nu] * e_rho_x_plus_nu * e_sig_x_plus_nu
#             #         * g.gamma[5] * U_mu_x_plus_nu *
#             #         g.adj(U_nu_x_plus_mu)) * sign_x_plus_nu
#             # six = g.eval(g.adj(U_nu_x_minus_nu) * e_rho_x_minus_nu *
#             #        e_sig_x_minus_nu * g.gamma[5] * g.cshift(links[mu], nu, -1) *
#             #        g.cshift(U_nu_x_plus_mu, nu, -1)) * sign_x_minus_nu
#             # seven = g.eval(links[nu] * U_mu_x_plus_nu * g.cshift(e_rho_x_plus_nu, mu, 1) *
#             #          g.cshift(e_sig_x_plus_nu, mu, 1) * g.gamma[5] *
#             #          g.adj(U_nu_x_plus_mu)) * g.cshift(sign_x_plus_mu, nu, 1)
#             # eight = g.eval(g.adj(U_nu_x_minus_nu) * g.cshift(links[mu], nu, -1) *
#             #          g.cshift(e_rho_x_plus_mu, nu, -1) *
#             #          g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
#             #          g.cshift(U_nu_x_plus_mu, nu, -1)) * g.cshift(sign_x_plus_mu, nu, -1)
#             # Emutilde += 0.125 * val * (- one + two - three + four - five + six - seven + eight)
#         # return (Emu, Emutilde)
#         return Emu


        
            
    # def compute_link_action(self, mu):
    #     R = g.real(self.grid)
    #     R[:] = 0
    #     # E, Etil = staple(links, e, mu)
    #     E = self.staple(mu)
    #     R = g.trace(self.U[mu] * E)
    #     return (-self.kappa / 16) * R


    # def compute_tet_action(self, mu):
    #     V = g.real(self.grid)
    #     V[:] = 0
    #     eslash = self.make_eslash()
    #     F = self.eenv(eslash, mu)
    #     V = sign(det(self.e)) * g.trace(eslash[mu] * F)
    #     return V

    # def compute_total_action(self,):
    #     want = g.real(self.grid)
    #     want[:] = 0
    #     for mu in range(4):
    #         B = self.compute_tet_action(mu)
    #         want += B
    #     return want

    

    

    # def update_links(self,):
    #     for mu in range(4):
    #         # action = compute_action(links, e)
    #         action = self.compute_link_action(mu)
    #         # V = g.lattice(links[mu])
    #         V_eye = g.identity(self.U[mu])
    #         # g.message(V_eye)
    #         V = self.random_links(scale=0.1)
    #         # g.message(V)
    #         V = g.where(self.mask, V, V_eye)
    #         lo = self.U[mu]
    #         # links_prime = links.copy() # copy links?
    #         # g.message(links_prime)
    #         lp = g.eval(V * lo)
    #         self.U[mu] = g.eval(V * lo)
    #         # links_prime[mu] = g.eval(V * links[mu])
    #         # action_prime = compute_action(links_prime, e)
    #         action_prime = self.compute_link_action(mu)
    #         prob = g.component.exp(action - action_prime)
    #         # g.message(prob)
    #         rn = g.lattice(prob)
    #         self.rng.uniform_real(rn)
    #         accept = rn < prob
    #         accept *= self.mask
    #         self.U[mu] @= g.where(accept, lp, lo)
    #         # print(links[mu][0,0,0,0], lo[0,0,0,0], lp[0,0,0,0])
    #         # print('==================')
        
        
    # def update_tetrads(self,):
    #     for mu in range(4):
    #         for a in range(4):
    #             # action = compute_action(links, e)
    #             action = self.compute_tet_action(mu)
    #             ii_eye = g.identity(self.e[mu][a])
    #             ii = self.random_shift(scale=1.)
    #             ii = g.where(self.mask, ii, ii_eye)
    #             eo = self.e[mu][a]
    #             dete = det(self.e)
    #             # print(eo[0,0,0,0])
    #             ep = g.eval(ii + eo)
    #             # print(ep[0,0,0,0])
    #             # print(ep, e[mu][a])
    #             # assert False
    #             # e_prime = 1
    #             self.e[mu][a] = g.eval(ii + eo)
    #             detep = det(self.e)
    #             # print(eo[0,0,0,0], e[mu][a][0,0,0,0])
    #             # print(e[mu][a][0,0,0,0], e_prime[mu][a][0,0,0,0])
    #             # action_prime = compute_action(links, e_prime)
    #             action_prime = self.compute_tet_action(mu)
    #             # print(np.sum(g.eval(action_prime)[:]), np.sum(g.eval(action)[:]))
    #             meas = g.component.pow(self.K)(g.component.abs(detep) * g.component.inv(g.component.abs(dete)))
    #             prob = g.eval(g.component.exp(action - action_prime) * meas)
    #             rn = g.lattice(prob)
    #             self.rng.uniform_real(rn)
    #             accept = rn < prob
    #             accept *= self.mask
    #             self.e[mu][a] @= g.where(accept, ep, eo)
    #             # print(e[mu][a][0,0,0,0], eo[0,0,0,0], ep[0,0,0,0])
    #             # print(np.sum(g.eval(compute_tet_action(links, e, mu))[:]))
    #             # print('==================')
                
            
            
    # def update_fields(self,):
    #     self.update_links()
    #     self.update_tetrads()

    # def run(self, nswps, kappa, lam, alpha, K, crosscheck=False):
    #     self.crosscheck = crosscheck
    #     self.kappa = kappa
    #     self.lam = lam
    #     self.K = K
    #     self.alpha = alpha
    #     # need to do this masking for even odd update
    #     grid_eo = self.grid.checkerboarded(g.redblack)
    #     self.mask_rb = g.complex(grid_eo)
    #     self.mask_rb[:] = 1
    #     self.mask = g.complex(self.grid)

    #     self.measurements = list()
    #     for swp in range(nswps):
    #         # print(swp)
    #         self.sweep(swp)

    # def sweep(self, swp):
    #     plaq = g.qcd.gauge.plaquette(self.U)
    #     R_2x1 = g.qcd.gauge.rectangle(self.U, 2, 1)
    #     the_det = np.real(np.mean(det(self.e)[:]))
    #     act = np.real(np.sum(g.eval(self.compute_total_action())[:]) / self.L**4)
    #     self.measurements.append([plaq, R_2x1,the_det,act])
    #     # act2 = np.sum(g.eval(compute_action_check(U, e))[:])
    #     # # act2 = g.eval(compute_action_check(U, e))[0,0,0,0]
    #     # act1 = g.real(grid)
    #     # act1[:] = 0
    #     # for mu in range(4):
    #     #     act1 += g.eval(compute_link_action(U, e, mu))
    #     # act1 = np.sum(act1[:])
    #     g.message(f"Metropolis {swp} has det = {the_det}, P = {plaq}, R_2x1 = {R_2x1}, act = {act}")
    #     if self.crosscheck:
    #         check_act = np.real(np.sum(g.eval(self.compute_action_check())[:]) / self.L**4)
    #         g.message(f"Cross check action = {check_act}")
    #     # act2 = np.mean(g.eval(compute_action_check(U, e))[:])
    #     # act1 = g.real(grid)
    #     # act1[:] = 0
    #     # for mu in range(4):
    #     #     act1 += g.eval(compute_tet_action(U, e, mu))
    #     # act1 = np.mean(act1[:])
    #     # g.message(f"action1 = {act1}, action2 = {act2}")
    #     for cb in [g.even, g.odd]:
    #         self.mask[:] = 0
    #         self.mask_rb.checkerboard(cb)
    #         g.set_checkerboard(self.mask, self.mask_rb)
    #         self.update_fields()



        



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

def random_four_matrix(emom):
    for mu in range(4):
        for a in range(4):
            emom[mu][a] = rng.normal(g.real(grid))

if __name__ == "__main__":
    levi = make_levi()
    levi3 = three_levi()


    kappa = g.default.get_float("--kappa", 1.0)
    lam = g.default.get_float("--lambda", 1.0)

    g.default.set_verbose("omf4")

    grid = g.grid([4, 4, 4, 8], g.double)
    rng = g.random("hmc-pure-gauge")

    U = make_Us()
    e = [[rng.normal(g.real(grid)) for a in range(4)] for mu in range(4)]

    # conjugate momenta
    Umom = [g.mspin(grid) for mu in range(4)]
    random_algebra_element(Umom)
    emom = g.group.cartesian(e)


    # Log
    g.message(f"Lattice = {grid.fdimensions}")
    g.message("Actions:")
    # action for conj. momenta
    a0 = g.qcd.scalar.action.mass_term()
    g.message(f" - {a0.__name__}")

    # EH action
    a1 = Ggauge(kappa, lam)
    g.message(f" - {a1.__name__}")

    a2 = g.qcd.scalar.action.mass_term()
    g.message(f" - {a0.__name__}")


    def hamiltonian():
        return a0(Umom) + a1(U, e) + a2(emom)

    # molecular dynamics
    sympl = g.algorithms.integrator.symplectic

    # print(a1.Ugradient(U, e, U))
    # assert False

    ipU = sympl.update_p(Umom, lambda: a1.Ugradient(U, e, U))
    iqU = sympl.update_q(U, lambda: a0.gradient(Umom, Umom))
    mdintU = sympl.OMF4(5, ipU, iqU)
    g.message(f"Integration scheme:\n{mdintU}")
    ipe = sympl.update_p(emom, lambda: a1.egradient(U, e))
    iqe = sympl.update_q(e, lambda: a2.gradient(emom, emom))
    mdinte = sympl.OMF4(5, ipe, iqe)
    g.message(f"Integration scheme:\n{mdinte}")
    # metropolis
    metro = g.algorithms.markov.metropolis(rng)

    # MD units
    tau = 0.008
    g.message(f"tau = {tau} MD units")


    def hmcU(tau, mom):
        # rng.normal_element(mom)
        random_algebra_element(mom)
        # mom = make_Us()
        accrej = metro(U)
        h0 = hamiltonian()
        # print(h0)
        # assert False
        mdintU(tau)
        # print(a1(U, e))
        # assert False
        h1 = hamiltonian()
        # print(h1)
        return [accrej(h1, h0), h1 - h0]
    
    def hmce(tau, mom):
        random_four_matrix(mom)
        accrej = metro(e)
        h0 = hamiltonian()
        # print(h0)
        # assert False
        mdinte(tau)
        # print(a1(U, e))
        # assert False
        h1 = hamiltonian()
        # print(h1)
        return [accrej(h1, h0), h1 - h0]


    # thermalization
    ntherm = 100
    for i in range(1, 11):
        h = []
        timer = g.timer("hmc")
        for _ in range(ntherm // 10):
            timer("trajectory")
            h += [hmcU(tau, Umom), hmce(tau, emom)]
            # print(h)
            # assert False
        h = np.array(h)
        timer()
        g.message(f"{i*10} % of thermalization completed")
        g.message(timer)
        g.message(
            f"Plaquette = {g.qcd.gauge.plaquette(U)}, Acceptance = {np.mean(h[:,0]):.2f}, |dH| = {np.mean(np.abs(h[:,1])):.4e}"
        )
        g.message(
            f"dete = {np.sum(det(e)[:])}, Acceptance = {np.mean(h[:,0]):.2f}, |dH| = {np.mean(np.abs(h[:,1])):.4e}"
        )
    assert False

    # production
    history = []
    data = []
    n = 100
    dtrj = 10
    for i in range(n):
        for k in range(dtrj):
            history += [hmc(tau, mom)]
        data += [g.qcd.gauge.plaquette(U)]
        g.message(f"Trajectory {i}, {history[-1]}")

    history = numpy.array(history)
    g.message(f"Acceptance rate = {numpy.mean(history[:,0]):.2f}")
    g.message(f"<|dH|> = {numpy.mean(numpy.abs(history[:,1])):.4e}")

    data = numpy.array(data)
    g.message(f"<plaq>   = {numpy.mean(data[:,0])}")
