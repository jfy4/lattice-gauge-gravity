#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gpt as g
import itertools as it
import numpy as np
import copy

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

def make_eslash(e):
    eslash = [g.mspin(grid) for mu in range(4)]
    for mu in range(4):
        eslash[mu][:] = 0
    for mu in range(4):
        for a in range(4):
            eslash[mu] += g.gamma[a].tensor()*e[mu][a]
    return eslash


# def compute_action_check(link, e):
#     R = g.real(grid)
#     R[:] = 0
#     # Rsq = g.real(grid)
#     # Rsq[:] = 0
#     vol = g.real(grid)
#     vol[:] = 0
#     # GB = g.real(grid)
#     # GB[:] = 0
#     eslash = make_eslash(e)
#     # eslash = [g.mspin(grid) for mu in range(4)]
#     # for mu in range(4):
#     #     for a in range(4):
#     #         eslash[mu] += g.gamma[a].tensor()*e[mu][a]
#     for idx, val in levi.items():
#         # print(idx, val)
#         mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
#         Gmunu = g.qcd.gauge.field_strength(link, mu, nu)
#         # Grhosig = g.qcd.gauge.field_strength(link, rho, sig)
#         R += g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
#         # R += r
#         vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
#     #     GB += g.trace(g.gamma[5] * Gmunu * Grhosig) * val
#     # Rsq = g.component.pow(2)(R)
#     # action = sign(det(e)) * ((-1) * (kappa / 16) * R +
#     #                                (lam / 96) * vol +
#     #                                alpha * Rsq +
#     #                                beta * GB)
#     # volp = np.mean(g.eval(vol)[:])
#     # g.message(f"vol = {volp}")
#     action = sign(det(e)) * ((-1) * (kappa / 16) * R) + (lam / 96) * vol)
#     # action = R * sign(det(e))
#     return action

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


def random_links(scale=1.0):
    lnV = g.mspin(grid) 
    lnV[:] = 0
    for a in range(0, 4):
        for b in range(0, 4):
            lnV += Ji2[a][b] * rng.normal(g.complex(grid), sigma=scale)
    V = g.mspin(grid)
    V = g.matrix.exp(lnV)
    return V

def random_shift(scale=1.0):
    return rng.normal(g.real(grid), sigma=scale)


def staple(links, e, mu):
    Emu = g.mspin(grid)
    Emu[:] = 0
    Emutilde = g.mspin(grid)
    Emutilde[:] = 0
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
        
        U_nu_x_plus_mu = g.cshift(links[nu], mu, 1)
        U_nu_x_minus_nu = g.cshift(links[nu], nu, -1)
        U_mu_x_plus_nu = g.cshift(links[mu], nu, 1)
        
        one = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
               g.adj(links[nu]) * eslash[rho] * eslash[sig] *
               g.gamma[5]) * sign(det(e))
        two = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
               g.adj(g.cshift(links[mu], nu, -1)) * U_nu_x_minus_nu *
               eslash[rho] * eslash[sig] * g.gamma[5]) * sign(det(e))
        three = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
                 g.gamma[5] * U_nu_x_plus_mu *
                 g.adj(U_mu_x_plus_nu) * g.adj(links[nu])) * sign_x_plus_mu
        four = g.eval(e_rho_x_plus_mu * e_sig_x_plus_mu *
                g.gamma[5] * g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                g.adj(g.cshift(links[mu], nu, -1)) *
                U_nu_x_minus_nu) * sign_x_plus_mu
        five = g.eval(U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
                e_rho_x_plus_nu * e_sig_x_plus_nu *
                g.gamma[5] * g.adj(links[nu])) * sign_x_plus_nu
        six = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
               g.adj(g.cshift(links[mu], nu, -1)) *
               e_rho_x_minus_nu * e_sig_x_minus_nu *
               g.gamma[5] * U_nu_x_minus_nu) * sign_x_minus_nu
        seven = g.eval(U_nu_x_plus_mu * g.cshift(e_rho_x_plus_mu, nu, 1) *
                 g.cshift(e_sig_x_plus_mu, nu, 1) * g.gamma[5] *
                 g.adj(U_mu_x_plus_nu) *
                 g.adj(links[nu])) * g.cshift(sign_x_plus_mu, nu, 1)
        eight = g.eval(g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                 g.cshift(e_rho_x_plus_mu, nu, -1) *
                 g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
                 g.adj(g.cshift(links[mu], nu, -1)) *
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


def eenv(links, eslash, mu):
    Vmu = g.mspin(grid)
    Vmu[:] = 0
    Wmu = g.mspin(grid)
    Wmu[:] = 0
    for (nu, rho, sig), val in levi3[mu].items():
        Vmu += eslash[nu] * eslash[rho] * eslash[sig] * g.gamma[5] * val
        Wmu += (eslash[nu] * g.gamma[5] *
                g.qcd.gauge.field_strength(links, rho, sig) * val)
    return (lam / 96)*Vmu - (kappa / 16)*Wmu
    # return Vmu
        
            
def compute_link_action(links, e, mu):
    R = g.real(grid)
    R[:] = 0
    # eslash = make_eslash(e)
    # E, Etil = staple(links, e, mu)
    E = staple(links, e, mu)
    # R = g.trace(links[mu] * E) + g.trace(Etil * g.adj(links[mu]))
    R = g.trace(links[mu] * E)
    # R = g.trace(Etil * g.adj(links[mu]))
    return (-kappa / 16) * R
    # return R


def compute_tet_action(links, e, mu):
    V = g.real(grid)
    V[:] = 0
    eslash = make_eslash(e)
    F = eenv(links, eslash, mu)
    V = sign(det(e)) * g.trace(eslash[mu] * F)
    return V
    # V = g.trace(eslash[mu] * F)
    # return V

# levi3 = three_levi()
# for idx, val in  levi3[mu]:
#     nu, rho, sig = idx[0], idx[1], idx[2]
#     g.trace(g.gamma[5] * g.qcd.gauge.field_strength(link, mu, nu) * eslash[rho] * eslash[sig])

def compute_total_action(links, e):
    want = g.real(grid)
    want[:] = 0
    for mu in range(4):
        # A = compute_link_action(links, e, mu)
        B = compute_tet_action(links, e, mu)
        want += B
    return want


def update_links(links, e, mask):
    for mu in range(4):
        # action = compute_action(links, e)
        action = compute_link_action(links, e, mu)
        # V = g.lattice(links[mu])
        V_eye = g.identity(links[mu])
        # g.message(V_eye)
        V = random_links(scale=0.1)
        # g.message(V)
        V = g.where(mask, V, V_eye)
        lo = links[mu]
        # links_prime = links.copy() # copy links?
        # g.message(links_prime)
        lp = g.eval(V * lo)
        links[mu] = g.eval(V * lo)
        # links_prime[mu] = g.eval(V * links[mu])
        # action_prime = compute_action(links_prime, e)
        action_prime = compute_link_action(links, e, mu)
        prob = g.component.exp(action - action_prime)
        # g.message(prob)
        rn = g.lattice(prob)
        rng.uniform_real(rn)
        accept = rn < prob
        accept *= mask
        links[mu] @= g.where(accept, lp, lo)
        # print(links[mu][0,0,0,0], lo[0,0,0,0], lp[0,0,0,0])
        # print('==================')
        
        
def update_tetrads(links, e, mask):
    for mu in range(4):
        for a in range(4):
            # action = compute_action(links, e)
            action = compute_tet_action(links, e, mu)
            ii_eye = g.identity(e[mu][a])
            ii = random_shift(scale=1.)
            ii = g.where(mask, ii, ii_eye)
            eo = e[mu][a]
            dete = det(e)
            # print(eo[0,0,0,0])
            ep = g.eval(ii + eo)
            # print(ep[0,0,0,0])
            # print(ep, e[mu][a])
            # assert False
            # e_prime = 1
            e[mu][a] = g.eval(ii + eo)
            detep = det(e)
            # print(eo[0,0,0,0], e[mu][a][0,0,0,0])
            # print(e[mu][a][0,0,0,0], e_prime[mu][a][0,0,0,0])
            # action_prime = compute_action(links, e_prime)
            action_prime = compute_tet_action(links, e, mu)
            # print(np.sum(g.eval(action_prime)[:]), np.sum(g.eval(action)[:]))
            meas = g.component.pow(K)(g.component.abs(detep) * g.component.inv(g.component.abs(dete)))
            prob = g.eval(g.component.exp(action - action_prime) * meas)
            rn = g.lattice(prob)
            rng.uniform_real(rn)
            accept = rn < prob
            accept *= mask
            e[mu][a] @= g.where(accept, ep, eo)
            # print(e[mu][a][0,0,0,0], eo[0,0,0,0], ep[0,0,0,0])
            # print(np.sum(g.eval(compute_tet_action(links, e, mu))[:]))
            # print('==================')
            
            
            
def update(links, e, mask):
    update_links(links, e, mask)
    update_tetrads(links, e, mask)


# In[4]:

if __name__ == "__main__":

    # initialize lattice
    L = 4
    grid = g.grid([L]*4, g.double)
    g.message(grid)
    rng = g.random("seed string")   
    
    # parameters
    kappa = 5.
    lam = 5.
    K = -1.
    # alpha = 1
    # beta = 1
    nswp = 2000
    
    # make the tetrads
    e = [[rng.normal(g.real(grid)) for a in range(4)] for mu in range(4)]
    
    
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
        
        
    # In[8]:
    levi = make_levi()
    # print(levi)
    levi3 = three_levi()
    # print(levi3)
    # assert False
    # a link update
    
    # need to do this masking for even odd update
    grid_eo = grid.checkerboarded(g.redblack)
    mask_rb = g.complex(grid_eo)
    mask_rb[:] = 1
    mask = g.complex(grid)

#     act2 = np.sum(g.eval(compute_action_check(U, e))[:])
#     # act2 = g.eval(compute_action_check(U, e))[0,0,0,0]
#     act1 = g.real(grid)
#     act1[:] = 0
#     for mu in range(4):
#         act1 += g.eval(compute_link_action(U, e, mu))
#     act1 = np.sum(act1[:])
#     # act1 = act1[0,0,0,0]
#     e00 = e[0][0][0,0,0,0]
# #     GGGG = g.qcd.gauge.field_strength(U,0,1)[0,0,0,0][0,1]
#     g.message(f"action1 = {act1}, action2 = {act2}")
    # assert False
    # g.message(e[0][0])
    meas = list()
    for i in range(nswp):
        plaq = g.qcd.gauge.plaquette(U)
        R_2x1 = g.qcd.gauge.rectangle(U, 2, 1)
        the_det = np.real(np.mean(det(e)[:]))
        act = np.real(np.sum(g.eval(compute_total_action(U, e))[:]) / L**4)
        meas.append([plaq, R_2x1,the_det,act])
        # act2 = np.sum(g.eval(compute_action_check(U, e))[:])
        # # act2 = g.eval(compute_action_check(U, e))[0,0,0,0]
        # act1 = g.real(grid)
        # act1[:] = 0
        # for mu in range(4):
        #     act1 += g.eval(compute_link_action(U, e, mu))
        # act1 = np.sum(act1[:])
        g.message(f"Metropolis {i} has det = {the_det}, P = {plaq}, R_2x1 = {R_2x1}, act = {act}")
        # act2 = np.mean(g.eval(compute_action_check(U, e))[:])
        # act1 = g.real(grid)
        # act1[:] = 0
        # for mu in range(4):
        #     act1 += g.eval(compute_tet_action(U, e, mu))
        # act1 = np.mean(act1[:])
        # g.message(f"action1 = {act1}, action2 = {act2}")
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            update(U, e, mask)
    
    np.save("measure_K" + str(K) + "_kappa" + str(kappa) + "_lam" + str(lam) + ".npy", meas)
        
            
# In[ ]:




