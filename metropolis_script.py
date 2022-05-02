#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gpt as g
import itertools as it
import numpy as np

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
        for a in range(4):
            eslash[mu] += g.gamma[a].tensor()*e[mu][a]
    return eslash


# def compute_action(link, e):
#     R = g.real(grid)
#     R[:] = 0
#     Rsq = g.real(grid)
#     Rsq[:] = 0
#     vol = g.real(grid)
#     vol[:] = 0
#     GB = g.real(grid)
#     GB[:] = 0
#     eslash = make_eslash(e)
#     # eslash = [g.mspin(grid) for mu in range(4)]
#     # for mu in range(4):
#     #     for a in range(4):
#     #         eslash[mu] += g.gamma[a].tensor()*e[mu][a]
#     for idx, val in levi.items():
# #         print(idx)
#         mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
#         Gmunu = g.qcd.gauge.field_strength(link, mu, nu)
#         Grhosig = g.qcd.gauge.field_strength(link, rho, sig)
#         r = g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
#         R += r
#         vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
#         GB += g.trace(g.gamma[5] * Gmunu * Grhosig) * val
#     Rsq = g.component.pow(2)(R)
#     action = sign(det(e)) * ((-1) * (kappa / 16) * R +
#                                    (lam / 96) * vol +
#                                    alpha * Rsq +
#                                    beta * GB)
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


# I'll finish this up
def staple(links, eslash, mu):
    Emu = g.mspin(grid)
    Emu[:] = 0
    for (nu,rho,sig), val in levi3[mu].items():
        e_rho_x_plus_mu = g.cshift(eslash[rho], mu, 1)
        e_sig_x_plus_mu = g.cshift(eslash[sig], mu, 1)
        e_rho_x_plus_nu = g.cshift(eslash[rho], nu, 1)
        e_sig_x_plus_nu = g.cshift(eslash[sig], nu, 1)
        e_rho_x_minus_nu = g.cshift(eslash[rho], nu, -1)
        e_sig_x_minus_nu = g.cshift(eslash[sig], nu, -1)
        
        U_nu_x_plus_mu = g.cshift(links[nu], mu, 1)
        U_nu_x_minus_nu = g.cshift(links[nu], nu, -1)
        U_mu_x_plus_nu = g.cshift(links[mu], nu, 1)
        
        one = (U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
               g.adj(links[nu]) * eslash[rho] * eslash[sig] *
               g.gamma[5])
        two = (g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
               g.adj(g.cshift(links[mu], nu, -1)) * U_nu_x_minus_nu *
               eslash[rho] * eslash[sig] * g.gamma[5])
        three = (e_rho_x_plus_mu * e_sig_x_plus_mu *
                 g.gamma[5] * U_nu_x_plus_mu *
                 g.adj(U_mu_x_plus_nu) * g.adj(links[nu]))
        four = (e_rho_x_plus_mu * e_sig_x_plus_mu *
                g.gamma[5] * g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                g.adj(g.cshift(links[mu], nu, -1)) * U_nu_x_minus_nu)
        five = (U_nu_x_plus_mu * g.adj(U_mu_x_plus_nu) *
                e_rho_x_plus_nu * e_sig_x_plus_nu *
                g.gamma[5] * g.adj(links[nu]))
        six = (g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
               g.adj(g.cshift(links[mu], nu, -1)) *
               e_rho_x_minus_nu * e_sig_x_minus_nu *
               g.gamma[5] * U_nu_x_minus_nu)
        seven = (U_nu_x_plus_mu * g.cshift(e_rho_x_plus_mu, nu, 1) *
                 g.cshift(e_sig_x_plus_mu, nu, 1) * g.gamma[5] *
                 g.adj(U_mu_x_plus_nu) * g.adj(links[nu]))
        eight = (g.adj(g.cshift(U_nu_x_plus_mu, nu, -1)) *
                 g.cshift(e_rho_x_plus_mu, nu, -1) *
                 g.cshift(e_sig_x_plus_mu, nu, -1) * g.gamma[5] *
                 g.adj(g.cshift(links[mu], nu, -1)) * U_nu_x_minus_nu)
        Emu += 0.125 * val * (one - two + three - four + five - six + seven - eight)
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
        
            
def compute_link_action(links, e, mu):
    R = g.real(grid)
    R[:] = 0
    eslash = make_eslash(e)
    st = staple(links, eslash, mu)
    R = g.trace(links[mu] * st) - g.trace(g.adj(links[mu] * st))
    return sign(det(e)) * (-kappa / 64) * R


def compute_tet_action(links, e, mu):
    V = g.real(grid)
    V[:] = 0
    eslash = make_eslash(e)
    F = eenv(links, eslash, mu)
    V = sign(det(e)) * g.trace(eslash[mu] * F)
    return V

# levi3 = three_levi()
# for idx, val in  levi3[mu]:
#     nu, rho, sig = idx[0], idx[1], idx[2]
#     g.trace(g.gamma[5] * g.qcd.gauge.field_strength(link, mu, nu) * eslash[rho] * eslash[sig])


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
        links_prime = links[:] # copy links?
        # g.message(links_prime)
        links_prime[mu] = g.eval(V * links[mu])
        # action_prime = compute_action(links_prime, e)
        action_prime = compute_link_action(links_prime, e, mu)
        prob = g.component.exp(action - action_prime)
        # g.message(prob)
        rn = g.lattice(prob)
        rng.uniform_real(rn)
        accept = rn < prob
        accept *= mask
        links[mu] @= g.where(accept, links_prime[mu], links[mu])
        
        
def update_tetrads(links, e, mask):
    for mu in range(4):
        for a in range(4):
            # action = compute_action(links, e)
            action = compute_tet_action(links, e, mu)
            ii_eye = g.identity(e[mu][a])
            ii = random_shift(scale=0.1)
            ii = g.where(mask, ii, ii_eye)
            e_prime = e[:]
            e_prime[mu][a] = g.eval(ii + e[mu][a])
            # action_prime = compute_action(links, e_prime)
            action_prime = compute_tet_action(links, e_prime, mu)
            prob = g.component.exp(action - action_prime)
            rn = g.lattice(prob)
            rng.uniform_real(rn)
            accept = rn < prob
            accept *= mask
            e[mu][a] @= g.where(accept, e_prime[mu][a], e[mu][a])
            
            
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
    kappa = 1
    lam = 1
    alpha = 1
    beta = 1
    nswp = 10
    
    # make the tetrads
    e = [[rng.normal(g.real(grid)) for a in range(4)] for mu in range(4)]
    
    
    # Mike's links
    # make log U
    lnU = [g.mspin(grid) for mu in range(4)]
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
    levi3 = three_levi()
    # a link update
    
    # need to do this masking for even odd update
    grid_eo = grid.checkerboarded(g.redblack)
    mask_rb = g.complex(grid_eo)
    mask_rb[:] = 1
    mask = g.complex(grid)
    
    # g.message(e[0][0])
    
    for i in range(nswp):
        plaq = g.qcd.gauge.plaquette(U)
        R_2x1 = g.qcd.gauge.rectangle(U, 2, 1)
        the_det = np.real(np.mean(det(e)[:]))
        g.message(f"Metropolis {i} has det = {the_det}, P = {plaq}, R_2x1 = {R_2x1}")
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            update(U, e, mask)
            
        
            
# In[ ]:




