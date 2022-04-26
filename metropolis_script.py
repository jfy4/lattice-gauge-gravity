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



def compute_action(link, e):
    R = g.real(grid)
    R[:] = 0
    Rsq = g.real(grid)
    Rsq[:] = 0
    vol = g.real(grid)
    vol[:] = 0
    GB = g.real(grid)
    GB[:] = 0
    eslash = [g.mspin(grid) for mu in range(4)]
    for mu in range(4):
        for a in range(4):
            eslash[mu] += g.gamma[a].tensor()*e[mu][a]
    for idx, val in levi.items():
#         print(idx)
        mu, nu, rho, sig = idx[0], idx[1], idx[2], idx[3]
        Gmunu = g.qcd.gauge.field_strength(link, mu, nu)
        Grhosig = g.qcd.gauge.field_strength(link, rho, sig)
        r = g.trace(g.gamma[5] * Gmunu * eslash[rho] * eslash[sig]) * val
        R += r
        vol += g.trace(g.gamma[5] * eslash[mu] * eslash[nu] * eslash[rho] * eslash[sig]) * val
        GB += g.trace(g.gamma[5] * Gmunu * Grhosig) * val
    Rsq = g.component.pow(2)(R)
    action = sign(det(e)) * ((-1) * (kappa / 16) * R +
                                   (lam / 96) * vol +
                                   alpha * Rsq +
                                   beta * GB)
    return action

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

# def three_levi():
#     arr = {i:dict() for i in range(4)}
#     for i,j,k,l in it.product(range(4), repeat=4):
#         prod = (i-j)*(i-k)*(i-l)*(j-k)*(j-l)*(k-l)
#         if prod == 0:
#             continue
#         else:
#             if prod > 0:
#                 arr[i][(j,k,l)] = 1
#             else:
#                 arr[i][(j,k,l)] = -1
#     return arr


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


# levi3 = three_levi()
# for idx, val in  levi3[mu]:
#     nu, rho, sig = idx[0], idx[1], idx[2]
#     g.trace(g.gamma[5] * g.qcd.gauge.field_strength(link, mu, nu) * eslash[rho] * eslash[sig])


def update_links(links, e, mask):
    for mu in range(4):
        action = compute_action(links, e)
#         V = g.lattice(links[mu])
        V_eye = g.identity(links[mu])
#         g.message(V_eye)
        V = random_links(scale=0.1)
#         g.message(V)
        V = g.where(mask, V, V_eye)
        links_prime = links # copy links?
#         g.message(links_prime)
        links_prime[mu] = g.eval(V * links[mu])
        action_prime = compute_action(links_prime, e)
    
        prob = g.component.exp(action - action_prime)
#         g.message(prob)
        rn = g.lattice(prob)
        rng.uniform_real(rn)
        accept = rn < prob
        accept *= mask
        links[mu] @= g.where(accept, links_prime[mu], links[mu])
        
        
def update_tetrads(links, e, mask):
    for mu in range(4):
        for a in range(4):
            action = compute_action(links, e)
            ii_eye = g.identity(e[mu][a])
            ii = random_shift(scale=0.1)
            ii = g.where(mask, ii, ii_eye)
            e_prime = e
            e_prime[mu][a] = g.eval(ii + e[mu][a])
            action_prime = compute_action(links, e_prime)
            
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
    nswp = 5
    
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
    # a link update
    
    # need to do this masking for even odd update
    grid_eo = grid.checkerboarded(g.redblack)
    mask_rb = g.complex(grid_eo)
    mask_rb[:] = 1
    mask = g.complex(grid)
    
    # g.message(e[0][0])
    
    for i in range(nswp):
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            update(U, e, mask)
            
    # g.message(e[0][0])
        
            
# In[ ]:




