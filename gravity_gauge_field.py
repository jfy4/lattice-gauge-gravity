import argparse
import gpt as g
import itertools as it

parser = argparse.ArgumentParser()
parser.add_argument('--mpi', default="1.1.1.1", type=str, help='mpi ranks passed to gpt')
globals().update(vars(parser.parse_args()))

# build lattice
L = 4
grid = g.grid([L,L,L,L], g.double)

# build identity gauge field
lnU = [ g.mspin(grid) for mu in range(0,4) ]
for mu in range(0,4):
    lnU[mu][:] = 0
g.message("\ninitialized gauge field to identity, lnU_{0}[0,0,0,0] = ", lnU[0][0,0,0,0])

# build Spin(4) generators
Ji2 = [ [ (g.gamma[a].tensor()*g.gamma[b].tensor() - g.gamma[b].tensor()*g.gamma[a].tensor())/8 for b in range(0,4) ] for a in range(0,4) ]
g.message("\nbuilt Spin(4) generators")
for a in range(0, 4):
    for b in range(0, 4):
        g.message("i/2 * J_{",a,b,"} = ", Ji2[a][b])

# build random spin connection
rng = g.random("seed string")
omega = [ [ [ rng.normal(g.complex(grid)) for b in range(0,4) ] for a in range(0,4) ] for mu in range(0, 4) ]
#for mu in range(0, 4):
#    for a in range(0, 4):
#        for b in range(0, a):
#            omega[mu][b][a] = g.eval( omega[mu][a][b]*(-1) )
g.message("\nbuilt random spin connection")
for mu in range(0, 4):
    for a in range(0, 4):
        for b in range(0, 4):
            g.message("omega_{",mu,"}^{",a,b,"}[0,0,0,0] = ", omega[mu][a][b][0,0,0,0])

# build gauge field from matrix exponential
for mu in range(0, 4):
    for a in range(0, 4):
        for b in range(0, 4):
            lnU[mu] += Ji2[a][b]*omega[mu][a][b]
g.message("(i/2 * J_{ab} * omega_0^{ab})[0,0,0,0] = ", lnU[0][0,0,0,0])
U = [ g.mspin(grid) for mu in range(0,4) ]
for mu in range(0,4):
    U[mu] = g.matrix.exp(lnU[mu])
g.message("\nU_mu = exp(i/2 * J_{ab} * omega_mu^{ab})")
g.message("U[0][0,0,0,0] = ", U[0][0,0,0,0])

# test matrix exponential function
tol = 1/1e12
test1 = g.eval(g.gamma["I"].tensor()*g.component.cos(omega[0][0][0]) + g.gamma[1].tensor()*g.component.sin(omega[0][0][0])*1j)[0,0,0,0]
test2 = g.eval(g.matrix.exp(g.gamma[1].tensor()*omega[0][0][0]*1j))[0,0,0,0]
eps2 = g.norm2(test1-test2)
g.message("\ntesting matrix exponentiation ||a-b||^2=", eps2,"=0?")
assert(eps2 < tol**2)

# test gauge field is unitary
g.message("\ntesting unitarity")
for mu in range(0,4):
    one = g.complex(grid)
    one[:] = 1
    eye = g.eval(g.gamma["I"].tensor()*one)
    eps = U[mu]*g.adj(U[mu]) - eye
    eps2 = g.norm2(eps)
    g.message("|U_{",mu,"} * U^dagger_{",mu,"} - 1|^2/Vol^2 = ",eps2/L**8,"=0?")
    assert(eps2/L**8 < tol**2)

# build frame field
e = [[rng.normal(g.real(grid)) for a in range(4)] for mu in range(4)]

g.message("\nbuilt random frame field")

for mu in range(0, 4):
    for a in range(0, 4):
        g.message("e_{",mu,a,"} = ", e[mu][a][0,0,0,0])

# build Levi-Civita connection
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
levi = make_levi()

# build frame field determinant
def det(e):
    want = g.real(grid)
    want[:] = 0
    for idx, val in levi.items():
        want += (e[0][idx[0]] * e[1][idx[1]] *
                     e[2][idx[2]] * e[3][idx[3]] *
                     val)
    return g.eval(want)
edet = det(e)

# test determinant function
emat = g.mspin(grid)
emat[0,0,0,0] = g.mspin([[e[mu][a][0,0,0,0] for a in range(4)] for mu in range(4)])
ematdet = g.matrix.det(emat)[0,0,0,0]
g.message("\ntesting Levi-Civita determinant")
g.message("det(e[0,0,0,0]) = ",edet[0,0,0,0]," = ",ematdet, "?")
assert(abs(edet[0,0,0,0] - ematdet) < tol)

# build eslash
eslash = [ g.mspin(grid) for mu in range(0,4) ]
for mu in range(0,4):
    eslash[mu][:] = 0
    for a in range(0,4):
        eslash[mu] += g.gamma[a].tensor()*e[mu][a]
    g.message("eslash_",mu,"[0,0,0,0] =",eslash[mu][0,0,0,0])

# test determinant from spin trace
g.message("\ntesting eslash")
e_zero = g.real(grid)
e_zero[:] = 0
for idx, val in levi.items():
    e_zero += (g.trace(eslash[idx[0]] * eslash[idx[1]] *
                     eslash[idx[2]] * eslash[idx[3]]) *
                     val)

g.message("\neps^{mu nu rho sigma}Tr(esl_mu esl_nu esl_rho esl_sigma) = ",e_zero[0,0,0,0], "= 0?")
assert(abs(e_zero[0,0,0,0]) < tol)

# test determinant from spin trace
edet_prime = g.real(grid)
edet_prime[:] = 0
for idx, val in levi.items():
    edet_prime += (1/96. * g.trace(eslash[idx[0]] * eslash[idx[1]] *
                     eslash[idx[2]] * eslash[idx[3]] * g.gamma[5]) *
                     val)

g.message("\neps^{mu nu rho sigma}Tr(g5 esl_mu esl_nu esl_rho esl_sigma)/96 = det(e[0,0,0,0])?")
g.message("\ndet(e[0,0,0,0]) = ",edet[0,0,0,0]," = ",edet_prime[0,0,0,0], "?")
assert(abs(edet[0,0,0,0] - edet_prime[0,0,0,0]) < tol)

# build sign of det
sign_edet = g.eval(g.component.abs(edet) * g.component.inv(edet))
g.message("\nsign(det(e[0,0,0,0])) = ",sign_edet[0,0,0,0])

# build curvature scalar
# spin algebra most easily gives e(x) R(x) = 1/8 Tr(g5 G_{mu nu} esl_rho esl_sigma)
def curvature_scalar_times_e(eslash, U, verbose=False):
    R_e = g.real(grid)
    R_e[:] = 0
    for idx, val in levi.items():
        Gmunu = g.qcd.gauge.field_strength(U, idx[0], idx[1])
        if verbose:
            g.message("\nG_{",idx[0],",",idx[1],"} [0,0,0,0] = ",Gmunu[0,0,0,0])
            for a in range(0, 4):
                for b in range(0, 4):
                    g.message("\nG_{",idx[0],",",idx[1],"}^{",a,",",b,"} [0,0,0,0] = ",g.trace(Gmunu[0,0,0,0]*g.gamma[a]*g.gamma[b])/4)
            g.message("\nG_{",idx[0],",",idx[1],"}^{5} [0,0,0,0] = ",g.trace(Gmunu[0,0,0,0]*g.gamma[5])/4)
        R_e += (1/8. * g.trace(Gmunu *
                     eslash[idx[2]] * eslash[idx[3]] * g.gamma[5]) *
                     val)

    return g.eval(R_e)


R_e = curvature_scalar_times_e(eslash, U, verbose=True)
g.message("\ncalculated e R = eps^{mu nu rho sigma}Tr(g5 esl_mu esl_nu esl_rho esl_sigma)/8")
g.message("\ne R [0,0,0,0] = ",R_e[0,0,0,0])

# R derived from e * R
R = g.eval(R_e * g.component.inv(edet))
g.message("\nR[0,0,0,0] = ",R[0,0,0,0])

# |e| R derived from e * R
R_term = g.eval(R_e * g.component.abs(edet) * g.component.inv(edet))
g.message("\n|e| R [0,0,0,0] = ",R_term[0,0,0,0])

# |e| R^2 derived from e * R
R2_term = g.eval(g.component.pow(2)(R_e) * g.component.inv(g.component.abs(edet)))
g.message("\n|e| R^2 [0,0,0,0] = ",R2_term[0,0,0,0])

# wrong parity curvature scalar
def curvature_scalar_tilde_times_e(eslash, U, verbose=False):
    R_tilde_e = g.real(grid)
    R_tilde_e[:] = 0
    for idx, val in levi.items():
        Gmunu = g.qcd.gauge.field_strength(U, idx[0], idx[1])
        if verbose:
            g.message("\nG_{",idx[0],",",idx[1],"} [0,0,0,0] = ",Gmunu[0,0,0,0])
            for a in range(0, 4):
                for b in range(0, 4):
                    g.message("\nG_{",idx[0],",",idx[1],"}^{",a,",",b,"} [0,0,0,0] = ",g.trace(Gmunu[0,0,0,0]*g.gamma[a]*g.gamma[b])/4)
        R_tilde_e += (1/8. * g.trace(Gmunu *
                     eslash[idx[2]] * eslash[idx[3]]) *
                     val)
    return g.eval(R_tilde_e)

R_tilde_e = curvature_scalar_tilde_times_e(eslash, U)
g.message("\ncalculated e R_tilde = eps^{mu nu rho sigma}Tr(esl_mu esl_nu esl_rho esl_sigma)/8")
g.message("\ne R_tilde [0,0,0,0] = ",R_tilde_e[0,0,0,0])

# R_tilde derived from e * R
R_tilde = g.eval(R_tilde_e * g.component.inv(edet))
g.message("\nR_tilde[0,0,0,0] = ",R_tilde[0,0,0,0])

# |e| RTilde^2 derived from e * RTilde
R_tilde2_term = g.eval(g.component.pow(2)(R_tilde_e) * g.component.inv(g.component.abs(edet)))
g.message("\n|e| R_tilde^2 [0,0,0,0] = ",R_tilde2_term[0,0,0,0])

# build inverse frame field
eInv = [[g.real(grid) for mu in range(4)] for a in range(4)]
for a in range(0,4):
    for mu in range(0,4):
        eInv[a][mu][:] = 0

for idx_a, val_a in levi.items():
    for idx_mu, val_mu in levi.items():
        eInv[idx_a[0]][idx_mu[0]] += (1/6. * e[idx_mu[1]][idx_a[1]]
                                    * e[idx_mu[2]][idx_a[2]] * e[idx_mu[3]][idx_a[3]]
                                    * val_a * val_mu) * g.component.inv(edet)

g.message("\nbuilt inverse frame field")
for a in range(0, 4):
    for mu in range(0, 4):
        g.message("eInv_{",a,mu,"} = ", eInv[a][mu][0,0,0,0])

# test inverse frame field
ematInv = g.matrix.inv(emat)[0,0,0,0]
eps2 = 0
for a in range(0,4):
    for mu in range(0,4):
        eps2 += abs(ematInv[a,mu] - eInv[a][mu][0,0,0,0])**2
g.message("\ntesting Levi-Civita inverse, error^2 =", eps2,"=0?")
assert(eps2 < tol**2)

# build eInvslash
eInvslash = [ g.mspin(grid) for mu in range(0,4) ]
for mu in range(0,4):
    eInvslash[mu][:] = 0
    for a in range(0,4):
        eInvslash[mu] += g.gamma[a].tensor()*eInv[a][mu]
    g.message("eInvslash_",mu,"[0,0,0,0] =",eInvslash[mu][0,0,0,0])

# build inverse metric
gInv = [[g.real(grid) for mu in range(4)] for a in range(4)]
for mu in range(0,4):
    for nu in range(0,4):
        gInv[mu][nu] = g.eval(g.trace(eInvslash[mu] * eInvslash[nu])/4)

g.message("\nbuilt inverse metric")
for mu in range(0, 4):
    for nu in range(0, 4):
        g.message("gInv_{",mu,nu,"} = ", gInv[mu][nu][0,0,0,0])

# build metric
metric = [[g.real(grid) for mu in range(4)] for a in range(4)]
for mu in range(0,4):
    for nu in range(0,4):
        metric[mu][nu] = g.eval(g.trace(eslash[mu] * eslash[nu])/4)

g.message("\nbuilt metric")
for mu in range(0, 4):
    for nu in range(0, 4):
        g.message("metric_{",mu,nu,"} = ", metric[mu][nu][0,0,0,0])

# test metric x inverse metric
gmat = g.mspin(grid)
gmat[0,0,0,0] = g.mspin([[metric[mu][nu][0,0,0,0] for nu in range(4)] for mu in range(4)])

gInvmat = g.mspin(grid)
gInvmat[0,0,0,0] = g.mspin([[gInv[mu][nu][0,0,0,0] for nu in range(4)] for mu in range(4)])

metric_test = g.eval(gmat * gInvmat)
g.message("\ng_{mu nu} g^{nu rho} [0,0,0,0] = ",metric_test[0,0,0,0])
eps2 = g.norm2(metric_test[0,0,0,0] - g.gamma["I"].tensor())
g.message("\ntesting g_{mu nu} g^{nu rho} = delta_mu^rho, error^2 =", eps2,"=0?")
assert(eps2 < tol**2)

# test metric determinant
metric_det = g.eval(g.matrix.det(gmat))
g.message("\ntesting det(g) =",metric_det[0,0,0,0],"=",g.component.pow(2)(edet)[0,0,0,0],"=det(e^2)?")
eps = abs(metric_det[0,0,0,0] - g.component.pow(2)(edet)[0,0,0,0])
g.message("\ntesting det(g) = det(e^2), error =", eps,"=0?")
assert(eps < tol)

# build Riemann tensor
Riemann = [ [ [ [ g.real(grid) for sigma in range(0,4) ] for rho in range(0,4) ] for nu in range(0,4) ] for mu in range(0,4) ]

for mu in range(0, 4):
    for nu in range(0, 4):
        for rho in range(0, 4):
            for sigma in range(0, 4):
                if mu == nu:
                    Riemann[mu][nu][rho][sigma][:] = 0
                else:
                    Gmunu = g.qcd.gauge.field_strength(U, mu, nu)
                    Riemann[mu][nu][rho][sigma] = g.eval( g.trace(Gmunu * eslash[rho] * eslash[sigma]) )

g.message("\nbuilt Riemann tensor")


g.message("\nRiemann tensor 0 1 2 3 = ", Riemann[0][1][2][3][0,0,0,0])

g.message("\nRiemann tensor 0 1 3 2 = ", Riemann[0][1][3][2][0,0,0,0])

g.message("\nRiemann tensor 1 0 2 3 = ", Riemann[1][0][2][3][0,0,0,0])

g.message("\nRiemann tensor 2 3 0 1 = ", Riemann[2][3][0][1][0,0,0,0])

# build plaquette

mu = 0
nu = 1

Gmunu = g.qcd.gauge.field_strength(U, mu, nu)
TrGmunu = g.eval( g.trace(Gmunu) )
Trg5Gmunu = g.eval( g.trace(Gmunu * g.gamma[5]) )
g.message("\nTr(G_{",mu,nu,"}) = ", TrGmunu[0,0,0,0])
g.message("\nTr(g5 G_{",mu,nu,"}) = ", Trg5Gmunu[0,0,0,0])

g.message("\nG_{",mu,",",nu,"} [0,0,0,0] = ",Gmunu[0,0,0,0])
for a in range(0, 4):
    for b in range(0, 4):
        g.message("\nG_{",mu,",",nu,"}^{",a,",",b,"} [0,0,0,0] = ",g.trace(Gmunu[0,0,0,0]*g.gamma[a]*g.gamma[b])/4)
g.message("\nG_{",mu,",",nu,"}^{5} [0,0,0,0] = ",g.trace(Gmunu[0,0,0,0]*g.gamma[5])/4)

Pmunu = g.eval(U[mu] * g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu]))
TrPmunu = g.eval( g.trace(Pmunu) )
Trg5Pmunu = g.eval( g.trace(Pmunu * g.gamma[5]) )
g.message("\nTr(P_{",mu,nu,"}) = ", TrPmunu[0,0,0,0])
g.message("\nTr(g5 P_{",mu,nu,"}) = ", Trg5Pmunu[0,0,0,0])

g.message("\nP_{",mu,",",nu,"} [0,0,0,0] = ",Pmunu[0,0,0,0])
for a in range(0, 4):
    for b in range(0, 4):
        g.message("\nP_{",mu,",",nu,"}^{",a,",",b,"} [0,0,0,0] = ",g.trace(Pmunu[0,0,0,0]*g.gamma[a]*g.gamma[b])/4)
g.message("\nP_{",mu,",",nu,"}^{5} [0,0,0,0] = ",g.trace(Pmunu[0,0,0,0]*g.gamma[5])/4)

def not_field_strength(U, mu, nu):
    assert mu != nu
    # v = staple_up - staple_down
    v = g.eval(
        g.cshift(U[nu], mu, 1) * g.adj(g.cshift(U[mu], nu, 1)) * g.adj(U[nu])
        + g.cshift(g.adj(g.cshift(U[nu], mu, 1)) * g.adj(U[mu]) * U[nu], nu, -1)
    )

    F = g.eval(U[mu] * v + g.cshift(v * U[mu], mu, -1))
    F @= 0.125 * (F + g.adj(F))
    return F

Hmunu = not_field_strength(U, mu, nu)
TrHmunu = g.eval( g.trace(Hmunu) )
Trg5Hmunu = g.eval( g.trace(Hmunu * g.gamma[5]) )
g.message("\nTr(H_{",mu,nu,"}) = ", TrHmunu[0,0,0,0])
g.message("\nTr(g5 H_{",mu,nu,"}) = ", Trg5Hmunu[0,0,0,0])

g.message("\nH_{",mu,",",nu,"} [0,0,0,0] = ",Hmunu[0,0,0,0])
for a in range(0, 4):
    for b in range(0, 4):
        g.message("\nH_{",mu,",",nu,"}^{",a,",",b,"} [0,0,0,0] = ",g.trace(Hmunu[0,0,0,0]*g.gamma[a]*g.gamma[b])/4)
g.message("\nH_{",mu,",",nu,"}^{5} [0,0,0,0] = ",g.trace(Hmunu[0,0,0,0]*g.gamma[5])/4)

# build Riemann tensor
n_Riemann = [ [ [ [ g.real(grid) for sigma in range(0,4) ] for rho in range(0,4) ] for nu in range(0,4) ] for mu in range(0,4) ]

for mu in range(0, 4):
    for nu in range(0, 4):
        for rho in range(0, 4):
            for sigma in range(0, 4):
                if mu == nu:
                    n_Riemann[mu][nu][rho][sigma][:] = 0
                else:
                    Hmunu = not_field_strength(U, mu, nu)
                    n_Riemann[mu][nu][rho][sigma] = g.eval( g.trace(Hmunu * eslash[rho] * eslash[sigma]) )

g.message("\nbuilt not Riemann tensor")


g.message("\nnot Riemann tensor 0 1 2 3 = ", n_Riemann[0][1][2][3][0,0,0,0])

g.message("\nnot Riemann tensor 0 1 3 2 = ", n_Riemann[0][1][3][2][0,0,0,0])

g.message("\nnot Riemann tensor 1 0 2 3 = ", n_Riemann[1][0][2][3][0,0,0,0])

g.message("\nnot Riemann tensor 2 3 0 1 = ", n_Riemann[2][3][0][1][0,0,0,0])
