import numpy as np

import scipy.linalg as slin

import matplotlib.pyplot as plt

from datetime import datetime
from quspin.basis import boson_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.block_tools import block_diag_hamiltonian
import copy

#L is odd
L = 19
N = 2 * L
J = 1

#delta_0
d0 = 0.8

#Delta_0
D0 = 2

U = 10

omega = 0.003
omegaF =0.005
phi0 = 0

n0=100
#driving params
Omega=n0*omega
n1=1
alpha=0
A=U*5
#############
#quench parameters
B1=3
B2=5
########
basisAll = boson_basis_1d(N, Nb=2)

T = 2 * np.pi / omega

tTot =  3*T
Q =500
dt = tTot / Q

sgm=2
# i0=basisAll.index("0"*(L-1)+"2"+"0"*L)
# # i0=basisAll.index("0"*(L-2)+"11"+"0"*L)
# psi0=np.zeros(basisAll.Ns,dtype=np.complex128)
# psi0[i0]=1




# def H1Mat():
#     onSite2 = [[U / 2, j, j] for j in range(0, N)]
#     onSite1 = [[-U / 2, j] for j in range(0, N)]
#     tiltedPot = [[omegaF * j, j] for j in range(0, N)]
#     staticPart = [["n", onSite1], ["n", tiltedPot], ["nn", onSite2]]
#
#     H1Tmp = hamiltonian(staticPart, [], basis=basisAll, dtype=np.complex128, check_symm=False,check_herm=False)
#     H1MatTmp = H1Tmp.toarray()
#     return H1MatTmp
#
#
# H1MatVal = H1Mat()
# U1 = slin.expm(-1j * dt / 2 * H1MatVal)


#position operator
posVals=[[j,j]for j in range(0,N)]
posList=[["n",posVals]]
xOpr=hamiltonian(posList,[],basis=basisAll,dtype=np.complex128)
xMat=xOpr.toarray()/2
# E,V=np.linalg.eigh(xMat)
# print(V)
# print(E)
#construct gaussian wavepacket



