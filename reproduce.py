#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:22:08 2021

@author: tsm
"""
import numpy as np
import scipy as sp
import scipy.linalg 
import matplotlib.pyplot as plt
from numpy import pi
import timeit
import matplotlib

from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import  boson_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#parameter 
L = 29;
N = L*2;

J = 1;
delta_0 = 0.8;
Delta_0 = 2;
U = 30
theta_0 = 0;
omega = 0.005;
phi_0 = 0;

T = 2*pi/omega;

N_T = 500;

t_total = np.linspace(0,T,N_T)
dt= t_total[1] - t_total[0]


loc_total = np.zeros(N_T)

basis= boson_basis_1d(N,Nb=2)


def delta_t(t):
    return delta_0*np.sin(omega*t + phi_0)

def Delta_t(t):
    return Delta_0*np.cos(omega*t + phi_0)

#Get Hamitonian matrix 
def Hami(J,delta,Delta,U):
    hop=[[-J-delta*(-1)**i,i,(i+1)%N] for i in range(N)] # PBC
    stagg_pot=[[Delta*(-1)**i,i] for i in range(N)]
    inter_1 = [[U/2,i, i ] for i in range(N)] # PBC
    inter_2 = [[-U/2,i,  ] for i in range(N)] # PBC
    
    static=[["+-",hop],["-+",hop],['n',stagg_pot],['nn',inter_1],['n',inter_2]]
    
    H=hamiltonian(static,[],basis=basis,dtype=np.float64)
    H_matrix = H.toarray()
    return H_matrix


#initial state 

i0 = basis.index("0"*28+"2" + "0"*29) # pick state from basis set
#print(basis)
print(i0)
psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0 # define state corresponding to the string "111000"


#location opeartor 
couple_loc = [[i,i] for i in range(0,N)]
static_loc = [['n',couple_loc]]
H_loc = hamiltonian(static_loc,[],basis=basis,dtype=np.float64)
loc_matrix = H_loc.toarray()/2
ini_loc = np.real( psi.T.conj() @ loc_matrix @ psi)


#start the time evolution
for idx,t in enumerate(t_total):
    
    Hami_m = Hami(J, delta_t(t), Delta_t(t), U)
    
    psi = scipy.linalg.expm(-1j*Hami_m*dt) @psi
    
    loc_total[idx] = np.real( psi.T.conj() @ loc_matrix @ psi) - ini_loc
    
    print(idx)
    print("loc = " + str(loc_total[idx] ))


#np.savetxt("location.txt",loc_total)    
    
plt.figure()
plt.plot(t_total,loc_total/2,"-r")
plt.xlabel("t")
plt.ylabel("Displacement")

              



