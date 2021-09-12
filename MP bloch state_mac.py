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
import copy

from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import  boson_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#parameter 
L = 35;
N = L*2;

J = 1;
delta_0 = 0.8;
Delta_0 = 2;
U = 10
theta_0 = 0;
omega = 0.005;
phi_0 = 0;

T = 2*pi/omega;

N_T = 500;

#t_total = np.linspace(0,T,N_T)
#dt= t_total[1] - t_total[0]


loc_total = np.zeros(N_T)

basis= boson_basis_1d(N,Nb=2)

k_total = np.linspace(0,pi,L)

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


H_m = Hami(J,0,2,U)

#initial state 

# i0 = basis.index("0"*9+"2" + "0"*9) # pick state from basis set
# #print(basis)
# print(i0)
# psi = np.zeros(basis.Ns,dtype=np.float64)
# psi[i0] = 1.0 # defin


#Eqn 1
def translate(str):
    return str[-2:] + str[:-2]

def get_state(state_fock):
    """"
    get the wavefunction given the fock state representation
    """
    i0 = basis.index(state_fock)
    psi = np.zeros(basis.Ns,dtype=np.complex64)
    psi[i0] = 1.0
    return psi

def idx_to_fock(idx):
    """
    get the fock representation given the index from 0 to N_H
    """
    state_num = basis[idx]
    state = basis.int_to_state(state_num)
    state = state[1:-1]
    state = state.replace(" ", "")
    return state

def idx_to_state(idx):
    """"
    get the wavefunction given the index from 0 to N_H
    """
    fock = idx_to_fock(idx)
    return get_state(fock)

#get the total set 

N_H = basis.Ns

#total fock state 
total_set = []
for idx in range(N_H):
    total_set.append(idx_to_fock(idx))

#selet the independent seed stata
seed_state = []
redudant_state = []

for state in total_set:
    if state in redudant_state:
        continue
    seed_state.append(state)
    
    for idx in range(1,L):
        state = translate(state)
        redudant_state.append(state)

        
#Eqn 2
def ortho_basis(k,str_n):
    # total_set = {str_n,}
    new_str = copy.copy(str_n)
    
    new_state = get_state(str_n)
    
    for idx in range(1,L):
        new_str = translate(new_str)
        # if new_str in total_set:
        #     break
        new_state += np.exp(1j*2*idx*k)*get_state(new_str)
        
        # total_set.add(new_str)
#        print(total_set)
        
    new_state = new_state / np.sqrt(L)
    return new_state




eigen_total = np.zeros([L,int(N_H/L)])
print("N_H/L="+str(N_H/L))
#build H_k

#Eqn 3
def H_k(k):
    H_mat = np.zeros([ int(N_H/L),int(N_H/L)],dtype = complex)
    for i,i_fock in enumerate(seed_state):
        for j,j_fock in enumerate(seed_state):
            i_state = ortho_basis(k,i_fock)
            j_state = ortho_basis(k,j_fock)
            H_mat[i,j] = i_state.conj().T @ H_m @ j_state
    return H_mat

    
#Eqn 4
for idx_k,k in enumerate(k_total):
    print("Finish " + str(100*idx_k/L ) + "%")
    H_k_mat = H_k(k)
    eigen_v =  np.sort(np.linalg.eigvals(H_k_mat))
    eigen_total[idx_k,:] = np.real(eigen_v)
   
    
#Fig 2   
plt.figure()
plt.plot(k_total,eigen_total ,".")
    
#The Block state
k0 = k_total[-5]
H_test = H_k(k0)

w,v = np.linalg.eig(H_test)
idx = w.argsort()[::-1]   
eigenValues = w[idx]
eigenVectors = v[:,idx]

choose = 0

eigen_v = eigenVectors[:,choose]

eigen_v_fock = np.zeros(N_H,dtype = complex)
for idx in range(len(eigen_v)):
    eigen_v_fock += eigen_v[idx] * ortho_basis(k0,seed_state[idx])

#plot the density distribution 
for x in range(N):
    coupling= [[1,x]]
    static_n = [['n',coupling]]
    exec(f'H_n_{x}_H = hamiltonian(static_n,[], dtype=np.complex128,basis=basis)')
    exec(f'H_n_{x} = H_n_{x}_H.toarray()')

density_total = np.zeros(N)

for x in range(N):
    exec(f'density_total[x] = eigen_v_fock.conj().T @ H_n_{x} @ eigen_v_fock')
    

plt.figure()
plt.bar(np.arange(N), density_total)
plt.show()
plt.close()