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
L = 25;
N = L*2;

J = 1;
delta_0 = 0.8;
Delta_0 = 2;
U = 10
theta_0 = 0;
omega = 0.005;
phi_0 = 0;

T = 2*pi/omega;

N_T = 2000;


wf = 5*omega


t_total = np.linspace(0,T,N_T)
dt= t_total[1] - t_total[0]


loc_total = np.zeros(N_T)

basis= boson_basis_1d(N,Nb=2)

k_total = np.linspace(0,pi,L)


# plt.figure()
# plt.plot(np.arange(L), gaussian(j0,sigma,np.arange(L)))



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
    
    H=hamiltonian(static,[],basis=basis,dtype=np.complex128)
    H_matrix = H.toarray()
    return H_matrix


H_m = Hami(J,0,2,U)

#initial state 

# i0 = basis.index("0"*9+"2" + "0"*9) # pick state from basis set
# #print(basis)
# print(i0)
# psi = np.zeros(basis.Ns,dtype=np.float64)
# psi[i0] = 1.0 # defin

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



N_k_H = int(N_H/L)

#build H_k


def H_k(k):
    H_mat = np.zeros([ N_k_H,N_k_H],dtype = complex)
    for i,i_fock in enumerate(seed_state):
        for j,j_fock in enumerate(seed_state):
            i_state = ortho_basis(k,i_fock)
            j_state = ortho_basis(k,j_fock)
            H_mat[i,j] = i_state.conj().T @ H_m @ j_state
    return H_mat

    
#save all the H_k and all the eigenvalue eigenvector 
All_H_k = np.zeros([L,N_k_H,N_k_H],dtype = complex)
All_eigen_value  = np.zeros([L, N_k_H])
All_eigen_vector = np.zeros([L, N_k_H,N_k_H],dtype = complex)

for idx_k,k in enumerate(k_total):
    print("Finish " + str(100*idx_k/L ) + "%")
    H_k_mat = H_k(k)
    All_H_k[idx_k,:,:] = H_k_mat
    
    eigen_value,eigen_vector =  np.linalg.eig(H_k_mat)
    idx = eigen_value.argsort()[::-1]    

    All_eigen_value[idx_k,:] = eigen_value[idx]
    
    
    All_eigen_vector[idx_k,:,:] = eigen_vector[:,idx]
    
   
plt.figure()
plt.plot(k_total,All_eigen_value ,".")
 

#gauge smooth 

#loop over each band
for idx_m in range(N_k_H):
    #loop over k
    for idx_s in range(L-1):
        phase_diff = np.imag(np.log(np.vdot(All_eigen_vector[idx_s,:,idx_m ], All_eigen_vector[idx_s+1,:,idx_m])))
            #update the eigenvetor the adjacent phase -> 0
        All_eigen_vector[idx_s+1,:,idx_m] = np.exp(-1j*phase_diff)*All_eigen_vector[idx_s+1,:,idx_m]
    total_diff = np.imag(np.log(np.vdot(All_eigen_vector[0,:,idx_m ], All_eigen_vector[-1,:,idx_m])))
    phase_avr = total_diff/(L-1)
    for idx_s in range(L-1):
        All_eigen_vector[idx_s,:,idx_m] =  np.exp(1j*phase_avr*idx_s)*All_eigen_vector[idx_s,:,idx_m]
        print("idx_s is " +str(idx_s))
#The Block state

# k0 = k_total[-5]
# H_test = H_k(k0)

# w,v = np.linalg.eig(H_test)
# idx = w.argsort()[::-1]   
# eigenValues = w[idx]
# eigenVectors = v[:,idx]



# eigen_v = eigenVectors[:,choose]

# eigen_v_fock = np.zeros(N_H,dtype = complex)
# for idx in range(len(eigen_v)):
#     eigen_v_fock += eigen_v[idx] * ortho_basis(k0,seed_state[idx])


    
choose = 0

#ortho basis to fock basis

def ortho_to_fock(k,state):
        fock_state = np.zeros(N_H,dtype = complex)
        for idx in range(len(state)):
            fock_state += state[idx] * ortho_basis(k,seed_state[idx])
        return fock_state

    #plot the density distribution 
for x in range(N):
    coupling= [[1,x]]
    static_n = [['n',coupling]]
    exec(f'H_n_{x}_H = hamiltonian(static_n,[], dtype=np.complex128,basis=basis)')
    exec(f'H_n_{x} = H_n_{x}_H.toarray()')


# #plot the density distribution of all the Bloch state 
# for idx_k1,k0 in enumerate(k_total):
        
#     state = np.zeros(N_H,dtype = complex)
    
        
            
#     state = ortho_to_fock(k0, All_eigen_vector[idx_k1,:,choose])   
    
#     #plot the density distribution 
#     for x in range(N):
#         coupling= [[1,x]]
#         static_n = [['n',coupling]]
#         exec(f'H_n_{x}_H = hamiltonian(static_n,[], dtype=np.complex128,basis=basis)')
#         exec(f'H_n_{x} = H_n_{x}_H.toarray()')
    
#     density_total = np.zeros(N)
    
#     for x in range(N):
#         exec(f'density_total[x] = state.conj().T @ H_n_{x} @ state')
        
    
#     plt.figure()
#     plt.bar(np.arange(N), density_total)
#     plt.title(str(k0))

# #get the gaussian state

# state = np.zeros(N_H,dtype = complex)
sigma = 0.5
k0 = pi
def gaussian(k):
        return np.exp(-(k-k0)**2/(4*sigma**2))
# for idx_k, k in enumerate(k_total):
#     eigen_state = All_eigen_vector[idx_k,:,choose]
#     state += ortho_to_fock(k, eigen_state)*gaussian(k)

# density_total = np.zeros(N)
    
# for x in range(N):
#     exec(f'density_total[x] = state.conj().T @ H_n_{x} @ state')
    

# plt.figure()
# plt.bar(np.arange(N), density_total)
# plt.title(str(k0))


#Wannier 



w_state = np.zeros(N_H,dtype = complex)
for idx_k, k in enumerate(k_total):
    eigen_state = All_eigen_vector[idx_k,:,choose]
    w_state += ortho_to_fock(k, eigen_state)*np.exp(-1j*2*k*9)*gaussian(k)

density_total = np.zeros(N)
    
w_state = w_state / np.sqrt(w_state.conj().T @ w_state)
psi =  w_state


for x in range(N):
    exec(f'density_total[x] = w_state.conj().T @ H_n_{x} @ w_state')
    

plt.figure()
plt.bar(np.arange(N), density_total)


# i0 = basis.index("0"*(L-1)+"2" + "0"*L) # pick state from basis set
# #print(basis)
# print(i0)
# psi = np.zeros(basis.Ns,dtype=np.float64)
# psi[i0] = 1.0 # define state corresponding to the string "111000"


#Get Hamitonian matrix 
def Hami_linear(J,delta,Delta,U):
    hop=[[-J-delta*(-1)**i,i,(i+1)] for i in range(N-1)] # PBC
    stagg_pot=[[Delta*(-1)**i,i] for i in range(N)]
    inter_1 = [[U/2,i, i ] for i in range(N)] # PBC
    inter_2 = [[-U/2,i   ] for i in range(N)] # PBC
    linear =  [[wf*i,i   ] for i in range(N)] # PBC
    
    static=[["+-",hop],["-+",hop],['n',stagg_pot],['nn',inter_1],['n',inter_2],['n', linear]]
    #static=[["+-",hop],["-+",hop],['n',stagg_pot],['nn',inter_1],['n', linear]]

    
    H=hamiltonian(static,[],basis=basis,dtype=np.float64)
    H_matrix = H.toarray()
    return H_matrix


#initial state 



#location opeartor 
couple_loc = [[i,i] for i in range(0,N)]
static_loc = [['n',couple_loc]]
H_loc = hamiltonian(static_loc,[],basis=basis,dtype=np.float64)
loc_matrix = H_loc.toarray()/2
ini_loc = np.real( psi.T.conj() @ loc_matrix @ psi)


#start the time evolution
for idx,t in enumerate(t_total):
    Hami_m = Hami_linear(J, delta_t(t), Delta_t(t), U)
    psi = scipy.linalg.expm(-1j*Hami_m*dt) @psi
    loc_total[idx] = np.real( psi.T.conj() @ loc_matrix @ psi) - ini_loc
    print("Finish " + str(idx *100/ N_T) +"%")
    print("loc = " + str(loc_total[idx] ))


#np.savetxt("location.txt",loc_total)    
    
plt.figure()
plt.plot(t_total,loc_total/2,"-r")
plt.plot(t_total, np.ones(len(t_total)),"--")
plt.xlabel("t")
plt.ylabel("Displacement")
plt.xlim(0,t_total[-1])
plt.title("pumping U = 10 N_splite = 2000 sigma= 0.5 ")


              
for x in range(N):
    coupling= [[1,x]]
    static_n = [['n',coupling]]
    exec(f'H_n_{x}_H = hamiltonian(static_n,[], dtype=np.complex128,basis=basis)')
    exec(f'H_n_{x} = H_n_{x}_H.toarray()')
    
density_total = np.zeros(N)

for x in range(N):
    exec(f'density_total[x] = psi.conj().T @ H_n_{x} @ psi')
    

plt.figure()
plt.bar(np.arange(N), density_total)
plt.title("particle density with linear potential")
