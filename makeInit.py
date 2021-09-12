from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
import numpy as np
import matplotlib.pyplot as plt
import copy

#choose odd L!
L=31

N=2*L

J=1
delta_0=0.8

Delta_0=2

U=10

theta_0=0
omega=0.005

phi_0=0
basisAll=boson_basis_1d(N,Nb=2)

def Hami(J,delta,Delta,U):
    hop=[[-J-delta*(-1)**i,i,(i+1)%N] for i in range(N)]
    stagg_pot=[[Delta*(-1)**i,i] for i in range(N)]
    inter_1=[[U/2,i,i] for i in range(N)]
    inter_2=[[-U/2,i] for i in range(N)]

    static=[["+-",hop],["-+",hop],["n",stagg_pot],["nn",inter_1],["n",inter_2]]
    H=hamiltonian(static,[],basis=basisAll,dtype=np.float64)

    return H.toarray()


H_m=Hami(J,delta_0,Delta_0,U)

def translate(stateStr):
    return stateStr[-2:]+stateStr[:-2]

def get_state(state_fock_str):
    '''

    :param state_fock_str:
    :return: vec repr
    '''
    i0=basisAll.index(state_fock_str)
    psi=np.zeros(basisAll.Ns,dtype=np.complex128)
    psi[i0]=1.0
    return psi

def idx_to_fock(idx):
    state_num=basisAll[idx]

    state=basisAll.int_to_state(state_num)
    state=state[1:-1]

    state=state.replace(" ","")
    return state

def  idx_to_state(idx):
    fock=idx_to_fock(idx)
    return get_state(fock)


N_H=basisAll.Ns
total_set=[]
for idx in range(N_H):
    total_set.append(idx_to_fock(idx))

seed_state=[]
redundant_state=[]

for state in total_set:
    if state in redundant_state:
        continue
    seed_state.append(state)

    for idx in range(1,L):

        state=translate(state)
        redundant_state.append(state)

def ortho_basis(k,str_n):
    new_str=copy.copy(str_n)
    new_state=get_state(str_n)

    for idx in range(1,L):
        new_str=translate(new_str)
        new_state+=np.exp(1j*2*idx*k)*get_state(new_str)
    new_state/=np.sqrt(L)
    return new_state


def H_k(k):
    H_mat = np.zeros([ int(N_H/L),int(N_H/L)],dtype = complex)
    for i,i_fock in enumerate(seed_state):
        for j,j_fock in enumerate(seed_state):
            i_state = ortho_basis(k,i_fock)
            j_state = ortho_basis(k,j_fock)
            H_mat[i,j] = i_state.conj().T @ H_m @ j_state
    return H_mat


#basis in |n> space
Br=[]
for stateTmp in seed_state:
    Br.append(stateTmp)
    for j in range(1,L):
        Br.append(translate(Br[-1]))



k0=0
kValsAll=[2*np.pi*(l-L/2)/(2*L) for l in range(0,L)]
H0Mat=H_k(k0)
w,v=np.linalg.eig(H0Mat)

idx=w.argsort()[::-1]
eigenValues=w[idx]
eigenVectors=v[:,idx]


j0=L
sgm=5
def f(k):
    return np.exp(-k**2*sgm**2+1j*j0*k)

n0=0
uValsAll=eigenVectors[:,n0]

psiInit=[]
Ds=int(N_H/L)
for b in range(0,Ds):
    for j in range(0,L):
        sumTmp=0
        for a in range(0,L):
            sumTmp+=f(kValsAll[a])*np.exp(1j*kValsAll[a]*2*j)
        sumTmp*=uValsAll[b]
        psiInit.append(sumTmp)


def l2Norm(vec):
    '''

    :param vec: input vector
    :return: l2 norm of vec
    '''
    tmp=0
    for j in range(0,len(vec)):
        tmp+=np.abs(vec[j])**2
    return np.sqrt(tmp)

psiInit/=l2Norm(psiInit)
psiVec=np.zeros(len(psiInit),dtype=complex)

for j in range(0,len(psiInit)):
    stateStrTmp=Br[j]
    psiVec+=psiInit[j]*get_state(stateStrTmp)

for x in range(N):
    coupling=[[1,x]]
    static_n=[["n",coupling]]
    exec(f"H_n_{x}_H=hamiltonian(static_n,[],dtype=np.complex128,basis=basisAll)")
    exec(f"H_n_{x}=H_n_{x}_H.toarray()")

density_total=np.zeros(N)
for x in range(N):
    exec(f"density_total[x]=psiVec.conj().T@H_n_{x}@psiVec")

plt.figure()
plt.plot(np.arange(N),np.abs(density_total))
plt.savefig("tmp2.png")
