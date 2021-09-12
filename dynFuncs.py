from consts import *


def delta(t):
    return d0 * np.sin(omega * t + phi0)


def Delta(t):
    return D0 * np.cos(omega * t + phi0)




def avgPos(psiq):
    '''

    :param psiq: wv func
    :return: average position
    '''
    return psiq.T.conj().dot(xMat).dot(psiq)


def reNormalization(vec):
    '''
    normalize vector
    :param vec: input vector
    :return:
    '''
    tmp2 = 0
    for elem in vec:
        tmp2 += np.abs(elem) ** 2
    tmp = np.sqrt(tmp2)
    vec /= tmp
    return vec


def selectInitVecFromBand(kOrder):
    '''

    :param kOrder: the kOrder th momentum
    :return:
    '''

    delta0=delta(0)
    Delta0=Delta(0)
    #construct Hamiltonian at t=0 without tilted potential
    hopPM=[[-J+(-1)**j*delta0,j,(j+1)%N] for j in range(0,N)]
    hopMP=[[-J+(-1)**j*delta0,j,(j+1)%N] for j in range(0,N)]
    stgPot=[[(-1)**j*Delta0,j] for j in range(0,N)]
    inter2=[[U/2,j,j]for j in range(0,N)]
    inter1=[[-U/2,j] for j in range(0,N)]

    staticPart=[["+-",hopPM],["-+",hopMP],["n",stgPot] ,["nn",inter2],["n",inter1]]

    # h0=hamiltonian(staticPart,[],basis=basisAll,dtype=np.complex128)
    # h0Mat=h0.toarray()
    # print("h0 = ",h0Mat)
    blocks=[dict(Nb=2,kblock=j,a=2)for j in range(0,N//2)]
    basis_args=(N,)
    FT,HBlock=block_diag_hamiltonian(blocks,staticPart,[],boson_basis_1d,basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
    HBlcArr=HBlock.toarray()
    FTArr=FT.toarray()




    # subMatrix=HBlockHamiltonian[[2*kOrder,2*kOrder+1]][:,[2*kOrder,2*kOrder+1]]
    ETmp,VTmp=np.linalg.eigh(HBlcArr)
    veckOrd=VTmp[:,kOrder]
    vecx=FTArr.dot(veckOrd)
    return vecx



def Hami(J,delta,Delta,U):
    '''

    :param J:
    :param delta:
    :param Delta:
    :param U:
    :return: Hamiltonian at t=0, without tilted potential
    '''
    hop=[[-J+delta*(-1)**i,i,(i+1)%N] for i in range(N)]
    stagg_pot=[[Delta*(-1)**i,i] for i in range(N)]
    inter_1=[[U/2,i,i] for i in range(N)]
    inter_2=[[-U/2,i] for i in range(N)]

    static=[["+-",hop],["-+",hop],["n",stagg_pot],["nn",inter_1],["n",inter_2]]
    H=hamiltonian(static,[],basis=basisAll,dtype=np.float64)

    return H.toarray()


def translate(stateStr):
    '''

    :param stateStr: Fock state basis in string
    :return: string of Fock state after cotranslation
    '''
    return stateStr[-2:]+stateStr[:-2]



def get_state(state_fock_str):
    '''

    :param state_fock_str: fock state basis in string
    :return: vec repr of fock state basis
    '''
    i0=basisAll.index(state_fock_str)
    psi=np.zeros(basisAll.Ns,dtype=np.complex128)
    psi[i0]=1.0
    return psi


def idx_to_fock(idx):
    '''

    :param idx: 0,1,...,Ns-1
    :return: the idx th Fock state in string
    '''
    state_num=basisAll[idx]

    state=basisAll.int_to_state(state_num)
    state=state[1:-1]

    state=state.replace(" ","")
    return state


def  idx_to_state(idx):
    '''

    :param idx: 0,1,...,Ns-1
    :return:
    '''
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


H_m=Hami(J,d0,D0,U)

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
def f(k):
    '''

    :param k: momentum
    :return: Gaussian factor
    '''
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
psi0Vec=np.zeros(len(psiInit),dtype=complex)

for j in range(0,len(psiInit)):
    stateStrTmp=Br[j]
    psi0Vec+=psiInit[j]*get_state(stateStrTmp)
