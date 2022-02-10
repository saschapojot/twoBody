from consts import *



def delta(t):
    return d0 * np.sin(omega * t + phi0)

def D(t):
    return D0 * np.cos(omega * t + phi0)



#delta(0) and Delta(0)
deltaVal0=delta(0)
DeltaVal0=D(0)
#construct Hs(0)
hop0=[[-J+deltaVal0*(-1)**m,m,(m+1)%N] for m in range(0,N)]
stagPot0=[[DeltaVal0*(-1)**m,m] for m in range(0,N)]
interN0=[[-U/2,m] for m in range(0,N)]
interNN0=[[U/2,m,m] for m  in range(0,N)]
static0=[["+-",hop0],["-+",hop0],["n",stagPot0],["n",interN0],["nn",interNN0]]
Hs0=hamiltonian(static0,[],basis=basisAll,dtype=np.complex128).toarray()
#construct H1= Hs(0)+Hq
hop1=[[-J+B1+deltaVal0*(-1)**m,m,(m+1)%N] for m in range(0,N)]
stagPot1=stagPot0
interN1=interN0
interNN1=interNN0
static1=[["+-",hop1],["-+",hop1],["n",stagPot1],["n",interN1],["nn",interNN1]]
H1=hamiltonian(static1,[],basis=basisAll,dtype=np.complex128).toarray()

#Floquet operator
U=slin.expm(-1j*1/2*Tq*H1)@slin.expm(-1j*1/2*Tq*Hs0)
#effective Hamiltonian
Heff=1j/Tq*slin.expm(U)

def translate(stateStr):
    '''

    :param stateStr: Fock state basis in string
    :return: string of Fock state after cotranslation
    '''
    return stateStr[-2:] + stateStr[:-2]


def get_state(state_fock_str):
    '''

    :param state_fock_str: fock state basis in string
    :return: vec repr of fock state basis
    '''
    i0 = basisAll.index(state_fock_str)
    psi = np.zeros(basisAll.Ns, dtype=np.complex128)
    psi[i0] = 1.0
    return psi
def idx_to_fock(idx):
    '''

    :param idx: 0,1,...,Ns-1
    :return: the idx th Fock state in string
    '''
    state_num = basisAll[idx]#is an int

    state = basisAll.int_to_state(state_num)
    state = state[1:-1]

    state = state.replace(" ", "")
    return state
def idx_to_state(idx):
    '''

    :param idx: 0,1,...,Ns-1
    :return: vector representation of idxth state
    '''
    fock = idx_to_fock(idx)
    return get_state(fock)

N_H=basisAll.Ns
total_set=[]
for idx in range(0,N_H):
    total_set.append(idx_to_fock(idx))

seed_list=[]
redundant_list=[]
for stateStr in total_set:
    if stateStr in redundant_list:
        continue
    else:
        seed_list.append(stateStr)
    for idx in range(1,L):
        stateStr=translate(stateStr)
        redundant_list.append(stateStr)

def ortho_basis(kVal, str_n):
    """
    Fock space to momentum space vector
    :param kVal:
    :param str_n: fock space string in seed_list
    :return: |k,n> vector
    """

    new_str=copy.deepcopy(str_n)
    ret_state_vec=get_state(str_n)
    for idx in range(1,L):
        new_str=translate(new_str)
        ret_state_vec+=np.exp(1j*2*idx*kVal)*get_state(new_str)
    ret_state_vec/=np.sqrt(L)
    return ret_state_vec


def H_k(kVal):
    """

    :param kVal:
    :return: Hamiltonian in momentum space with momentum kVal
    """
    retHkMat = np.zeros((int(N_H / L), int(N_H / L)), dtype=np.complex128)
    for i, i_fock_str in enumerate(seed_list):
        for j, j_fock_str in enumerate(seed_list):
            i_state_vec = ortho_basis(kVal, i_fock_str)
            j_state_vec = ortho_basis(kVal, j_fock_str)
            retHkMat[i, j] = i_state_vec.conj().T @ Heff @ j_state_vec
    return retHkMat

def ortho_to_fockvec(k,vec_in_k):
    """

    :param k:
    :param vec_in_k: vector in k space, solved from H(k)
    :return: vector in Fock space
    """
    ret_fock_vec=np.zeros(N_H,dtype=np.complex128)
    for idx in range(0,len(vec_in_k)):
        ret_fock_vec+=vec_in_k[idx]*ortho_basis(k,seed_list[idx])
    return ret_fock_vec



def eigValsAndEigVecs(kVal):
    """

    :param kVal:
    :return: eigenvalues and eigenvectors of Block H_k(kVal),
    sorted by eigenvalues
    """

    HKMat=H_k(kVal)

    eigVals, eigVecs=np.linalg.eigh(HKMat)
    inds=eigVals.argsort()
    return eigVals[inds],eigVecs[:,inds]

def avgPos(psiq):
    '''

    :param psiq: wv func
    :return: average position
    '''
    return psiq.T.conj().dot(xMat).dot(psiq)
def g(k,k0):
    """

    :param k:
    :param k0:
    :return: gaussian factor
    """
    return np.exp(-(k - k0) ** 2 *sgm**2)


tStart=datetime.now()
kValsLen=L
kIndAll = list(range(0, kValsLen))
kValsAll = [2 * np.pi * (l-kValsLen/2) / (2 * kValsLen) for l in kIndAll]
eigValsVecPairsAll=[]
for kTmp in kValsAll:
    valTmp,vecsTmp=eigValsAndEigVecs(kTmp)
    eigValsVecPairsAll.append([valTmp,vecsTmp])

bandNum=0
j0=L/2
init_state=np.zeros(N_H,dtype=complex)
for j in range(0,len(eigValsVecPairsAll)):
    vecTmp=eigValsVecPairsAll[j][1][bandNum]
    kTmp=kValsAll[j]
    init_state+=ortho_to_fockvec(kTmp,vecTmp)*np.exp(-1j*2*kTmp*j0)*g(kTmp,0)

init_state/=np.sqrt(init_state.conj().T@init_state)
psi0Vec=init_state

for x in range(N):
    coupling= [[1,x]]
    static_n = [['n',coupling]]
    exec(f'H_n_{x}_H = hamiltonian(static_n,[], dtype=np.complex128,basis=basisAll)')
    exec(f'H_n_{x} = H_n_{x}_H.toarray()')


density_total=np.zeros(N)

for x in range(N):
    exec(f'density_total[x] = init_state.conj().T @ H_n_{x} @ init_state')

plt.figure()
plt.plot(np.arange(N), np.abs(density_total))
plt.savefig("tmp2.png")
plt.close()
tEnd=datetime.now()
print(" init time: ",tEnd-tStart)

pltKValsAll=[]
pltPhasesAll=[]
for j in range(0,len(eigValsVecPairsAll)):
    kTmp=kValsAll[j]
    for phaseTmp in eigValsVecPairsAll[j][0]:

        pltKValsAll.append(kTmp)
        pltPhasesAll.append(phaseTmp)

plt.figure()
plt.scatter(pltKValsAll,pltPhasesAll,color="black",s=1)
plt.savefig("tmp3.png")
plt.close()