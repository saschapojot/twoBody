from consts import *




def delta(t):
    return d0 * np.sin(omega * t + phi0)

def D(t):
    return D0 * np.cos(omega * t + phi0)

def FloquetDriving(t):
    return V*np.cos(Omega*t)

def HamiltonianMat(t):
    """

    :param t:
    :return: Hamiltonian matrix at time t, without tilted potential
    """
    deltaVal=delta(0)
    DeltaVal=D(0)
    drivingVal=V*np.cos(Omega*t)

    hop=[[-J+deltaVal*(-1)**m,m,(m+1)%N] for m in range(0,N)]
    stagPot=[[DeltaVal*(-1)**m+drivingVal*(-1)**m,m] for m in range(0,N)]
    interN=[[-U/2,m] for m in range(0,N)]
    interNN=[[U/2,m,m] for m  in range(0,N)]
    static=[["+-",hop],["-+",hop],["n",stagPot],["n",interN],["nn",interNN]]

    HTmp=hamiltonian(static,[],basis=basisAll,dtype=np.complex128)

    return HTmp.toarray()


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


def H_k(t,kVal):
    """

    :param t: time
    :param kVal: momentum
    :return: Hamiltonian in momentum space, block kVal, at time t
    """
    HFockMat=HamiltonianMat(t)

    retHkMat=np.zeros((int(N_H/L),int(N_H/L)),dtype=np.complex128)
    for i, i_fock_str in enumerate(seed_list):
        for j, j_fock_str in enumerate(seed_list):
            i_state_vec=ortho_basis(kVal,i_fock_str)
            j_state_vec=ortho_basis(kVal,j_fock_str)
            retHkMat[i,j]=i_state_vec.conj().T @HFockMat@j_state_vec
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


def Vj(tj,kVal):
    """

    :param tj: time
    :param kVal:
    :return: exp(-i Delta t H(tj,k))
    """

    return slin.expm(-1j*deltat*H_k(tj,kVal))



tFlqValsAll=[deltat*j for j in range(0,QFlq)]

def UMat(kVal):
    """

    :param kVal:
    :return: Floquet operator for momentum kVal
    """
    VList=[]#sorted by descending order of time
    for tTmp in tFlqValsAll[::-1]:
        VList.append(Vj(tTmp,kVal))

    retUMat=np.eye(int(N_H/L),dtype=np.complex128)
    for VjMatTmp in VList:
        retUMat=retUMat@VjMatTmp
    return retUMat

def eigPhasesAndEigVecs(kVal):
    """

    :param k:
    :return: eigenphases and eigenvectors, sorted by eigenphases
    """
    UMatVal=UMat(kVal)
    eigVals,eigVecs=np.linalg.eig(UMatVal)
    eigPhases=np.array([-np.angle(elem) for elem in eigVals])
    inds=eigPhases.argsort()
    return eigPhases[inds],eigVecs[:,inds]

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
eigPhaseVecPairsAll=[]
for kTmp in kValsAll:
    phasesTmp,vecsTmp=eigPhasesAndEigVecs(kTmp)
    eigPhaseVecPairsAll.append([phasesTmp,vecsTmp])

bandNum=0
j0=L/1.5
init_state=np.zeros(N_H,dtype=complex)
for j in range(0,len(eigPhaseVecPairsAll)):
    vecTmp=eigPhaseVecPairsAll[j][1][bandNum]
    kTmp=kValsAll[j]
    init_state+=ortho_to_fockvec(kTmp,vecTmp)*np.exp(1j*2*kTmp*j0)*g(kTmp,0)

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
for j in range(0,len(eigPhaseVecPairsAll)):
    kTmp=kValsAll[j]
    for phaseTmp in eigPhaseVecPairsAll[j][0]:

        pltKValsAll.append(kTmp)
        pltPhasesAll.append(phaseTmp)

plt.figure()
plt.scatter(pltKValsAll,pltPhasesAll,color="black",s=1)
plt.savefig("tmp3.png")
plt.close()