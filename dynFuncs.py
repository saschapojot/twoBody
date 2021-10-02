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

    delta0 = delta(0)
    Delta0 = Delta(0)
    # construct Hamiltonian at t=0 without tilted potential
    hopPM = [[-J + (-1) ** j * delta0, j, (j + 1) % N] for j in range(0, N)]
    hopMP = [[-J + (-1) ** j * delta0, j, (j + 1) % N] for j in range(0, N)]
    stgPot = [[(-1) ** j * Delta0, j] for j in range(0, N)]
    inter2 = [[U / 2, j, j] for j in range(0, N)]
    inter1 = [[-U / 2, j] for j in range(0, N)]

    staticPart = [["+-", hopPM], ["-+", hopMP], ["n", stgPot], ["nn", inter2], ["n", inter1]]

    # h0=hamiltonian(staticPart,[],basis=basisAll,dtype=np.complex128)
    # h0Mat=h0.toarray()
    # print("h0 = ",h0Mat)
    blocks = [dict(Nb=2, kblock=j, a=2) for j in range(0, N // 2)]
    basis_args = (N,)
    FT, HBlock = block_diag_hamiltonian(blocks, staticPart, [], boson_basis_1d, basis_args, np.complex128,
                                        get_proj_kwargs=dict(pcon=True))
    HBlcArr = HBlock.toarray()
    FTArr = FT.toarray()

    # subMatrix=HBlockHamiltonian[[2*kOrder,2*kOrder+1]][:,[2*kOrder,2*kOrder+1]]
    ETmp, VTmp = np.linalg.eigh(HBlcArr)
    veckOrd = VTmp[:, kOrder]
    vecx = FTArr.dot(veckOrd)
    return vecx


def Hami(J, delta, Delta, U):
    '''

    :param J:
    :param delta:
    :param Delta:
    :param U:
    :return: Hamiltonian at t=0, without tilted potential
    '''
    hop = [[-J + delta * (-1) ** i, i, (i + 1) % N] for i in range(N)]
    stagg_pot = [[Delta * (-1) ** i, i] for i in range(N)]
    inter_1 = [[U / 2, i, i] for i in range(N)]
    inter_2 = [[-U / 2, i] for i in range(N)]

    static = [["+-", hop], ["-+", hop], ["n", stagg_pot], ["nn", inter_1], ["n", inter_2]]
    H = hamiltonian(static, [], basis=basisAll, dtype=np.float64)

    return H.toarray()


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
    state_num = basisAll[idx]

    state = basisAll.int_to_state(state_num)
    state = state[1:-1]

    state = state.replace(" ", "")
    return state


def idx_to_state(idx):
    '''

    :param idx: 0,1,...,Ns-1
    :return:
    '''
    fock = idx_to_fock(idx)
    return get_state(fock)



tStart=datetime.now()

N_H = basisAll.Ns
total_set = []
for idx in range(N_H):
    total_set.append(idx_to_fock(idx))

seed_state = []
redundant_state = []

for state in total_set:
    if state in redundant_state:
        continue
    seed_state.append(state)

    for idx in range(1, L):
        state = translate(state)
        redundant_state.append(state)


def ortho_basis(k, str_n):
    "to momentum space"
    """
    :return |k,n>
    """
    new_str = copy.copy(str_n)
    new_state = get_state(str_n)

    for idx in range(1, L):
        new_str = translate(new_str)
        new_state += np.exp(1j * 2 * idx * k) * get_state(new_str)
    new_state /= np.sqrt(L)
    return new_state


H_m = Hami(J, delta(0), Delta(0), U)


def H_k(k):
    H_mat = np.zeros([int(N_H / L), int(N_H / L)], dtype=complex)
    for i, i_fock in enumerate(seed_state):
        for j, j_fock in enumerate(seed_state):
            i_state = ortho_basis(k, i_fock)
            j_state = ortho_basis(k, j_fock)
            H_mat[i, j] = i_state.conj().T @ H_m @ j_state
    return H_mat


def ortho_to_fock(k,state):
    '''

    :param k: momentum
    :param state: vector in momentum space
    :return: vector in real space
    '''
    fock_state=np.zeros(N_H,dtype=complex)
    for idx in range(0,len(state)):
        fock_state+=state[idx]*ortho_basis(k,seed_state[idx])
    return fock_state

def g(k,k0):
    """

    :param k:
    :param k0:
    :return: gaussian factor
    """
    return np.exp(-(k - k0) ** 2 *sgm**2)
j0 = L/2

Ds = int(N_H / L)
def f(k):
    '''

    :param k: momentum
    :return: Gaussian factor
    '''
    return np.exp(-k ** 2 * sgm ** 2 + 1j *j0 * k)
# basis in |n> space
# Br = []
# for stateTmp in seed_state:
#     Br.append(stateTmp)
#     for j in range(1, L):
#         Br.append(translate(Br[-1]))

# k0=0
# kValsAll=[2*np.pi*(l-L/2)/(2*L) for l in range(0,L)]
# H0Mat=H_k(k0)
# w,v=np.linalg.eig(H0Mat)
#
# # idx=w.argsort()[::-1]
# idx=w.argsort()
# eigenValues=w[idx]
#
# eigenVectors=v[:,idx]

kValsLen=L
kIndAll = list(range(0, kValsLen))
kValsAll = [2 * np.pi * (l-kValsLen/2) / (2 * kValsLen) for l in kIndAll]

#all H(k) blocks, each with momentum k
All_H_k = [H_k(kTmp) for kTmp in kValsAll]# is a list of Ds by Ds matrices
All_eigen_value=[]# is a list of vectors, each vector contains Ds eigenvalues
All_eigen_vector=[]#is a list of matrices, each matrix contains Ds eigenvectors
for idx_k,k in enumerate(kValsAll):
    print("finish "+str(100*idx_k/L)+"%")

    eigenvaluesTmp, eigenvectorsTmp=np.linalg.eigh(All_H_k[idx_k])
    idxTmp=eigenvaluesTmp.argsort()[::-1]
    All_eigen_value.append(eigenvaluesTmp[idxTmp])
    All_eigen_vector.append(eigenvectorsTmp[:,idxTmp])


#gauge smooth
#loop over band
for idx_m in range(Ds):
    #loop over k
    for idx_s in range(L-1):
        phase_diff=np.imag(np.log(np.vdot(All_eigen_vector[idx_s][:,idx_m],All_eigen_vector[idx_s+1][:, idx_m])))
        #update the eigenvector the adjacent phase -> 0
        All_eigen_vector[idx_s+1][:,idx_m]*=np.exp(-1j*phase_diff)
    total_diff=np.imag(np.log(np.vdot(All_eigen_vector[0][:,idx_m],All_eigen_vector[-1][:,idx_m])))
    phase_avg=total_diff/(L-1)
    for idx_s in range(L-1):
        All_eigen_vector[idx_s][:,idx_m]*=np.exp(1j*phase_avg*idx_s)
        # print("idx_s is "+str(idx_s))




# n0=0
# uValsAll=eigenVectors[:,n0]

# psiInit = []

# for b in range(0,Ds):
#     for j in range(0,L):
#         sumTmp=0
#         for a in range(0,L):
#             sumTmp+=f(kValsAll[a])*np.exp(1j*kValsAll[a]*2*j)
#         sumTmp*=uValsAll[b]
#         psiInit.append(sumTmp)

# for b in range(0, Ds):
#     for j in range(0, L):
#         sumTmp = 0
#         for a in range(0, kValsLen):
#             sumTmp += zerothEigVecsAll[a][b] * f(kValsAll[a]) * np.exp(1j * kValsAll[a]  *2* j)
#         psiInit.append(sumTmp)


def l2Norm(vec):
    '''

    :param vec: input vector
    :return: l2 norm of vec
    '''
    tmp = 0
    for j in range(0, len(vec)):
        tmp += np.abs(vec[j]) ** 2
    return np.sqrt(tmp)

bandNum=0
w_state=np.zeros(N_H,dtype=complex)
for idx_k, k in enumerate(kValsAll):
    eigen_state=All_eigen_vector[idx_k][:,bandNum]
    w_state+=ortho_to_fock(k,eigen_state)*np.exp(-1j*2*k*j0)*g(k,0)

for x in range(N):
    coupling= [[1,x]]
    static_n = [['n',coupling]]
    exec(f'H_n_{x}_H = hamiltonian(static_n,[], dtype=np.complex128,basis=basisAll)')
    exec(f'H_n_{x} = H_n_{x}_H.toarray()')


density_total=np.zeros(N)
w_state/=np.sqrt(w_state.conj().T@w_state)
psi0Vec=w_state
for x in range(N):
    exec(f'density_total[x] = w_state.conj().T @ H_n_{x} @ w_state')

plt.figure()
plt.plot(np.arange(N), density_total)
plt.savefig("tmp2.png")

tEnd=datetime.now()
print(" init time: ",tEnd-tStart)