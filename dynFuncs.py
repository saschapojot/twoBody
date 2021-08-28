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