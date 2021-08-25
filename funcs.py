from consts import *


def delta(t):
    return d0 * np.sin(omega * t + phi0)


def Delta(t):
    return D0 * np.cos(omega * t + phi0)


def H0Mat(dTmp, DTmp):
    '''

    :param dTmp:
    :param DTmp:
    :return: time dependent part of the Hamiltonian
    '''
    hopPM = [[-J + (-1) ** j * dTmp, j, (j + 1) % N] for j in range(0, N)]  # PBC
    hopMP = [[-J + (-1) ** j * dTmp, j, (j + 1) % N] for j in range(0, N)]  # PBC
    stgPot = [[(-1) ** j * DTmp, j] for j in range(0, N)]

    staticPart = [["+-", hopPM], ["-+", hopMP], ["n", stgPot]]
    H0Tmp = hamiltonian(staticPart, [], basis=basisAll, dtype=np.complex128, check_symm=False)
    H0MatTmp = H0Tmp.toarray()
    return H0MatTmp


def S2(q, psiq):
    '''

    :param q:
    :param psiq:
    :return: psiq+1
    '''

    dTmp = delta((q + 1 / 2) * dt)
    DTmp = Delta((q + 1 / 2) * dt)

    U0 = slin.expm(-1j * dt * H0Mat(dTmp, DTmp))
    psiNext = U1.dot(U0).dot(U1).dot(psiq)
    return psiNext


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


def avgPos(psiq):
    '''

    :param psiq: wv func
    :return: average position
    '''
    return psiq.T.conj().dot(xMat).dot(psiq)

