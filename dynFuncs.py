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