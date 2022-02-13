import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import scipy.sparse.linalg as ssplin
from multiprocessing import Pool
#script for band and Chern number
#consts
alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1


a=1
b=1
T2=T1*b/a
omegaF=0#2*np.pi/T2
T=T1*b#total small time
Q=100#small time interval number
dt=T/Q
U=30
tValsAll=[dt*q for q in range(1,Q+1)]

q=3#sublattice number
L=11##unit cell number, must be odd
N=q*L #total sublattice number
M=50#beta num
betaValsAll=[2*np.pi*m/M for m in range(0,M)]#adiabatic parameter
phiValsAll=[2*np.pi*l/L for l in range(0,L)]#bloch momentum
#basis
basisAll=boson_basis_1d(N,Nb=2)
basisAllInString=[basisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in basisAll]
Ds=int(basisAll.Ns/L)#momentum space dimension=seed states number

threadNum=24
def coTranslation(stateStr):
    """

    :param stateStr: state repr in string
    :return: translate 1 lattice=2 sublattices to the right
    """
    return stateStr[-q:]+stateStr[:-q]

tStart=datetime.now()
seedStatesAll=[]
redundantStatesAll=[]
for stateStrTmp in basisAllInString:
    if stateStrTmp in redundantStatesAll:
        continue
    else:
        seedStatesAll.append(stateStrTmp)
        nextStr=stateStrTmp[:]
        for j in range(1,L):
            nextStr=coTranslation(nextStr)
            redundantStatesAll.append(nextStr)


def strToVec(stateStr):
    """

    :param stateStr: basis repr in string
    :return: vector repr for stateStr
    """
    i0=basisAll.index(stateStr)
    psi=np.zeros(basisAll.Ns,dtype=complex)
    psi[i0]=1
    return psi

seedVecsAll=[strToVec(seed) for seed in seedStatesAll]

def seedToMomentumEigVec(phi,seedStr):
    """

    :param phi: momentum value
    :param seedStr: seed state in string
    :return: momentum eigvec |phi,n>
    """
    retVec=np.zeros(basisAll.Ns,dtype=complex)
    nextStr=seedStr[:]
    for j in range(0,L):
        retVec+=np.exp(1j*phi*j)*strToVec(nextStr)
        nextStr=coTranslation(nextStr)
    retVec/=np.sqrt(L)
    return retVec


momentumEigVecsAll=[]#first indexed by phi, then by the order in seedStatesAll
for phi in phiValsAll:
    vecListTmp=[]
    for seed in seedStatesAll:
        vecListTmp.append(seedToMomentumEigVec(phi,seed))
    momentumEigVecsAll.append(vecListTmp)

def HCSCMat(t,beta):
    """

    :param t:
    :param beta:
    :return: sparse Hamiltonian matrix in csc format
    """

    #a_{m}^{+}a_{m+1} coef
    hoppingPM=[[J/2*np.exp(-1j*omegaF*t),j,(j+1)%N] for j in range(0,N)]
    #a_{m+1}^{+}a_{m} coef
    hoppingMP=[[J/2*np.exp(1j*omegaF*t),j,(j+1)%N] for j in range(0,N)]
    #onsite driving
    onsiteDriving=[[V*np.cos(2*np.pi*alpha*j-beta)*np.cos(Omega*t),j] for j in range(0,N)]
    #onsite 2
    onSite2=[[U/2,j,j] for j in range(0,N)]
    #onsite 1
    onSite1=[[-U/2,j] for j in range(0,N)]

    hListTmp=[["+-",hoppingPM],["-+",hoppingMP],["n",onsiteDriving],["nn",onSite2],["n",onSite1]]
    HTmp=hamiltonian(hListTmp,[],dtype=np.complex128,basis=basisAll)
    return HTmp.tocsc()


def UMat(beta):
    """

    :param beta:  adiabatic parameter
    :return: full Floquet operator at beta
    """
    retU=np.eye(basisAll.Ns,dtype=complex)
    for tq in tValsAll[::-1]:
        retU=retU@ssplin.expm(-1j*dt*HCSCMat(tq,beta))

    return [beta,retU]


pool0=Pool(threadNum)
retAll0=pool0.map(UMat,betaValsAll)
ret0Sorted=sorted(retAll0,key=lambda elem:elem[0])
# print(ret0Sorted)

# UMatsAll=[UMat(betaTmp) for betaTmp in betaValsAll]
UMatsAll=[elem[1] for elem in ret0Sorted]
def reducedFloquetMat(betaNum,phiNum):
    """

    :param betaNum: betaNum th beta
    :param phiNum: phiNum th phi
    :return: reduced Floquet operator
    """
    retMat=np.zeros((Ds,Ds),dtype=complex)
    UMatTmp=UMatsAll[betaNum]
    for j in range(0,Ds):
        for l in range(0,Ds):
            leftVec=seedVecsAll[j]
            rightVec=momentumEigVecsAll[phiNum][l]
            retMat[j,l]=leftVec.conj().T@UMatTmp@rightVec
    return retMat




def sortedEigPhaseAndVec(betaNumAndphiNumPair):
    """


    :return: sorted eigenphases and eigenvectors
    """
    betaNum=betaNumAndphiNumPair[0]
    phiNum=betaNumAndphiNumPair[1]
    reducedUMat=reducedFloquetMat(betaNum,phiNum)
    valsTmp,vecsTmp=np.linalg.eig(reducedUMat)
    phasesTmp=np.angle(valsTmp)
    inds=np.argsort(phasesTmp)
    sortedPhases=[phasesTmp[ind] for ind in inds]
    sortedVecs=[vecsTmp[:,ind] for ind in inds]
    return [betaNum,phiNum,sortedPhases,sortedVecs]

inDatsAll=[[betaNum,phiNum] for betaNum in range(0,M) for phiNum in range(0,L)]

pool1=Pool(threadNum)

retAll=pool1.map(sortedEigPhaseAndVec,inDatsAll)

tEnd=datetime.now()
print("computation time: ",tEnd-tStart)
#data serialization
pltBeta=[]
pltPhi=[]
pltPhase=[]
tableMat=[]
for itemTmp in retAll:
    betaNum,phiNum,phases,vecs=itemTmp
    oneRow=[betaNum,phiNum]
    oneRow.extend(phases)
    tableMat.append(oneRow)
    betaTmp=betaValsAll[betaNum]
    phiTmp=phiValsAll[phiNum]
    for phaseTmp in phases:
        pltBeta.append(betaTmp/np.pi)
        pltPhi.append(phiTmp/np.pi)
        pltPhase.append(phaseTmp/np.pi)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111,projection='3d')
sct0=ax.scatter(pltBeta, pltPhi, pltPhase,marker="." ,c="blue")
ax.set_xlabel("$\\beta/\pi$")
ax.set_ylabel("$\phi/\pi$")
ax.set_zlabel("eigenphase$/\pi$")
plt.title("$T_{1}=$"+str(T1)
          #+", $\omega_{F}=0$"
          + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          )



plt.savefig("spectrumT1"+str(T1)
            +"omegaF=0"
           # +"a"+str(a)+"b"+str(b)
            +".png")
# plt.show()
plt.close()

tableArr=np.array(tableMat)
np.savetxt("dataSpectrumT1"+str(T1)
            +"omegaF=0"
           #+"a"+str(a)+"b"+str(b)
           +".csv",tableArr,delimiter=",",newline="\n")