import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import scipy.sparse.linalg as ssplin
from multiprocessing import Pool


#script for obc band
#consts
alpha=1/3
T1=2
J=2.5
V=2.5
Omega=2*np.pi/T1


a=5
b=2
T2=T1*b/a
omegaF=2*np.pi/T2
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
weight=0.6
localLength=int(N*0.15)
threadNum=24



def HCSCMat(t,beta):
    """

    :param t:
    :param beta:
    :return: sparse Hamiltonian matrix in csc format under obc
    """

    #a_{m}^{+}a_{m+1} coef
    hoppingPM=[[J/2*np.exp(-1j*omegaF*t),j,j+1] for j in range(0,N-1)]
    #a_{m+1}^{+}a_{m} coef
    hoppingMP=[[J/2*np.exp(1j*omegaF*t),j,j+1] for j in range(0,N-1)]
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

    return retU



def sortedEigPhaseAndEigVec(beta):
    """

    :param beta:
    :return: sorted eigenphases and eigenvectors
    """
    UMatTmp=UMat(beta)
    eigVals, eigVecs=np.linalg.eig(UMatTmp)
    eigPhases=np.angle(eigVals)
    inds=np.argsort(eigPhases)
    sortedPhases=[eigPhases[ind] for ind in inds]
    sortedVecs=[eigVecs[:,ind] for ind in inds]
    return sortedPhases,sortedVecs

leftSites=range(0,localLength)
righSites=range(0,N)[-localLength:]

leftSiteOpList=[]
rightSiteOpList=[]
for j in leftSites:
    listTmp=[[1,j]]
    staticTmp=[["n",listTmp]]
    opTmp=hamiltonian(staticTmp,[],dtype=np.complex128,basis=basisAll)
    arrTmp=opTmp.tocsc()
    leftSiteOpList.append(arrTmp)

for j in righSites:
    listTmp=[[1,j]]
    staticTmp=[["n",listTmp]]
    opTmp=hamiltonian(staticTmp,[],dtype=np.complex128,basis=basisAll)
    arrTmp=opTmp.tocsc()
    rightSiteOpList.append(arrTmp)

def computeLeftWeight(vec):
    tmp=0
    for arrTmp in leftSiteOpList:
        dotTmp=arrTmp.dot(vec)
        tmp+=vec.conj().T.dot(dotTmp)
    return np.real(tmp)


def computeRightWeight(vec):
    tmp=0
    for arrTmp in rightSiteOpList:
        dotTmp=arrTmp.dot(vec)
        tmp+=vec.conj().T.dot(dotTmp)
    return np.real(tmp)


def selectEdgeStates(vec):
    """

    :param vec:
    :return: left edge 0
             right egde 1
             else 2
    """
    leftWeight=computeLeftWeight(vec)
    rightWeight=computeRightWeight(vec)

    if leftWeight>=weight:
        return 0
    elif rightWeight>=weight:
        return 1
    else:
        return 2


def partitionPhaseAndVec(beta):
    """

    :param beta:
    :return: [beta, [[phaseLeft, vecLeft]], [[phaseRight, vecRight]],[[phaseMiddle, vecMiddle]]]
    """
    retListLeft=[]
    retListRight=[]
    retListMid=[]
    phasesAll, vecsAll=sortedEigPhaseAndEigVec(beta)
    for i, vecTmp in enumerate(vecsAll):
        kindTmp=selectEdgeStates(vecTmp)
        if kindTmp==0:
            retListLeft.append([phasesAll[i],vecTmp])
        elif kindTmp==1:
            retListRight.append([phasesAll[i],vecTmp])
        else:
            retListMid.append([phasesAll[i],vecTmp])

    return [beta,retListLeft, retListRight,retListMid]



pool1=Pool(threadNum)
tStart=datetime.now()
retAll=pool1.map(partitionPhaseAndVec,betaValsAll)
tEnd=datetime.now()
print("computation time: ",tEnd-tStart)


#data serialization
pltBetaLeft=[]
pltPhaseLeft=[]

pltBetaRight=[]
pltPhaseRight=[]

pltBetaMid=[]
pltPhaseMid=[]


for itemTmp in retAll:
    beta,retListLeft, retListRight,retListMid=itemTmp
    if len (retListLeft)>0:
        for leftPairTmp in retListLeft:
            pltBetaLeft.append(beta/np.pi)
            pltPhaseLeft.append(leftPairTmp[0]/np.pi)
    if len(retListRight)>0:
        for rightPairTmp in retListRight:
            pltBetaRight.append(beta/np.pi)
            pltPhaseRight.append(rightPairTmp[0]/np.pi)
    if len(retListMid)>0:
        for midPairTmp in retListMid:
            pltBetaMid.append(beta/np.pi)
            pltPhaseMid.append(midPairTmp[0]/np.pi)


sVal=2
plt.figure()
plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
         + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          )


plt.scatter(pltBetaLeft,pltPhaseLeft,color="magenta",marker=".",s=sVal,label="left")
plt.scatter(pltBetaRight,pltPhaseRight,color="cyan",marker=".",s=sVal,label="right")
plt.scatter(pltBetaMid,pltPhaseMid,color="black",marker=".",s=sVal,label="bulk")
plt.xlabel("$\\beta/\pi$")
plt.ylabel("eigenphase/\pi")
plt.legend()
plt.savefig("obcT1"+str(T1)
            # +"omegaF=0"
            +"a"+str(a)+"b"+str(b)+".png"
             )