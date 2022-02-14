import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import scipy.sparse.linalg as ssplin
from multiprocessing import Pool

#script for pumping
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
U=10
tValsAll=[dt*q for q in range(1,Q+1)]

q=3#sublattice number
L=51##unit cell number, must be odd
N=q*L #total sublattice number
M=500#beta num
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


def seedToMomentumEigVec(kVal,seedStr):
    """

    :param kVal: momentum value
    :param seedStr: seed state in string
    :return: momentum eigvec |k,n>
    """
    retVec=np.zeros(basisAll.Ns,dtype=complex)
    nextStr=seedStr[:]
    for j in range(0,L):
        retVec+=np.exp(1j*kVal*j)*strToVec(nextStr)
        nextStr=coTranslation(nextStr)
    retVec/=np.sqrt(L)
    return retVec


###construct all seed states in vector form
seedVecsAll=[strToVec(seed) for seed in seedStatesAll]


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
    :return: sparse rotated Hamiltonian matrix in csc format
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

    return retU

beta0=betaValsAll[0]
UMatsAll=[UMat(beta0)]
def reducedFloquetMat(betaNum,phiNum):
    """

    :param betaNum: betaNum th beta, =0
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



#all [beta0Num, phiNum] pairs
inDatsAll=[[0,phiNum]  for phiNum in range(0,L)]



pool0=Pool(threadNum)

retAll0=pool0.map(sortedEigPhaseAndVec,inDatsAll)


#bandNum=0,1,...,Ds-1
# print("Ds="+str(Ds))

bandNum=0
R=int(L/2)
ws=np.zeros(basisAll.Ns,dtype=complex)
for a in range(0,L):
    phiaNum=retAll0[a][1]
    phia=phiValsAll[phiaNum]
    psiVeca=retAll0[a][3][bandNum]
    for b in range(0,Ds):
        seedTmp=seedStatesAll[b][:]
        for j in range(0,L):
            vecTmp=strToVec(seedTmp)
            ws+=vecTmp*psiVeca[b]*np.exp(1j*(j-R)*phia)
            seedTmp=coTranslation(seedTmp)

ws/=np.linalg.norm(ws,ord=2)
dataAll=[ws]
subLatOpList=[]
for j in range(0,N):
    listTmp = [[1, j]]
    staticTmp = [["n", listTmp]]
    opTmp = hamiltonian(staticTmp, [], dtype=np.complex128, basis=basisAll)
    arrTmp = opTmp.tocsc()
    subLatOpList.append(arrTmp)


def onSiteMagnitude(vec):
    """

    :param vec: full vec of length N
    :return:
    """
    mag=[]
    for arrTmp in subLatOpList:
        xTmp=arrTmp.dot(vec)
        yTmp=vec.conj().T.dot(xTmp)
        mag.append(np.real(yTmp))
    return mag

posVals=[[j,j]for j in range(0,N)]
posList=[["n",posVals]]
xOpr=hamiltonian(posList,[],basis=basisAll,dtype=np.complex128)
xMat=xOpr.tocsc()/3

def avePos(vec):
    """

    :param vec:
    :return: avg position in lattice number
    """
    xTmp=xMat.dot(vec)
    return np.real(vec.conj().T.dot(xTmp))
# mag=onSiteMagnitude(ws)
# plt.figure()
# plt.plot(range(0,N),mag)
# plt.savefig("tmp.png")

#construct real space Hamiltonian for evolution
#static part

onSite2=[[U/2,m,m] for m in range(0,N)]
onSite1=[[-U/2,m] for m in range(0,N)]
hoppingCoef=[[J/2,m,(m+1)%N] for m in range(0,N)]#same for +- and -+
onSiteLin=[[omegaF*m,m] for m in range(0,N)]
staticPart=[["+-",hoppingCoef],["-+",hoppingCoef],["nn",onSite2],["n",onSite1],["n",onSiteLin]]
#dynamical functions
def driving(t):
    return np.cos(Omega*t)

#before evolution, plot init wavepcaket
magInit=onSiteMagnitude(ws)
plt.figure()
plt.plot(range(0,N),magInit,color="black")
plt.savefig("init.png")
plt.close()

tEvStart=datetime.now()
for beta in betaValsAll:
    drivingCoefs=[[V*np.cos(2*np.pi*alpha*m-beta),m] for m in range(0,N)]
    dynPart=[["n",drivingCoefs,driving,[]]]
    HTmp=hamiltonian(staticPart,dynPart,static_fmt="csr",dtype=np.complex128,basis=basisAll)
    psiCurr=dataAll[-1]
    tStart=0
    tEndList=[T]
    psiNext=HTmp.evolve(psiCurr,tStart,tEndList,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False)[:,0]
    dataAll.append(psiNext)


tEvEnd=datetime.now()
print("evolution time :", tEvEnd-tEvStart)

#after evolution plot final wavepacket
psiLast=dataAll[-1]
magLast=onSiteMagnitude(psiLast)
plt.figure()
plt.plot(range(0,N),magLast,color="black")
plt.savefig("last.png")
plt.close()

#calculate drift
posAll=[]
for vec in dataAll:
    posAll.append(avePos(vec))

drift=[elem-posAll[0] for elem in posAll]
dis=round(drift[-1],3)
#plot drift
plt.figure()
plt.plot(range(0,M+1),drift,color="black")
plt.title("$T_{1}=$"+str(T1)
          #+", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +"$\omega_{F}=0$"
          +", pumping = "+str(dis)+", band"+str(bandNum))
plt.xlabel("$t/T$")
plt.savefig("T1"+str(T1)
            # +"a"+str(a)+"b"+str(b)
            +"omegaF=0"
            +"band"+str(bandNum)+"displacement.png")
plt.close()