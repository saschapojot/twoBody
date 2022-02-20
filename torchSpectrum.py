import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import torch
#script for band and Chern number
#consts
alpha=1/3
T1=4
J=2.5
V=2.5
Omega=2*np.pi/T1

a=4
b=1
T2=T1*b/a
omegaF=2*np.pi/T2
T=T1*b#total small time
Q=100#small time interval number
dt=T/Q
U=10

tTicksAll=range(0,Q+1)
tValsAll=[dt*q for q in tTicksAll]



subLatNum=3#sublattice number
L=7##unit cell number, must be odd
N=subLatNum*L #total sublattice number
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
    return stateStr[-subLatNum:]+stateStr[:-subLatNum]


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
    # retVec/=np.sqrt(L)
    return retVec


momentumEigVecsAll=[]#first indexed by phi, then by the order in seedStatesAll
for phi in phiValsAll:
    vecListTmp=[]
    for seed in seedStatesAll:
        vecListTmp.append(seedToMomentumEigVec(phi,seed))
    momentumEigVecsAll.append(vecListTmp)


def HMat(qm):
    """

    :param q: time index
    :param m: beta index
    :return:
    """
    q,m=qm
    t=tValsAll[q]
    beta=betaValsAll[m]
    # a_{m}^{+}a_{m+1} coef
    hoppingPM = [[J / 2 * np.exp(-1j * omegaF * t), j, (j + 1) % N] for j in range(0, N)]
    # a_{m+1}^{+}a_{m} coef
    hoppingMP = [[J / 2 * np.exp(1j * omegaF * t), j, (j + 1) % N] for j in range(0, N)]
    # onsite driving
    onsiteDriving = [[V * np.cos(2 * np.pi * alpha * j - beta) * np.cos(Omega * t), j] for j in range(0, N)]
    # onsite 2
    onSite2 = [[U / 2, j, j] for j in range(0, N)]
    # onsite 1
    onSite1 = [[-U / 2, j] for j in range(0, N)]

    hListTmp = [["+-", hoppingPM], ["-+", hoppingMP], ["n", onsiteDriving], ["nn", onSite2], ["n", onSite1]]
    HTmp = hamiltonian(hListTmp, [], dtype=np.complex128, basis=basisAll)
    return [q,m,HTmp.toarray()]

qmAll=[[q,m] for q in range(1,Q+1) for m in range(0,M)]

tHMatStart=datetime.now()
pool0=Pool(threadNum)
ret0=pool0.map(HMat,qmAll)

tHMatEnd=datetime.now()

print("HMat time: ",tHMatEnd-tHMatStart)
print("Ns is "+str(basisAll.Ns))
print("Ds is "+str(Ds))
tInitStart=datetime.now()
tensorHMatAll=torch.zeros((Q,M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)
for itemTmp in ret0:
    q,m,matHTmp=itemTmp
    position=Q-q
    tensorHMatAll[position,m,:,:]=torch.from_numpy(-1j*dt*matHTmp)

tInitEnd=datetime.now()
print("initialization time: ",tInitEnd-tInitStart)



# tensorHMatAll=tensorHMatAll.cuda()
# torch.cuda.synchronize()
# tExpStart=datetime.now()
# uTensorAll=tensorHMatAll.matrix_exp()
# torch.cuda.synchronize()
# tExpEnd=datetime.now()
#
# print("exp time: ",tExpEnd-tExpStart)
tExpStart=datetime.now()
UqTensorMat=torch.zeros((Q,M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)
for q in range(0,Q):
    UqTensorMat[q,:,:,:]=tensorHMatAll[q,:,:,:].matrix_exp()
tExpEnd=datetime.now()

print("exp time: ",tExpEnd-tExpStart)


# print("pytorch version is "+torch.__version__)
prodUTensor=torch.zeros((M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)
#initialize with identity matrix
tProdStart=datetime.now()
for j in range(0,M):
    prodUTensor[j,:,:]=torch.eye(basisAll.Ns,dtype=torch.cfloat)

for q in range(0,Q):
    prodUTensor=torch.bmm(prodUTensor,UqTensorMat[q,:,:,:])

tProdEnd=datetime.now()
print("prod time: ",tProdEnd-tProdStart)


def reducedFloquetMat(betaNumphiNum):
    """

    :param betaNum: beta index
    :param phiNum: phi index
    :return:
    """
    betaNum,phiNum=betaNumphiNum
    retMat=np.zeros((Ds,Ds),dtype=complex)
    UMatTmp=prodUTensor[betaNum,:,:].numpy()
    for j in range(0,Ds):
        for l in range(0,Ds):
            #left vec has one elem 1, the rest are 0, therefore it selects a row
            leftVecInd=basisAll.index(seedStatesAll[j])
            rightVec=momentumEigVecsAll[phiNum][l]
            vecTmp=UMatTmp[leftVecInd,:]
            retMat[j,l]=np.dot(vecTmp,rightVec)

    return [betaNum,phiNum,retMat]



pool1=Pool(threadNum)

betaNumAndPhiNumAll=[[m,r] for m in range(0,M) for r in range(0,L)]

tReducedFlMatStart=datetime.now()
ret1=pool1.map(reducedFloquetMat,betaNumAndPhiNumAll)
tReducedFlMatEnd=datetime.now()
print("reduced Floquet mat time: ",tReducedFlMatEnd-tReducedFlMatStart)

#m=0,1,...,M-1, r=0,1,...,L-1
#index of matrix is mL+r
tInitRedFlStart=datetime.now()
reducedFlMatTensor=torch.zeros((M*L,Ds,Ds),dtype=torch.cfloat)
for itemTmp in ret1:
    m,r,rUMat=itemTmp
    reducedFlMatTensor[m*L+r,:,:]=torch.from_numpy(rUMat)
tInitRedFlEnd=datetime.now()
print("initialize reduced Floquet matrix tensor time: ",tInitRedFlEnd-tInitRedFlStart)

tEigStart=datetime.now()
eigTensor, vecTensor=torch.linalg.eig(reducedFlMatTensor)
tEigEnd=datetime.now()
print("Eig time: ",tEigEnd-tEigStart)

phasesAll=torch.angle(eigTensor)

indsAll=torch.argsort(phasesAll)


#data serialization

dataAll=[]
for j in range(0,len(phasesAll)):

    r=j%L
    m=int((j-r)/L)
    oneRow=[m,r]
    indsTmp=indsAll[j]
    phasesTmp=phasesAll[j]
    phasesTmpSorted=[phasesTmp[ind]/np.pi for ind in indsTmp]
    oneRow.extend(phasesTmpSorted)
    dataAll.append(oneRow)

dataAll=np.array(dataAll)

sortedByPhiDataAll=np.array(sorted(dataAll,key=lambda row:row[1]))#sort phiNum, such that the first M rows correspond to phi=0

pltByBetaDataAll=sortedByPhiDataAll[:M,:]
#plt by beta, section phi=0
#data serialization
pltByBetaBeta=[]
pltByBetaPhase=[]
for oneRow in pltByBetaDataAll:
    m=oneRow[0]
    for onePhase in oneRow[2:]:
        pltByBetaBeta.append(2*m/M)
        pltByBetaPhase.append(onePhase)

phiValStr=", $\phi=$"+str(0)
plt.figure()
plt.scatter(pltByBetaBeta,pltByBetaPhase,color="blue",s=1)
plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
         + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +phiValStr
          )
plt.xlabel("$\\beta/\pi$")
plt.ylabel("eigenphase$/\pi$")
plt.savefig("torchT1="+str(T1)
            # +"omegaF=0"
              +"a"+str(a)+"b"+str(b)
    +"phi=0.png")
plt.close()
#plot by phi, section beta=0
sortedByBetaAll=np.array(sorted(dataAll,key=lambda row: row[0]))
pltByPhiDataAll=sortedByBetaAll[:L,:]
#data serialization
pltByPhiPhi=[]
pltByPhiPhase=[]
for oneRow in pltByPhiDataAll:
    r=oneRow[1]
    for onePhase in oneRow[2:]:
        pltByPhiPhi.append(2*r/L)
        pltByPhiPhase.append(onePhase)

betaValStr=", $\\beta=$"+str(0)

plt.figure()
plt.scatter(pltByPhiPhi,pltByPhiPhase,color="blue",s=1)
plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
         + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +betaValStr
          )
plt.xlabel("$\phi/\pi$")
plt.ylabel("eigenphase$/\pi$")
plt.savefig("torchT1="+str(T1)
            # +"omegaF=0"
              +"a"+str(a)+"b"+str(b)
    +"beta=0.png")
plt.close()

phaseTable=dataAll[:,2:]
#col of distToBandBelow is dist of a band to the band below
distToBandBelow=np.zeros(phaseTable.shape,dtype=float)
for n in range(0,Ds):
    distTmp=np.abs(phaseTable[:,n]-phaseTable[:,(n-1)%Ds])
    distToBandBelow[:,n]=distTmp[:]#deep copy

#mod 2
for n in range(0,Ds):
    distToBandBelow[:,n]=distToBandBelow[:,n]%2

#col of distToBandAbove is dist of a band to the band above
distToBandAbove=np.zeros(phaseTable.shape,dtype=float)
for n in range(0,Ds):
    distToBandAbove[:,n]=distToBandBelow[:,(n+1)%Ds][:]#deep copy

#staticstics of dist
minDistList=[]
avgDistList=[]
for n in range(0,Ds):
    tmp1=min(distToBandBelow[:,n])
    tmp2=min(distToBandAbove[:,n])
    minDistList.append(min(tmp1,tmp2))

for n in range(0,Ds):
    vec1=distToBandBelow[:,n][:]#deep copy
    vec2=distToBandAbove[:,n][:]#deep copy
    vec=np.append(vec1,vec2)
    avgDistList.append(np.mean(vec))

#sort by descending order of minDistList

inds=np.argsort(minDistList)[::-1]

sortedMinDist=[minDistList[ind] for ind in inds]
sortedAvgDist=[avgDistList[ind] for ind in inds]
dataOut=np.array([inds,sortedMinDist,sortedAvgDist]).T

dtFrm=pd.DataFrame(data=dataOut,columns=["ind","minDist/pi","avgDist/pi"])

dtFrm.to_csv("torchDistT1"+str(T1)
             # +"omegaF=0"
             +"a"+str(a)+"b"+str(b)
             +"U="+str(U)
             +".csv", index=False
             )
