import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import scipy.sparse.linalg as ssplin
from multiprocessing import Pool
import torch

#optimized script for pbc band
#consts
alpha=1/3
T1=4
J=2.5
V=2.5
Omega=2*np.pi/T1


a=3
b=1
T2=T1*b/a
omegaF=2*np.pi/T2
T=T1*b#total small time
Q=100#small time interval number
dt=T/Q
U=10
tTicksRange=range(0,Q+1)
tValsAll=[dt*q for q in range(0,Q+1)]#t0=0, t1,...,tQ

# q=3#sublattice number
subLatNum=3
L=11##unit cell number, must be odd
N=subLatNum*L #total sublattice number
M=50#beta num
betaValsAll=[2*np.pi*m/M for m in range(0,M)]#adiabatic parameter
phiValsAll=[2*np.pi*l/L for l in range(0,L)]#bloch momentum
#basis
basisAll=boson_basis_1d(N,Nb=2)
basisAllInString=[basisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in basisAll]
Ds=int(basisAll.Ns/L)#momentum space dimension=seed states number

threadNum=24

def generateAq(q):
    """

    :param q:
    :return: [q, Aq]
    """
    tq=tValsAll[q]
    #a_{m}^{+}a_{m+1}
    hoppingPM=[[J/2*np.exp(-1j*omegaF*tq),m,(m+1)%N] for m in range(0,N)]
    #a_{m}a_{m+1}^{+}
    hoppingMP=[[J/2*np.exp(1j*omegaF*tq),m,(m+1)%N] for m in range(0,N)]
    hList=[["+-",hoppingPM],["-+",hoppingMP]]
    h1Tmp=hamiltonian(hList,[],dtype=np.complex128,basis=basisAll)
    return [q, ssplin.expm(-1j*dt*h1Tmp.tocsc())]

tGenAqStart=datetime.now()
pool0=Pool(threadNum)
ret0=pool0.map(generateAq,tTicksRange)
tGenAqEnd=datetime.now()
print("generate Aq matrices time: ",tGenAqEnd-tGenAqStart)
sortedRet0=sorted(ret0,key=lambda elem:elem[0])#[[q, Aq]],q=0,1,...,Q

def strToIndex(str):
    """

    :param str: basis state in string repr
    :return: index of str
    """

    return basisAll.index(str)

def coTranslation(stateStr):
    """

    :param stateStr: state repr in string
    :return: translate 1 lattice=2 sublattices to the right
    """
    return stateStr[-subLatNum:]+stateStr[:-subLatNum]

def coTranslateRepeat(stateStr,a):
    """

    :param stateStr: input basis state
    :param a: number of times of cotranslation operation on stateStr
    :return: stateStr costranslated a times
    """
    nextStr=stateStr[:]#deep copy
    while a>0:
        nextStr=coTranslation(nextStr)
        a-=1
    return nextStr

seedStatesAll=[]# in string repr
redundantStatesAll=[]#in string repr
for stateStrTmp in basisAllInString:
    if stateStrTmp in redundantStatesAll:
        continue
    else:
        seedStatesAll.append(stateStrTmp)
        nextStr=stateStrTmp[:]
        for j in range(1,L):
            nextStr=coTranslation(nextStr)
            redundantStatesAll.append(nextStr)



def AqElem(q,j,a,l):
    """
    :param q:
    :param j:
    :param a:
    :param l:
    :return: return matrix element of Aq, according to <nj|Aq \tau^{a}|nl>
    """
    seedStrLeft=seedStatesAll[j]
    seedStrRight=seedStatesAll[l]
    seedRightTranslated=coTranslateRepeat(seedStrRight,a)

    indLeft=strToIndex(seedStrLeft)
    indRight=strToIndex(seedRightTranslated)
    return sortedRet0[q][1][indLeft,indRight]



def findIndexAndNumber(stateStr):
    """

    :param stateStr: basis state in string repr
    :return: if the string contains one 2, return the index of 2 and 2
            if the string contains two 1', return each index and 1
    """
    retList2=[]
    retList11=[]

    for ind,char in enumerate(stateStr):
        if char=="2":
            retList2.append([ind,int(char)])
            return retList2#once finds 2, return, because there is only one 2
        elif char=="1":
            retList11.append([ind,int(char)])

    return retList11


def Z(m,q,a,l):
    """

    :param m: beta index
    :param q: time step number
    :param a: cotranslation number
    :param l: index of a seed state
    :return:
    """
    tq=tValsAll[q]
    betam=betaValsAll[m]
    stateStr=coTranslateRepeat(seedStatesAll[l],a)#\tau^{a}|nl>
    siteAndNumList=findIndexAndNumber(stateStr)
    retZ=1
    for itemTmp in siteAndNumList:
        site,num=itemTmp
        retZ*=np.exp(-1j*dt*V*np.cos(2*np.pi*alpha*site-betam)*np.cos(Omega*tq)*num-1j*dt*U/2*num*(num-1))
    return retZ




def UTildeElem(q,j,l,m,r):
    """

    :param q: time index
    :param j: seed state index
    :param l: seed state index
    :param m: beta index
    :param r: phi index
    :return:
    """
    phir=phiValsAll[r]
    rst=0
    for a in range(0,L):
        rst+=AqElem(q,j,a,l)*Z(m,q,a,l)*np.exp(1j*a*phir)
    return rst


def UTildeMat(q,m,r):
    """

    :param q: time index
    :param m: beta index
    :param r: phi index
    :return: Utilde matrix (reduced Floquet matrix)
    """
    retMat=np.zeros((Ds,Ds),dtype=complex)
    for j in range(0,Ds):
        for l in range(0,Ds):
            retMat[j,l]=UTildeElem(q,j,l,m,r)

    return retMat


def UTildeMatWrapper(qmr):
    """

    :param qmr:
    :return:
    """
    q,m,r=qmr
    return [m,r,UTildeMat(q,m,r)]


def generateOneTensor(q):
    """

    :param q: time step
    :return: all reduced Floquet matrices at time step q
    """

    indsAll=[[q,m,r] for m in range(0,M) for r in range(0,L)]
    pool1=Pool(threadNum)
    ret1=pool1.map(UTildeMatWrapper,indsAll)

    retTensor=torch.zeros((M*L,Ds,Ds),dtype=torch.cfloat)
    for itemTmp in ret1:
        m,r,uTildeMat=itemTmp
        retTensor[m*L+r,:,:]=torch.from_numpy(uTildeMat)

    return retTensor



def generateUTilde():
    """

    :return: product of reduced Floquet matrices at each m and r
    """
    retUTildeTensor=torch.zeros((M*L,Ds,Ds),dtype=torch.cfloat)
    for j in range(0,M*L):
        retUTildeTensor[j,:,:]=torch.eye(Ds,dtype=torch.cfloat)
    #matrices at time step Q,Q-1,...,1
    tInitStart=datetime.now()
    tensorAtAllTimes=torch.zeros((Q,M*L,Ds,Ds),dtype=torch.cfloat)
    for q in range(0,Q):
        timeTick=Q-q
        tensorAtAllTimes[q,:,:,:]=generateOneTensor(timeTick)
    tInitEnd=datetime.now()
    print("initialization time: ",tInitEnd-tInitStart)
    tCalStart=datetime.now()
    # torch.cuda.synchronize()
    # retUTildeTensor=retUTildeTensor.cuda()
    # tensorAtAllTimes=tensorAtAllTimes.cuda()
    for q in range(0,Q):
        retUTildeTensor=torch.bmm(retUTildeTensor,tensorAtAllTimes[q,:,:,:])
    # torch.cuda.synchronize()
    tCalcEnd=datetime.now()
    print("calc prod time: ",tCalcEnd-tCalStart)
    return retUTildeTensor




valsTensor,vecstensor=torch.linalg.eig(generateUTilde())

#data serialization
pltBeta=[]
pltPhi=[]
pltPhase=[]
# print(torch.abs(valsTensor))
tableMat=[]
for j in range(0,M*L):
    r=j%L
    m=int((j-r)/L)
    oneRow=[m,r]
    for elem in valsTensor[j]:
        pltBeta.append(2*m/M)
        pltPhi.append(2*r/L)
        pltPhase.append(np.angle(elem.item())/np.pi)
        oneRow.append(elem.item())
    tableMat.append(oneRow)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111,projection='3d')
sct0=ax.scatter(pltBeta, pltPhi, pltPhase,marker="." ,c="blue")
ax.set_xlabel("$\\beta/\pi$")
ax.set_ylabel("$\phi/\pi$")
ax.set_zlabel("eigenphase$/\pi$")
plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
          + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +", $U=$"+str(U)
          )
plt.savefig("GPUspectrumT1"+str(T1)
            # +"omegaF=0"
           +"a"+str(a)+"b"+str(b)
            +"U="+str(U)+".png"
            )
# plt.show()
plt.close()

tableArr=np.array(tableMat)
np.savetxt("GPUdataSpectrumT1"+str(T1)
            # +"omegaF=0"
           +"a"+str(a)+"b"+str(b)
           +"U="+str(U)+".csv",tableArr,delimiter=",",newline="\n")



