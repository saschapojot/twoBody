import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import scipy.sparse.linalg as ssplin
from multiprocessing import Pool
import torch
import gc
import pandas as pd
#script for obc band using pytorch
#consts
alpha=1/3

J=2.5
V=2.5





q=3#sublattice number
L=21##unit cell number, must be odd
N=q*L #total sublattice number
M=50#beta num
betaValsAll=[2*np.pi*m/M for m in range(0,M)]#adiabatic parameter


#basis
basisAll=boson_basis_1d(N,Nb=2)
basisAllInString=[basisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in basisAll]
Ds=int(basisAll.Ns/L)#momentum space dimension=seed states number
weight=0.6
localLength=int(N*0.15)
threadNum=24
stepLength=0.05

#allocate memory for tensor that will be used repeatedly
prodUTensor=torch.zeros((M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)

def calcConsts(a,b,T1):
    """

    :param a:
    :param b:
    :param T1:
    :return:Q,dt, T, T2 , omegaF
    """

    T2=T1*b/a
    T=T1*b
    Q=int(T/stepLength)
    if Q>200:
        Q=200
    dt=T/Q
    print("Q="+str(Q))
    print("dt="+str(dt))
    return [Q,dt,T,T2]

def HMat(dataList):
    """

    :param dataList: [q,m,dt, T1, T2, U]
    :return:
    """
    q, m, dt, T1, T2 ,U=dataList
    Omega=2*np.pi/T1
    omegaF=2*np.pi/T2
    t=q*dt
    beta=betaValsAll[m]
    # a_{m}^{+}a_{m+1} coef
    hoppingPM = [[J / 2 * np.exp(-1j * omegaF * t), j, (j + 1) ] for j in range(0, N-1)]
    # a_{m+1}^{+}a_{m} coef
    hoppingMP = [[J / 2 * np.exp(1j * omegaF * t), j, (j + 1) ] for j in range(0, N-1)]
    # onsite driving
    onsiteDriving = [[V * np.cos(2 * np.pi * alpha * j - beta) * np.cos(Omega * t), j] for j in range(0, N)]
    # onsite 2
    onSite2 = [[U / 2, j, j] for j in range(0, N)]
    # onsite 1
    onSite1 = [[-U / 2, j] for j in range(0, N)]

    hListTmp = [["+-", hoppingPM], ["-+", hoppingMP], ["n", onsiteDriving], ["nn", onSite2], ["n", onSite1]]
    HTmp = hamiltonian(hListTmp, [], dtype=np.complex128, basis=basisAll,check_herm=False,check_symm=False,check_pcon=False)
    return [q, m, HTmp.toarray()]

def genOneHMat(qmdtT1T2U):
    """

    :param qmabT1U: q, m, dt, T1,T2,U
    :return: HMat
    """
    q,m,dt,T1,T2,U=qmdtT1T2U
    inDataList=[q,m,dt, T1, T2, U]
    return HMat(inDataList)

def genHTensor(T1,T2,T,Q,dt,U):
    """

    :param a:
    :param b:
    :param T1:
    :param U:
    :return: a tensor for HMat for all q for all m
    """
    # Q, dt, T, T2=calcConsts(a,b,T1)
    inDataAll=[[q, m, dt, T1,T2,U] for q in range(1,Q+1) for  m in range(0,M)]
    pool0=Pool(threadNum)
    tHMatStart=datetime.now()
    ret0=pool0.map(genOneHMat,inDataAll)
    pool0.close()
    pool0.join()
    tHMatEnd=datetime.now()
    print("HMat time: ",tHMatEnd-tHMatStart)
    tInitStart=datetime.now()
    tensorHMatAll=torch.zeros((Q,M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)
    for itemTmp in ret0:
        q,m,matTmp=itemTmp
        position=Q-q
        tensorHMatAll[position,m,:,:]=torch.from_numpy(-1j*dt*matTmp)

    tInitEnd=datetime.now()
    print("initialization time: ",tInitEnd-tInitStart)
    del ret0
    gc.collect()
    return tensorHMatAll




def calcEig(Q,tensorHMatAll):
    """

    :param Q: time step num
    :param tensorHMatAll:
    :return: all eigenvals and eigenvecs
    """
    UqTensorMat = torch.zeros((Q, M, basisAll.Ns, basisAll.Ns), dtype=torch.cfloat)
    tExpStart = datetime.now()
    for q in range(0, Q):
        UqTensorMat[q, :, :, :] = tensorHMatAll[q, :, :, :].matrix_exp()
    tExpEnd = datetime.now()
    print("exp time: ", tExpEnd - tExpStart)
    #########matrix products
    global prodUTensor
    tProdStart = datetime.now()

    # set prodUTensor to id mat
    for j in range(0, M):
        prodUTensor[j, :, :] = torch.eye(basisAll.Ns, basisAll.Ns, dtype=torch.cfloat)

    for q in range(0,Q):
        prodUTensor=torch.bmm(prodUTensor,UqTensorMat[q,:,:,:])
    tProdEnd = datetime.now()
    print("prod time: ", tProdEnd - tProdStart)
    tEigStart=datetime.now()
    eigTensor, vecTensor=torch.linalg.eig(prodUTensor)
    tEigEnd = datetime.now()
    print("Eig time: ", tEigEnd - tEigStart)
    phasesTensor=torch.angle(eigTensor)
    return phasesTensor,vecTensor


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

a,b,T1,U=[1,1,1.0,1.0]

def partitionVecs():
    Q,dt,T,T2=calcConsts(a,b,T1)
    tensorHMatAll=genHTensor(T1,T2,T,Q,dt,U)
    phaseTensor, vecTensor=calcEig(Q,tensorHMatAll)
    del tensorHMatAll
    gc.collect()
    betaLeft=[]
    betaRight=[]
    betaMiddle=[]
    phasesLeft=[]
    phasesRight=[]
    phasesMidle=[]

    for m in range(0,len(phaseTensor)):
        phasesTmp=phaseTensor[m]
        vecsTmp=vecTensor[m]
        for i,  onePhase in enumerate(phasesTmp):
            oneVec=np.array(vecsTmp[:,i])
            onePhase = float(onePhase)
            stateNum=selectEdgeStates(oneVec)
            if stateNum==0:
                betaLeft.append(2*m/M)
                phasesLeft.append(onePhase/np.pi)
            elif stateNum==1:
                betaRight.append(2*m/M)
                phasesRight.append(onePhase/np.pi)
            else:
                betaMiddle.append(2*m/M)
                phasesMidle.append(onePhase/np.pi)

    return betaLeft,betaRight,betaMiddle,phasesLeft,phasesRight,phasesMidle




betaLeft,betaRight,betaMiddle, phasesLeft, phasesRight,phasesMiddle=partitionVecs()

lenMax=sorted([len(betaLeft),len(betaRight),len(betaMiddle)])[-1]
#fill with nan so that vectors have the same length
if len(betaLeft)<lenMax:
    betaLeft.extend([np.nan]*(lenMax-len(betaLeft)))
    phasesLeft.extend([np.nan]*(lenMax-len(phasesLeft)))
if len(betaRight)<lenMax:
    betaRight.extend([np.nan]*(lenMax-len(betaRight)))
    phasesRight.extend([np.nan]*(lenMax-len(phasesRight)))
if len(betaMiddle)<lenMax:
    betaMiddle.extend([np.nan]*(lenMax-len(betaMiddle)))
    phasesMiddle.extend([np.nan]*(lenMax-len(phasesMiddle)))


dtFrm=pd.DataFrame({"betaLeft":betaLeft,"phasesLeft":phasesLeft,"betaRight":betaRight,"phasesRight":phasesRight,
                    "betaMiddle":betaMiddle,"phasesMiddle":phasesMiddle})

dtFrm.to_csv("torchObcT1"+str(T1)
             # +"omegaF0"
             +"a"+str(a)+"b"+str(b)
             +"U"+str(U)+"L"+str(L)
             +".csv", index=False)



sVal=2
plt.figure()
plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
         + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +", $U=$"+str(U)
          )

plt.scatter(betaLeft,phasesLeft,color="magenta",marker=".",s=sVal,label="left")
plt.scatter(betaRight,phasesRight,color="cyan",marker=".",s=sVal,label="right")
plt.scatter(betaMiddle,phasesMiddle,color="black",marker=".",s=sVal,label="bulk")
plt.xlabel("$\\beta/\pi$")
plt.ylabel("eigenphase/\pi")
plt.legend()
plt.savefig("obcT1"+str(T1)
            # +"omegaF=0"
            +"a"+str(a)+"b"+str(b)
            +"U"+str(U)
            +".png"
             )
plt.close()