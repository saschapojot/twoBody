import  numpy as np
import matplotlib.pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime
import pandas as pd
from multiprocessing import Pool
import torch
import math
from pathlib import Path
#this script calculates band for a range of parameters
#going to scan  T1, U, a, b

#consts
alpha=1/3
J=2.5
V=2.5
T1List=[1,2,4]
UList=[0.1,1,10,20,30]
# Q=100#small time interval number, in this script Q should
#be determined first by time interval length, which is set to 0.05
#by default, and max Q=500
subLatNum=3#sublattice number
L=7##unit cell number, must be odd
N=subLatNum*L #total sublattice number
M=50#beta num
betaValsAll=[2*np.pi*m/M for m in range(0,M)]#adiabatic parameter
phiValsAll=[2*np.pi*r/L for r in range(0,L)]#bloch momentum
#basis
basisAll=boson_basis_1d(N,Nb=2)
basisAllInString=[basisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in basisAll]
Ds=int(basisAll.Ns/L)#momentum space dimension=seed states number

threadNum=24
stepLength=0.05

#####################################
#allocate memory for tensor that will be used repeatedly
prodUTensor=torch.zeros((M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)


###################################

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
    if Q>500:
        Q=500
    dt=T/Q
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
    return [q, m, HTmp.toarray()]



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



def generateAB():
    """

    :return: coprime a, b pairs <=20
    """
    start = 1
    endPast1 = 20 + 1
    pairsAll = []
    for i in range(start, endPast1 - 1):

        for j in range(start, endPast1):
            if math.gcd(i, j) > 1:
                continue
            else:
                pairsAll.append([i, j])
    return pairsAll


abList=generateAB()

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
    # tHMatStart=datetime.now()
    ret0=pool0.map(genOneHMat,inDataAll)
    # tHMatEnd=datetime.now()
    # print("HMat time: ",tHMatEnd-tHMatStart)
    # tInitStart=datetime.now()
    tensorHMatAll=torch.zeros((Q,M,basisAll.Ns,basisAll.Ns),dtype=torch.cfloat)
    for itemTmp in ret0:
        q,m,matTmp=itemTmp
        position=Q-q
        tensorHMatAll[position,m,:,:]=torch.from_numpy(-1j*dt*matTmp)

    # tInitEnd=datetime.now()
    # print("initialization time: ",tInitEnd-tInitStart)
    return tensorHMatAll



def calcEig(a,b,T1,U,Q,tensorHMatAll):
    """
    :param Q: time step num
    :param tensorHMatAll:
    :return: all eigenvals and eigenvecs
    """
    ######matrix exponential
    UqTensorMat = torch.zeros((Q, M, basisAll.Ns, basisAll.Ns), dtype=torch.cfloat)
    # UqTensorMat=UqTensorMat.cuda()
    # tensorHMatAll=tensorHMatAll.cuda()
    # torch.cuda.synchronize()
    # tExpStart = datetime.now()
    for q in range(0, Q):
        UqTensorMat[q, :, :, :] = tensorHMatAll[q, :, :, :].matrix_exp()
    # tExpEnd = datetime.now()
    # torch.cuda.synchronize()
    # print("exp time: ", tExpEnd - tExpStart)
    #########matrix products
    global prodUTensor
    # tProdStart = datetime.now()
    #set prodUTensor to id mat
    for j in range(0,M):
        prodUTensor[j,:,:]=torch.eye(basisAll.Ns,basisAll.Ns,dtype=torch.cfloat)

    for q in range(0,Q):
        prodUTensor=torch.bmm(prodUTensor,UqTensorMat[q,:,:,:])
    # tProdEnd = datetime.now()
    # print("prod time: ", tProdEnd - tProdStart)
    ##################reduced Floquet mats

    pool1=Pool(threadNum)
    betaNumAndPhiNumAll = [[m, r] for m in range(0, M) for r in range(0, L)]
    # tReducedFlMatStart = datetime.now()
    ret1 = pool1.map(reducedFloquetMat, betaNumAndPhiNumAll)
    # tReducedFlMatEnd = datetime.now()
    # print("reduced Floquet mat time: ", tReducedFlMatEnd - tReducedFlMatStart)
    #######reduced Floquet matrices eig
    # m=0,1,...,M-1, r=0,1,...,L-1
    # index of matrix is mL+r
    # tInitRedFlStart = datetime.now()
    reducedFlMatTensor = torch.zeros((M * L, Ds, Ds), dtype=torch.cfloat)
    for itemTmp in ret1:
        m, r, rUMat = itemTmp
        reducedFlMatTensor[m * L + r, :, :] = torch.from_numpy(rUMat)
    # tInitRedFlEnd = datetime.now()
    # print("initialize reduced Floquet matrix tensor time: ", tInitRedFlEnd - tInitRedFlStart)

    # tEigStart = datetime.now()
    eigTensor, vecTensor = torch.linalg.eig(reducedFlMatTensor)
    # tEigEnd = datetime.now()
    # print("Eig time: ", tEigEnd - tEigStart)

    #eigenphases and sort
    phasesAll = torch.angle(eigTensor)

    indsAll = torch.argsort(phasesAll)

    # data serialization
    dataAll = []
    for j in range(0, len(phasesAll)):
        r = j % L
        m = int((j - r) / L)
        oneRow = [m, r]
        indsTmp = indsAll[j]
        phasesTmp = phasesAll[j]
        phasesTmpSorted = [phasesTmp[ind] / np.pi for ind in indsTmp]
        oneRow.extend(phasesTmpSorted)
        dataAll.append(oneRow)

    dataAll = np.array(dataAll)
    ##########data output
    minVal=min(a,b)
    maxVal=max(a,b)
    outDirPrefix="./T1"+str(T1)+"/U"+str(U)+"/"+"a"+str(minVal)+"b"+str(maxVal)+"/"
    Path(outDirPrefix).mkdir(parents=True, exist_ok=True)
    # sort phiNum, such that the first M rows correspond to phi=0
    sortedByPhiDataAll = np.array(sorted(dataAll, key=lambda row: row[1]))
    # plt by beta, section phi=0
    pltByBetaDataAll = sortedByPhiDataAll[:M, :]
    # data serialization
    pltByBetaBeta = []
    pltByBetaPhase = []
    for oneRow in pltByBetaDataAll:
        m = oneRow[0]
        for onePhase in oneRow[2:]:
            pltByBetaBeta.append(2 * m / M)
            pltByBetaPhase.append(onePhase)
    #plt and save
    phiValStr = ", $\phi=$" + str(0)
    plt.figure()
    plt.scatter(pltByBetaBeta, pltByBetaPhase, color="blue", s=1)
    plt.title("$T_{1}=$" + str(T1)
              # +", $\omega_{F}=0$"
              + ", $T_{1}/T_{2}=$" + str(a) + "/" + str(b)
              + ", $U=$" + str(U)
              + phiValStr
              )
    plt.xlabel("$\\beta/\pi$")
    plt.ylabel("eigenphase$/\pi$")
    plt.savefig(outDirPrefix+"torchT1" + str(T1)
                # +"omegaF=0"
                + "a" + str(a) + "b" + str(b)
                + "U" + str(U)
                + "phi0.png")
    plt.close()
    # sort betaNum, such that the first L rows correspond to beta=0
    sortedByBetaAll = np.array(sorted(dataAll, key=lambda row: row[0]))
    # plt by phi, section beta=0
    pltByPhiDataAll = sortedByBetaAll[:L, :]
    # data serialization
    pltByPhiPhi = []
    pltByPhiPhase = []
    for oneRow in pltByPhiDataAll:
        r = oneRow[1]
        for onePhase in oneRow[2:]:
            pltByPhiPhi.append(2 * r / L)
            pltByPhiPhase.append(onePhase)
    #plt and save
    betaValStr = ", $\\beta=$" + str(0)
    plt.figure()
    plt.scatter(pltByPhiPhi, pltByPhiPhase, color="blue", s=1)
    plt.title("$T_{1}=$" + str(T1)
              # +", $\omega_{F}=0$"
              + ", $T_{1}/T_{2}=$" + str(a) + "/" + str(b)
              + ", $U=$" + str(U)
              + betaValStr
              )
    plt.xlabel("$\phi/\pi$")
    plt.ylabel("eigenphase$/\pi$")
    plt.savefig(outDirPrefix+"torchT1" + str(T1)
                # +"omegaF=0"
                + "a" + str(a) + "b" + str(b)
                + "U" + str(U)
                + "beta0.png")
    plt.close()

    #######statistics
    phaseTable = dataAll[:, 2:]
    # col of distToBandBelow is dist of a band to the band below
    distToBandBelow = np.zeros(phaseTable.shape, dtype=float)
    for n in range(0, Ds):
        distTmp = np.abs(phaseTable[:, n] - phaseTable[:, (n - 1) % Ds])
        distToBandBelow[:, n] = distTmp[:]  # deep copy
    # mod 2
    for n in range(0, Ds):
        distToBandBelow[:, n] = distToBandBelow[:, n] % 2

    # col of distToBandAbove is dist of a band to the band above
    distToBandAbove = np.zeros(phaseTable.shape, dtype=float)
    for n in range(0, Ds):
        distToBandAbove[:, n] = distToBandBelow[:, (n + 1) % Ds][:]  # deep copy

    # staticstics of dists
    minDistList = []
    avgDistList = []
    for n in range(0, Ds):
        tmp1 = min(distToBandBelow[:, n])
        tmp2 = min(distToBandAbove[:, n])
        minDistList.append(min(tmp1, tmp2))

    for n in range(0, Ds):
        vec1 = distToBandBelow[:, n][:]
        vec2 = distToBandAbove[:, n][:]  # deep copy
        vec = np.append(vec1, vec2)
        avgDistList.append(np.mean(vec))

    # sort by descending order of minDistList
    indsMinDist = np.argsort(minDistList)[::-1]#col 0 of output table
    #col 1 of output table
    sortedByMinDist_MinDist1 = [minDistList[ind] for ind in indsMinDist]
    #col 2 of output table
    sortedByMinDist_AvgDist2 = [avgDistList[ind] for ind in indsMinDist]
    #sort by descending order pf avgDistList
    indsAvgDist=np.argsort(avgDistList)[::-1]#col 3 of output table
    # col4 of output table
    sortedByAvgDist_MinDist4=[minDistList[ind] for ind in indsAvgDist]
    #col 5 of output table
    sortedByAvgDist_AgvDist5=[avgDistList[ind] for ind in indsAvgDist]

    dataOut=np.array([indsMinDist,sortedByMinDist_MinDist1,sortedByMinDist_AvgDist2,
                      indsAvgDist,sortedByAvgDist_MinDist4,sortedByAvgDist_AgvDist5]).T

    dtFrm=pd.DataFrame(data=dataOut,columns=["indsMinDist","minDist/pi","avgDist/pi",
                                             "indsAvgDist","minDist/pi","avgDist/pi"])
    dtFrm.to_csv(outDirPrefix+"torchDistT1"+str(T1)
             # +"omegaF=0"
             +"a"+str(a)+"b"+str(b)
             +"U="+str(U)
             +".csv", index=False)







def run():
    tAllStart=datetime.now()
    for T1 in T1List:
        for U in UList:
            for onePair in abList:
                a,b=onePair
                Q, dt, T, T2 = calcConsts(a, b, T1)
                tensorHMatAll=genHTensor(T1,T2,T,Q,dt,U)
                calcEig(a,b,T1,U,Q,tensorHMatAll)
    tAllEnd=datetime.now()
    print("total time: ",tAllEnd-tAllStart)






#execution
run()