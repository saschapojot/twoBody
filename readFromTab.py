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

#this script verifies if a wannier state can be constructed, and calculated Chern  number
a=2
b=1
T1=1.0
U=0.1

#consts
alpha=1/3
J=2.5
V=2.5
L=21
subLatNum=3#sublattice number
#read from csv file to determine M, L, etc.
minVal = min(a, b)
maxVal = max(a, b)
dirPrefix="./OneBandT1"+str(T1)+"/U"+str(U)+"/"+"a"+str(minVal)+"b"+str(maxVal)+"/"
inVecCSVFileName=dirPrefix+"vecsAllT1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)+"L"+str(L)+".csv"



def stringToInt(str):
    #convert str to index
    return int(np.real(complex(str)))
def stringVecToInt(strVec):
    #convert a vector of strings to a vector of indices
    rst=[]
    for str in strVec:
        rst.append(stringToInt(str))
    return rst

def stringToComplex(str):
    #convert str to a complex number
    return complex(str)

def stringVecToComplex(strVec):
    #convert a vector of strings to a vector of complex numbers
    rst=[]
    for str in strVec:
        rst.append(stringToComplex(str))
    return rst

inDat=pd.read_csv(inVecCSVFileName,header=None)
#each row, elem 0 is betaNum, elem 1 is phiNum

betaIndsAll=stringVecToInt(inDat.iloc[:,0])
phiIndsAll=stringVecToInt(inDat.iloc[:,1])
M=max(betaIndsAll)+1
L=max(phiIndsAll)+1
N=subLatNum*L #total sublattice number
betaValsAll=[2*np.pi*m/M for m in range(0,M)]#adiabatic parameter
phiValsAll=[2*np.pi*r/L for r in range(0,L)]#bloch momentum
#basis
basisAll=boson_basis_1d(N,Nb=2)
basisAllInString=[basisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in basisAll]
Ds=int(basisAll.Ns/L)#momentum space dimension=seed states number

# print("M="+str(M)+", L="+str(L)+", Ns="+str(basisAll.Ns))
#read vec from csv file
bandNum=0
vecTab=[]
for index, row in inDat.iterrows():
    oneRow=[]
    oneRow.append(stringToInt(row[0]))#betaNum
    oneRow.append(stringToInt(row[1]))#phiNum
    vecTmp=row[(2+bandNum*Ds):(2+(bandNum+1)*Ds)]
    oneRow.extend(stringVecToComplex(vecTmp))
    vecTab.append(oneRow)

vecTab=np.array(vecTab)

sortedTab=sorted(vecTab,key=lambda row: row[0])[:L]
#sort by betaNum, such that the first L arrays
#correspond to beta=0, take the first L arrays


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

#construct wannier state
R=int(L/2)
ws=np.zeros(basisAll.Ns,dtype=complex)
for a in range(0,L):
    phiaNum=stringToInt(sortedTab[a][1])
    phia=phiValsAll[phiaNum]
    psiVeca=sortedTab[a][2:]
    for b in range(0,Ds):
        seedTmp=seedStatesAll[b][:]
        for j in range(0,L):
            vecTmp=strToVec(seedTmp)
            ws+=vecTmp*psiVeca[b]*np.exp(1j*(j-R)*phia)
            seedTmp=coTranslation(seedTmp)

ws/=np.linalg.norm(ws,ord=2)

###################magnitude on every site
# subLatOpList=[]
# for j in range(0,N):
#     listTmp = [[1, j]]
#     staticTmp = [["n", listTmp]]
#     opTmp = hamiltonian(staticTmp, [], dtype=np.complex128, basis=basisAll)
#     arrTmp = opTmp.tocsc()
#     subLatOpList.append(arrTmp)
#
# def onSiteMagnitude(vec):
#     """
#
#     :param vec: full vec of length N
#     :return:
#     """
#     mag=[]
#     for arrTmp in subLatOpList:
#         xTmp=arrTmp.dot(vec)
#         yTmp=vec.conj().T.dot(xTmp)
#         mag.append(np.real(yTmp))
#     return mag
# mag=onSiteMagnitude(ws)
#
# plt.figure()
# plt.plot(range(0,N),mag,color="black")
# plt.savefig("ws.png")
#########################################
###################calculates chern number
#construct a tensor of vectors
vecFromBandTensor=np.zeros((M,L,Ds),dtype=complex)
for oneArray in vecTab:
    betaNum=stringToInt(oneArray[0])
    phiNum=stringToInt(oneArray[1])
    vecTmp=oneArray[2:]
    vecFromBandTensor[betaNum,phiNum,:]=vecTmp


chNumFromBand=0
for m in range(0,M):
    for r in range(0,L):
        chNumFromBand+=-np.angle(np.vdot(vecFromBandTensor[m,r,:],vecFromBandTensor[(m+1)%M,r,:])
                                 *np.vdot(vecFromBandTensor[(m+1)%M,r,:],vecFromBandTensor[(m+1)%M,(r+1)%L,:])
                                 *np.vdot(vecFromBandTensor[(m+1)%M,(r+1)%L,:],vecFromBandTensor[m,(r+1)%L,:])
                                 *np.vdot(vecFromBandTensor[m,(r+1)%L,:],vecFromBandTensor[m,r,:]))
chNumFromBand/=(2*np.pi)
print("Chern number of band "+str(bandNum)+" is "+str(chNumFromBand))

#####construct new Hamiltonian
newL=41
newN=subLatNum*newL
newBasisAll=boson_basis_1d(newN,Nb=2)
newBasisAllInString=[newBasisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in newBasisAll]

extraSitesOneSide=int((newN-N)/2)
leftExtraLength=extraSitesOneSide
rightExtraLength=extraSitesOneSide

def truncateStringFromNewBasis(newBasisStr):
    """

    :param newBasisStr: truncate string from the left and right
    :return:
    """
    return newBasisStr[leftExtraLength:-rightExtraLength]


def ifExtendedBasis(newBasisStr, oldBasisStr):
    """

    :param newBasisStr: a basis string in new basis(longer)
    :param oldBasisStr: a basis string in old basis(shorter)
    :return: whether the truncated new basis == old basis, i.e. whether the new basis is an extension of
    the old basis
    """
    truncatedNew=truncateStringFromNewBasis(newBasisStr)
    if truncatedNew==oldBasisStr:
        return 1
    else :
        return 0


def newWannierVec():
    newWS=np.zeros(newBasisAll.Ns,dtype=complex)
    for oldStr in basisAllInString:
        for newStr in newBasisAllInString:
            if ifExtendedBasis(newStr,oldStr)==0:
                continue
            else:
                jOld=basisAll.index(oldStr)
                jNew=newBasisAll.index(newStr)
                vecTmp=np.zeros(newBasisAll.Ns,dtype=complex)
                vecTmp[jNew]=1
                newWS+=ws[jOld]*vecTmp
    return newWS


###################new magnitude on every site
# newSubLatOpList=[]
# for j in range(0,newN):
#     listTmp = [[1, j]]
#     staticTmp = [["n", listTmp]]
#     opTmp = hamiltonian(staticTmp, [], dtype=np.complex128, basis=newBasisAll)
#     arrTmp = opTmp.tocsc()
#     newSubLatOpList.append(arrTmp)
#
# def newOnSiteMagnitude(vec):
#     """
#
#     :param vec: full vec of length newN
#     :return:
#     """
#     mag=[]
#     for arrTmp in newSubLatOpList:
#         xTmp=arrTmp.dot(vec)
#         yTmp=vec.conj().T.dot(xTmp)
#         mag.append(np.real(yTmp))
#     return mag
#
#
# mag=newOnSiteMagnitude(newWannierVec())
# plt.figure()
# plt.plot(range(0,newN),mag,color="black")
# plt.savefig("newws.png")

############################################