import numpy as np
import pandas as pd


#this script calculates pbc band distances

T1=2
a=1
b=1
U=10

inFileName="dataSpectrumT1"+str(T1)+"omegaF=0"+"U="+str(U)+".csv"#+"a"+str(a)+"b"+str(b)+".csv"



inData=np.loadtxt(inFileName,delimiter=",")
#col 0 is betaNum, col 1 is phiNum

betaNumMax=max(inData[:,0])
phiNumMax=max(inData[:,1])
M=int(betaNumMax)+1
L=int(phiNumMax)+1

Ds=len(inData[0,2:])

print("Ds = "+str(Ds))
#for each phases vector, 0th elem is lowest, Ds-1 the is highest


phaseTable=np.array(inData[:,2:])
phaseTable/=np.pi
#col of distToBandBelow is dist of a band to the band below
distToBandBelow=np.zeros(phaseTable.shape,dtype=float)
for n in range(0,Ds):
    distTmp=np.abs(phaseTable[:,n]-phaseTable[:,(n-1)%Ds])
    distToBandBelow[:,n]=distTmp[:]#deep copy

#mod 2
distToBandBelow[:,0]=distToBandBelow[:,0]%2
distToBandBelow[:,-1]=distToBandBelow[:,-1]%2

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

dtFrm.to_csv("DistT1"+str(T1)
             +"omegaF=0"
             # +"a"+str(a)+"b"+str(b)
             +"U="+str(U)
             +".csv", index=False
             )
