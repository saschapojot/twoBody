import numpy as np
import matplotlib.pyplot as plt

T1=2
a=1
b=1

U=10
inFileName="GPUdataSpectrumT1"+str(T1)+"omegaF=0"+"U="+str(U)+".csv"#+"a"+str(a)+"b"+str(b)+"U="+str(U)+".csv"
#+"omegaF=0"+".csv"



inData=np.loadtxt(inFileName,delimiter=",")

dataAll=[]
for row in inData:
    betaNumAndPhiNum=[int(row[0]),int(row[1])]
    phases=row[2:]
    dataAll.append([betaNumAndPhiNum,phases])
betaNumMax=max(inData[:,0])
phiNumMax=max(inData[:,1])
M=int(betaNumMax)+1
R=int(phiNumMax)+1

Ds=len(dataAll[0][1])

print("Ds = "+str(Ds))
sortedDataAll=sorted(dataAll,key= lambda elem: elem[0][1])

phiNum=0
pltData=[sortedDataAll[ind] for ind in range(0,M)]

pltBeta=[]
pltPhase=[]
for itemTmp in pltData:
    betaNum=itemTmp[0][0]
    for phase in itemTmp[1]:
        pltBeta.append(2*betaNum/M)
        pltPhase.append(phase/np.pi)

phiNum=sortedDataAll[0][0][1]
phiValStr=", $\phi=$"+str(2*phiNum/R)+"$\pi$"

plt.figure()
plt.scatter(pltBeta,pltPhase,color="blue",s=1)
plt.title("$T_{1}=$"+str(T1)
          +", $\omega_{F}=0$"
         # + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +phiValStr
          )
plt.xlabel("$\\beta/\pi$")
plt.ylabel("eigenphase$/\pi$")
plt.savefig("GPUT1="+str(T1)
            +"omegaF=0"
              # +"a"+str(a)+"b"+str(b)
    +"phi=0.png")