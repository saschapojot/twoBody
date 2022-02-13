import numpy as np
import matplotlib.pyplot as plt

T1=2
a=1
b=1


inFileName="dataSpectrumT1"+str(T1)+"omegaF=0"+".csv"#+"a"+str(a)+"b"+str(b)+".csv"



inData=np.loadtxt(inFileName,delimiter=",")

dataAll=[]
for row in inData:
    betaNumAndPhiNum=[int(row[0]),int(row[1])]
    phases=row[2:]
    dataAll.append([betaNumAndPhiNum,phases])
betaNumMax=max(inData[:,0])
phiNumMax=max(inData[:,1])
M=int(betaNumMax)+1
L=int(phiNumMax)+1

Ds=len(dataAll[0][1])

print("Ds = "+str(Ds))

sortedDataAll=sorted(dataAll,key=lambda elem:elem[0][0])
pltData=[sortedDataAll[ind] for ind in range(0,L)]
pltPhi=[]
pltPhase=[]
for itemTmp in pltData:
    phiNum=itemTmp[0][1]
    for phase in itemTmp[1]:
        pltPhi.append(2*phiNum/L)
        pltPhase.append(phase/np.pi)

betaNum=sortedDataAll[0][0][0]
betaValStr=", $\\beta=$"+str(2*betaNum/M)+"$\pi$"

plt.figure()
plt.scatter(pltPhi,pltPhase,color="blue",s=1)
plt.title("$T_{1}=$"+str(T1)
          +", $\omega_{F}=0$"
         # + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +betaValStr
          )
plt.xlabel("$\phi/\pi$")
plt.ylabel("eigenphase$/\pi$")
plt.savefig("T1="+str(T1)
            +"omegaF=0"
            #   +"a"+str(a)+"b"+str(b)
    +"beta=0.png")
