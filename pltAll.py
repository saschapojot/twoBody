import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
T1=2
a=6
b=5


inFileName="dataSpectrumT1"+str(T1)\
           +"omegaF=0"+".csv"
            #+"a"+str(a)+"b"+str(b)\

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

pltBetaAll=[]
pltPhiAll=[]
pltPhaseAll=[]

for itemTmp in dataAll:
    [betaNum,phiNum],phases=itemTmp
    for phaseTmp in phases:
        pltBetaAll.append(2*betaNum/M)
        pltPhiAll.append(2*phiNum/L)
        pltPhaseAll.append(phaseTmp/np.pi)


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111,projection='3d')

sct0=ax.scatter(pltBetaAll,pltPhiAll,pltPhaseAll,marker=".",c="blue")
ax.set_xlabel("$\\beta/\pi$")
ax.set_ylabel("$\phi/\pi$")
ax.set_zlabel("eigenphase$/\pi$")
plt.title("$T_{1}=$"+str(T1)
          +", $\omega_{F}=0$"
          #+ ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          )



plt.savefig("spectrumT1"+str(T1)
            +"omegaF=0"
            #+"a"+str(a)+"b"+str(b)
            +".png")
# plt.show()
plt.close()


