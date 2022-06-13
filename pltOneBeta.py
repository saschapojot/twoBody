import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#this script reads from all eigenphases and plots beta=0 section
T1=2
U=20
a=3
b=2
L=21#phi takes 21 values
M=50#beta takes 50 values

inDirPrefix="./plotsPumping/T1"+str(T1)+"U"+str(U)+"a"+str(a)+"b"+str(b)+"/"

inFileName=inDirPrefix+"phasesAllT1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)+"L21.csv"

inTab=pd.read_csv(inFileName,header=None)

dataWithIndices=[]
for j in range(0,len(inTab)):
    r=j%L
    m=int((j-r)/L)
    #[betaNum=m, phiNum=r, ...]
    oneRow=[m,r]
    oneRow.extend(inTab.iloc[j,:])
    dataWithIndices.append(oneRow)

#sort according to m, so that the first L=21 rows corrensponds to beta=0
sortedTabByBeta=np.array(sorted(dataWithIndices,key=lambda elem: elem[0]))

pltByPhiDataAll=sortedTabByBeta[:L,:]

#data serialization

pltByPhi_phi=[]
pltByPhi_phases=[]
for oneRow in  pltByPhiDataAll:
    r=oneRow[1]
    for onePhase in oneRow[2:]:
        pltByPhi_phi.append(2*r/L)
        pltByPhi_phases.append(onePhase)

minPhase=min(pltByPhi_phases)
maxPhase=max(pltByPhi_phases)
betaValStr = ", $\\beta=$" + str(0)

sVal=1
fig=plt.figure()
ax=fig.add_subplot(111)
ftSize=16
plt.title("$T_{1}=$"+str(T1)
          +", $U=$"+str(U)
          +", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          +betaValStr,fontsize=ftSize)
plt.scatter(pltByPhi_phi,pltByPhi_phases,color="blue",s=sVal)
ax.set_xlabel("$\phi/\pi$",fontsize=ftSize)
ax.xaxis.set_label_coords(0.5, -0.025)
ax.text(-0.12, 0.60, '(a)', size=15, color='black')
plt.xticks([0,2], fontsize=ftSize )
plt.xlim((0,2))
plt.ylim((minPhase-0.05,maxPhase+0.05))
plt.ylabel("eigenphase$/\pi$",fontsize=ftSize,labelpad=6)
ax.yaxis.set_label_position("right")
maxTick=round(maxPhase,2)
minTick=round(minPhase,2)
plt.yticks([minTick,0,maxTick],fontsize=ftSize )


plt.savefig(inDirPrefix+"phipbcT1"+str(T1)+"U"+str(U)+"a"+str(a)+"b"+str(b)+".eps")
