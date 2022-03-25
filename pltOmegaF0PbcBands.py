import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


L=21#phi takes 21 values
M=50#beta takes 50 values

T1=2
U=20



inDirPrefix="./plotsPumping/T1"+str(T1)+"U"+str(U)+"omegaF0/"

inFileName=inDirPrefix+"phasesAllT1"+str(T1)+"omegaF0"+"U"+str(U)+"L21.csv"

inTab=pd.read_csv(inFileName,header=None)

dataWithIndices=[]
for j in range(0,len(inTab)):
    r=j%L
    m=int((j-r)/L)
    #[betaNum=m, phiNum=r, ...]
    oneRow=[m,r]
    oneRow.extend(inTab.iloc[j,:])
    dataWithIndices.append(oneRow)

#sort according to r, so that the first M=50 rows corrensponds to phi=0

sortedTabByPhi=np.array(sorted(dataWithIndices,key=lambda elem: elem[1]))
pltByBetaDataAll=sortedTabByPhi[:M,:]

#data serialization
pltByBeta_beta=[]
pltByBeta_phase=[]
for oneRow in pltByBetaDataAll:
    m=oneRow[0]
    for onePhase in oneRow[2:]:
        pltByBeta_beta.append(2*m/M)
        pltByBeta_phase.append(onePhase)

minPhase=min(pltByBeta_phase)
maxPhase=max(pltByBeta_phase)

phiValStr = ", $\phi=$" + str(0)

sVal=1
fig=plt.figure()
ax=fig.add_subplot(111)
ftSize=16
plt.title("$T_{1}=$"+str(T1)
          +", $U=$"+str(U)
          +", $\omega_{F}=0$"
          +phiValStr,fontsize=ftSize)
plt.scatter(pltByBeta_beta,pltByBeta_phase,color="blue",s=sVal)
ax.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
ax.xaxis.set_label_coords(0.5, -0.025)
plt.xticks([0,2], fontsize=ftSize )
plt.xlim((0,2))
plt.ylim((minPhase-0.05,maxPhase+0.05))
plt.ylabel("eigenphase$/\pi$",fontsize=ftSize,labelpad=6)
ax.yaxis.set_label_position("right")
maxTick=round(maxPhase,2)
minTick=round(minPhase,2)
plt.yticks([minTick,0,maxTick],fontsize=ftSize )


plt.savefig(inDirPrefix+"pbcT1"+str(T1)+"U"+str(U)+"omegaF0"+".png")