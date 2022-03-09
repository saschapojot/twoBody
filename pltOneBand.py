import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
a=1
b=1
T1=1.0
U=1.0
L=21
# print(matplotlib.__version__)
#read from csv file to determine M, L, etc.
minVal = min(a, b)
maxVal = max(a, b)
dirPrefix="./OneBandT1"+str(T1)+"/U"+str(U)+"/"+"a"+str(minVal)+"b"+str(maxVal)+"/"
inPhasesCSVName=dirPrefix+"phasesAllT1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)+"L"+str(L)+".csv"

inDat=pd.read_csv(inPhasesCSVName,header=None)
bandNum=95

oneBand=inDat.iloc[:,bandNum]
M=int(len(oneBand)/L)

#data serialization
betaInPi=[]
phiInPi=[]
for j in range(0,len(oneBand)):
    r = j % L
    m = int((j - r) / L)
    betaInPi.append(2*m/M)
    phiInPi.append(2*r/L)

print("mean="+str(np.mean(oneBand)))
print("sd="+str(np.std(oneBand)))
fig=plt.figure()
ax=fig.gca(projection="3d")
ftSize=16
surf0 = ax.plot_trisurf(betaInPi, phiInPi, oneBand, linewidth=0.1, color="blue",label="band"+str(bandNum))
ax.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
ax.set_ylabel("$\phi/\pi$",fontsize=ftSize)
ax.set_zlabel("eigenphase$/\pi$",fontsize=ftSize)

plt.title("$T_{1}=$"+str(T1)
          # +", $\omega_{F}=0$"
          + ", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)
          ,fontsize=ftSize)
surf0._facecolors2d=surf0._facecolor3d
surf0._edgecolors2d=surf0._edgecolor3d
plt.legend()
plt.show()
plt.savefig(dirPrefix+"spectrumT1"+str(T1)
             # +"0"
            +"a"+str(a)+"b"+str(b)+"band"+str(bandNum)
            +".png")
# plt.show()
plt.close()