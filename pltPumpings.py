import matplotlib.pyplot as plt
import pandas as pd


T1=1
U=2
a=1
b=1
dirPrefix="./plotsPumping/"+"T1"+str(T1)+"U"+str(U)+"a"+str(a)+"b"+str(b)+"/"

inFile0=dirPrefix+"T1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)\
        +"band"+str(0)+"betaNum1200"+"displacementSize41.csv"



inTab0=pd.read_csv(inFile0)

timeSteps0=inTab0["t"]
pumping0=inTab0["drift"]



#red: 0
#green: 1
#darkorchid:2
##dimgrey: 95
plt.figure()
ftSize=16
plt.plot(timeSteps0,pumping0,color="red",label="band0")
plt.title("$T_{1}=$"+str(T1)+", $U=$"+str(U)+", $T_{1}/T_{2}=$"+str(a)+"/"+str(b),fontsize=ftSize)
plt.xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
plt.ylabel("pumping",fontsize=ftSize,labelpad=0.5)
plt.yticks([0,-1,-2],fontsize=ftSize)
plt.xticks(fontsize=ftSize)
xMax=1200
plt.hlines(y=-2,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
plt.xlim((0,xMax))
plt.legend(loc="best")

plt.savefig(dirPrefix
            +"a"+str(a)+"b"+str(b)
            +".png")