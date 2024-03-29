import matplotlib.pyplot as plt
import pandas as pd


T1=2
U=20
a=3
b=2

dirPrefix="./plotsPumping/"+"T1"+str(T1)+"U"+str(U)+"a"+str(a)+"b"+str(b)+"/"
########################band 0
# inFile0=dirPrefix+"T1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)\
#         +"band"+str(0)+"displacementSize71.csv"
#
# inTab0=pd.read_csv(inFile0)
#
# timeSteps0=inTab0["t"]
# pumping0=inTab0["drift"]
#########################band 1
# inFile1=dirPrefix+"T1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)\
#         +"band"+str(1)+"displacementSize71.csv"
# inTab1=pd.read_csv(inFile1)
#
# timeSteps1=inTab1["t"]
# pumping1=inTab1["drift"]

#############################band 2
# inFile2=dirPrefix+"T1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)\
#         +"band"+str(2)+"displacementSize71.csv"
# inTab2=pd.read_csv(inFile2)
#
# timeSteps2=inTab2["t"]
# pumping2=inTab2["drift"]
##############################band 95
inFile95=dirPrefix+"gaussianT1"+str(T1)+"a"+str(a)+"b"+str(b)+"U"+str(U)\
        +"band"+str(95)+"displacement.csv"
inTab95=pd.read_csv(inFile95)
timeStep95=inTab95["t"]
pumping95=inTab95["drift"]
##################################
#red: 0
#green: 1
#aqua:2
##navy: 95
plt.figure()
ftSize=16
plt.plot(timeStep95,pumping95,color="navy",label="band95")

# plt.plot(timeSteps1,pumping1,color="green",label="band1")
# plt.plot(timeSteps2,pumping2,color="aqua",label="band2")
# plt.plot(timeSteps0,pumping0,color="red",label="band0")
s=1/(2*0.5)
plt.title("$T_{1}=$"+str(T1)+", $U=$"+str(U)+", $T_{1}/T_{2}=$"+str(a)+"/"+str(b)+", Gaussian $\sigma=0.7$",fontsize=ftSize)
plt.xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
plt.ylabel("pumping",fontsize=ftSize,labelpad=0.5)
plt.yticks([-2,-1,0],fontsize=ftSize)
plt.xticks(fontsize=ftSize)
xMax=1000
plt.hlines(y=-2,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
# plt.hlines(y=4,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
plt.xlim((0,xMax))
plt.legend(loc="right",fontsize=ftSize)
plt.text(-60, 0.17, '(c)', size=15, color='black')
plt.savefig(dirPrefix
            +"gaussianT1"+str(T1)
            +"U"+str(U)
            +"a"+str(a)+"b"+str(b)
            +".eps")