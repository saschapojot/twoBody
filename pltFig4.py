import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

inDir="./fig4/"
#####################
#T11U2a1b1
dir1="T11U2a1b1/"
inCsvSpectrumT11U2a1b1=inDir+dir1+"phasesAllT11a1b1U2L21.csv"
inCsvPumpingT11U2a1b1Band0=inDir+dir1+"T11a1b1U2band0displacementSize41.csv"
inCsvObcT11U2a1b1=inDir+dir1+"torchObcT11U2a1b1.csv"
#T11U10a1b1
dir2="T11U10a1b1/"
inCsvSpectrumT11U10a1b1=inDir+dir2+"phasesAllT11a1b1U10L21.csv"
inCsvPumpingT11U10a1b1Band95=inDir+dir2+"T11a1b1U10band95displacementSize41.csv"
inCsvObcT11U10a1b1=inDir+dir2+"torchObcT11U10a1b1.csv"
#T11U2a1b2
dir3="T11U2a1b2/"
inCsvSpectrumT11U2a1b2=inDir+dir3+"phasesAllT11a1b2U2L21.csv"
inCsvPumpingT11U2a1b2Band0=inDir+dir3+"T11a1b2U2band0displacementSize71.csv"
inCsvObcT11U2a1b2=inDir+dir3+"torchObcT11U2a1b2.csv"
#T11U10a1b2
dir4="T11U10a1b2/"
inCsvSpectrumT11U10a1b2=inDir+dir4+"phasesAllT11a1b2U10L21.csv"
inCsvPumpingT11U10a1b2Band0=inDir+dir4+"T11a1b2U10band0displacementSize71.csv"
inCsvPumpingT11U10a1b2Band1=inDir+dir4+"T11a1b2U10band1displacementSize71.csv"
inCsvPumpingT11U10a1b2Band2=inDir+dir4+"T11a1b2U10band2displacementSize71.csv"
inCsvPumpingT11U10a1b2Band95=inDir+dir4+"T11a1b2U10band95displacementSize71.csv"
inCsvObcT11U10a1b2=inDir+dir4+"torchObcT11U10a1b2.csv"
####################
ftSize=16
fig=plt.figure(figsize=(28,21))
L=21#phi takes 21 values
M=50#beta takes 50 values

####row1, T11U2a1b1
#spectrum
ax1=fig.add_subplot(4,3,1)
inTab1=pd.read_csv(inCsvSpectrumT11U2a1b1,header=None)
dataWithIndices1=[]
for j in range(0,len(inTab1)):
    r=j%L
    m=int((j-r)/L)
    #[betaNum=m, phiNum=r, ...]
    oneRow=[m,r]
    oneRow.extend(inTab1.iloc[j,:])
    dataWithIndices1.append(oneRow)

#sort according to r, so that the first M=50 rows corrensponds to phi=0

sortedTabByPhi1=np.array(sorted(dataWithIndices1,key=lambda elem: elem[1]))
pltByBetaDataAll1=sortedTabByPhi1[:M,:]

#data serialization
pltByBeta_beta1=[]
pltByBeta_phase1=[]
for oneRow in pltByBetaDataAll1:
    m=oneRow[0]
    for onePhase in oneRow[2:]:
        pltByBeta_beta1.append(2*m/M)
        pltByBeta_phase1.append(onePhase)

minPhase1=min(pltByBeta_phase1)
maxPhase1=max(pltByBeta_phase1)

phiValStr1 = ", $\phi=$" + str(0)

sVal=2


ftSize=16
ax1.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(2)
          +", $T_{1}/T_{2}=$"+str(1)+"/"+str(1)
          +phiValStr1,fontsize=ftSize)
ax1.scatter(pltByBeta_beta1,pltByBeta_phase1,color="blue",s=sVal)
ax1.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
ax1.xaxis.set_label_coords(0.5, -0.025)
ax1.text(-0.11, 0.56, '(a)', size=15, color='black')#numbering of figures
ax1.set_xticks([0,2], fontsize=ftSize )
ax1.set_xlim((0,2))
ax1.set_ylim((minPhase1-0.05,maxPhase1+0.05))
ax1.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax1.yaxis.set_label_position("right")
maxTick1=round(maxPhase1,2)
minTick1=round(minPhase1,2)
ax1.set_yticks([minTick1,0,maxTick1],fontsize=ftSize )
#pumping T11U2a1b1
ax2=fig.add_subplot(4,3,2)
inTabPumpingT11U2a1b1Band0=pd.read_csv(inCsvPumpingT11U2a1b1Band0)
ax2.plot(inTabPumpingT11U2a1b1Band0["t"],inTabPumpingT11U2a1b1Band0["drift"]
         ,color="red",label="band0")

ax2.set_title("$T_{1}=$"+str(1)+", $U=$"+str(2)+", $T_{1}/T_{2}=$"+str(1)+"/"+str(1),fontsize=ftSize)
ax2.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax2.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax2.set_yticks([-2,-1,0],fontsize=ftSize)


xMax=1200
xTicks=np.linspace(0,xMax,5)
ax2.set_xticks(xTicks,fontsize=ftSize)
ax2.hlines(y=-2,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
# plt.hlines(y=4,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
ax2.set_xlim((0,xMax))
ax2.legend(loc="best",fontsize=ftSize)
ax2.text(-80, 0.4, '(b)', size=15, color='black')#numbering of figure
#T11U2a1b1 obc
inTabObcT11U2a1b1=pd.read_csv(inCsvObcT11U2a1b1)
ax3=fig.add_subplot(4,3,3)
betaLeft3=inTabObcT11U2a1b1["betaLeft"]
betaLeftData3=[beta for beta in betaLeft3 if ~np.isnan(beta)]

phasesLeft3=inTabObcT11U2a1b1["phasesLeft"]
phasesLeftData3=[ph for ph in phasesLeft3 if ~np.isnan(ph)]
maxPhaseLeft3=max(phasesLeftData3)
minPhaseLeft3=min(phasesLeftData3)

betaRight3=inTabObcT11U2a1b1["betaRight"]
betaRightData3=[beta for beta in betaRight3 if ~np.isnan(beta)]

phasesRight3=inTabObcT11U2a1b1["phasesRight"]
phasesRightData3=[ph for ph in phasesRight3 if ~np.isnan(ph)]
maxPhaseRight3=max(phasesRightData3)
minPhaseRight3=min(phasesRightData3)

betaMiddle3=inTabObcT11U2a1b1["betaMiddle"]
betaMiddleData3=[beta for beta in betaMiddle3 if ~np.isnan(beta)]

phasesMiddle3=inTabObcT11U2a1b1["phasesMiddle"]
phasesMiddleData3=[ph for ph in phasesMiddle3 if ~np.isnan(ph)]
maxPhaseMiddle3=max(phasesMiddleData3)
minPhaseMiddle3=min(phasesMiddleData3)

maxPhase3=sorted([maxPhaseLeft3,maxPhaseRight3,maxPhaseMiddle3])[-1]
minPhase3=sorted([minPhaseLeft3,minPhaseRight3,minPhaseMiddle3])[0]
sVal=2
ax3.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(2)
         + ", $T_{1}/T_{2}=$"+str(1)+"/"+str(1)
             ,fontsize=ftSize)
ax3.scatter(betaLeftData3,phasesLeftData3,color="magenta",marker=".",s=sVal,label="left")
ax3.scatter(betaRightData3,phasesRightData3,color="cyan",marker=".",s=sVal,label="right")
ax3.scatter(betaMiddleData3,phasesMiddleData3,color="black",marker=".",s=sVal,label="bulk")
ax3.set_xlabel("$\\beta/\pi$",fontsize=ftSize)

ax3.xaxis.set_label_coords(0.5, -0.025)
ax3.set_xticks([0,2], fontsize=ftSize )
ax3.set_xlim((0,2))
ax3.set_ylim((minPhase3,maxPhase3))
ax3.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax3.yaxis.set_label_position("right")
ax3.text(-0.1, 0.5, '(c)', size=15, color='black')#numbering of figure
maxTick3=round(maxPhase3,2)
minTick3=round(minPhase3,2)
ax3.set_yticks([minTick3,0,maxTick3],fontsize=ftSize )
lgnd =ax3.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])
######################################################
######################################################
#T11U10a1b1
#spectrum
ax4=fig.add_subplot(4,3,4)
inTab4=pd.read_csv(inCsvSpectrumT11U10a1b1,header=None)
dataWithIndices4=[]
for j in range(0,len(inTab4)):
    r=j%L
    m=int((j-r)/L)
    #[betaNum=m, phiNum=r, ...]
    oneRow4=[m,r]
    oneRow4.extend(inTab4.iloc[j,:])
    dataWithIndices4.append(oneRow4)

#sort according to r, so that the first M=50 rows corrensponds to phi=0

sortedTabByPhi4=np.array(sorted(dataWithIndices4,key=lambda elem: elem[1]))
pltByBetaDataAll4=sortedTabByPhi4[:M,:]

#data serialization
pltByBeta_beta4=[]
pltByBeta_phase4=[]
for oneRow in pltByBetaDataAll4:
    m=oneRow[0]
    for onePhase in oneRow[2:]:
        pltByBeta_beta4.append(2*m/M)
        pltByBeta_phase4.append(onePhase)

minPhase4=min(pltByBeta_phase4)
maxPhase4=max(pltByBeta_phase4)

phiValStr4 = ", $\phi=$" + str(0)

sVal=2


ftSize=16
ax4.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(10)
          +", $T_{1}/T_{2}=$"+str(1)+"/"+str(1)
          +phiValStr4,fontsize=ftSize)
ax4.scatter(pltByBeta_beta4,pltByBeta_phase4,color="blue",s=sVal)
ax4.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
ax4.xaxis.set_label_coords(0.5, -0.025)
ax4.text(-0.11, 0.9, '(d)', size=15, color='black')#numbering of figures
ax4.set_xticks([0,2], fontsize=ftSize )
ax4.set_xlim((0,2))
ax4.set_ylim((minPhase4-0.05,maxPhase4+0.05))
ax4.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax4.yaxis.set_label_position("right")
maxTick4=round(maxPhase4,2)
minTick4=round(minPhase4,2)
ax4.set_yticks([minTick4,0,maxTick4],fontsize=ftSize )
#T11U10a1b1 pumping
ax5=fig.add_subplot(4,3,5)
inTabPumpingT11U10a1b1Band95=pd.read_csv(inCsvPumpingT11U10a1b1Band95)
ax5.plot(inTabPumpingT11U10a1b1Band95["t"],inTabPumpingT11U10a1b1Band95["drift"]
         ,color="navy",label="band95")

ax5.set_title("$T_{1}=$"+str(1)+", $U=$"+str(10)+", $T_{1}/T_{2}=$"+str(1)+"/"+str(1),fontsize=ftSize)
ax5.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax5.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax5.set_yticks([-2,-1,0],fontsize=ftSize)


xMax=2000
xTicks=np.linspace(0,xMax,5)
ax5.set_xticks(xTicks,fontsize=ftSize)
ax5.hlines(y=-2,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
# plt.hlines(y=4,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
ax5.set_xlim((0,xMax))
ax5.legend(loc="best",fontsize=ftSize)
ax5.text(-85, 0.4, '(e)', size=15, color='black')#numbering of figure
#T11U10a1b1 obc
inTabObcT11U10a1b1=pd.read_csv(inCsvObcT11U10a1b1)
ax6=fig.add_subplot(4,3,6)
betaLeft6=inTabObcT11U10a1b1["betaLeft"]
betaLeftData6=[beta for beta in betaLeft6 if ~np.isnan(beta)]

phasesLeft6=inTabObcT11U10a1b1["phasesLeft"]
phasesLeftData6=[ph for ph in phasesLeft6 if ~np.isnan(ph)]
maxPhaseLeft6=max(phasesLeftData6)
minPhaseLeft6=min(phasesLeftData6)

betaRight6=inTabObcT11U10a1b1["betaRight"]
betaRightData6=[beta for beta in betaRight6 if ~np.isnan(beta)]

phasesRight6=inTabObcT11U10a1b1["phasesRight"]
phasesRightData6=[ph for ph in phasesRight6 if ~np.isnan(ph)]
maxPhaseRight6=max(phasesRightData6)
minPhaseRight6=min(phasesRightData6)

betaMiddle6=inTabObcT11U10a1b1["betaMiddle"]
betaMiddleData6=[beta for beta in betaMiddle6 if ~np.isnan(beta)]

phasesMiddle6=inTabObcT11U10a1b1["phasesMiddle"]
phasesMiddleData6=[ph for ph in phasesMiddle6 if ~np.isnan(ph)]
maxPhaseMiddle6=max(phasesMiddleData6)
minPhaseMiddle6=min(phasesMiddleData6)

maxPhase6=sorted([maxPhaseLeft6,maxPhaseRight6,maxPhaseMiddle6])[-1]
minPhase6=sorted([minPhaseLeft6,minPhaseRight6,minPhaseMiddle6])[0]
sVal=2
ax6.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(10)
         + ", $T_{1}/T_{2}=$"+str(1)+"/"+str(1)
             ,fontsize=ftSize)
ax6.scatter(betaLeftData6,phasesLeftData6,color="magenta",marker=".",s=sVal,label="left")
ax6.scatter(betaRightData6,phasesRightData6,color="cyan",marker=".",s=sVal,label="right")
ax6.scatter(betaMiddleData6,phasesMiddleData6,color="black",marker=".",s=sVal,label="bulk")
ax6.set_xlabel("$\\beta/\pi$",fontsize=ftSize)

ax6.xaxis.set_label_coords(0.5, -0.025)
ax6.set_xticks([0,2], fontsize=ftSize )
ax6.set_xlim((0,2))
ax6.set_ylim((minPhase6,maxPhase6))
ax6.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax6.yaxis.set_label_position("right")
ax6.text(-0.1, 0.87, '(f)', size=15, color='black')#numbering of figure
maxTick6=round(maxPhase6,2)
minTick6=round(minPhase6,2)
ax6.set_yticks([minTick6,0,maxTick6],fontsize=ftSize )
lgnd =ax6.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])
######################################################
#T11U2a1b2
#spectrum

ax7=fig.add_subplot(4,3,7)
inTab7=pd.read_csv(inCsvSpectrumT11U2a1b2,header=None)
dataWithIndices7=[]
for j in range(0,len(inTab7)):
    r=j%L
    m=int((j-r)/L)
    #[betaNum=m, phiNum=r, ...]
    oneRow7=[m,r]
    oneRow7.extend(inTab7.iloc[j,:])
    dataWithIndices7.append(oneRow7)

#sort according to r, so that the first M=50 rows corrensponds to phi=0

sortedTabByPhi7=np.array(sorted(dataWithIndices7,key=lambda elem: elem[1]))
pltByBetaDataAll7=sortedTabByPhi7[:M,:]

#data serialization
pltByBeta_beta7=[]
pltByBeta_phase7=[]
for oneRow in pltByBetaDataAll7:
    m=oneRow[0]
    for onePhase in oneRow[2:]:
        pltByBeta_beta7.append(2*m/M)
        pltByBeta_phase7.append(onePhase)

minPhase7=min(pltByBeta_phase7)
maxPhase7=max(pltByBeta_phase7)

phiValStr7 = ", $\phi=$" + str(0)

sVal=2


ftSize=16
ax7.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(2)
          +", $T_{1}/T_{2}=$"+str(1)+"/"+str(2)
          +phiValStr7,fontsize=ftSize)
ax7.scatter(pltByBeta_beta7,pltByBeta_phase7,color="blue",s=sVal)
ax7.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
ax7.xaxis.set_label_coords(0.5, -0.025)
ax7.text(-0.11, 0.6, '(g)', size=15, color='black')#numbering of figures
ax7.set_xticks([0,2], fontsize=ftSize )
ax7.set_xlim((0,2))
ax7.set_ylim((minPhase7-0.05,maxPhase7+0.05))
ax7.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax7.yaxis.set_label_position("right")
maxTick7=round(maxPhase7,2)
minTick7=round(minPhase7,2)
ax7.set_yticks([minTick7,0,maxTick7],fontsize=ftSize )

#T11U2a1b2 pumping

ax8=fig.add_subplot(4,3,8)
inTabPumpingT11U2a1b2Band0=pd.read_csv(inCsvPumpingT11U2a1b2Band0)
ax8.plot(inTabPumpingT11U2a1b2Band0["t"],inTabPumpingT11U2a1b2Band0["drift"]
         ,color="red",label="band0")

ax8.set_title("$T_{1}=$"+str(1)+", $U=$"+str(2)+", $T_{1}/T_{2}=$"+str(1)+"/"+str(2),fontsize=ftSize)
ax8.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax8.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax8.set_yticks([-2,-1,0],fontsize=ftSize)


xMax=1200
xTicks=np.linspace(0,xMax,5)
ax8.set_xticks(xTicks,fontsize=ftSize)
ax8.hlines(y=-2,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
# plt.hlines(y=4,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
ax8.set_xlim((0,xMax))
ax8.legend(loc="best",fontsize=ftSize)
ax8.text(-85, 0.3, '(h)', size=15, color='black')#numbering of figure

#T11U2a1b2 obc
inTabObcT11U2a1b2=pd.read_csv(inCsvObcT11U2a1b2)
ax9=fig.add_subplot(4,3,9)
betaLeft9=inTabObcT11U2a1b2["betaLeft"]
betaLeftData9=[beta for beta in betaLeft9 if ~np.isnan(beta)]

phasesLeft9=inTabObcT11U2a1b2["phasesLeft"]
phasesLeftData9=[ph for ph in phasesLeft9 if ~np.isnan(ph)]
maxPhaseLeft9=max(phasesLeftData9)
minPhaseLeft9=min(phasesLeftData9)

betaRight9=inTabObcT11U2a1b2["betaRight"]
betaRightData9=[beta for beta in betaRight9 if ~np.isnan(beta)]

phasesRight9=inTabObcT11U2a1b2["phasesRight"]
phasesRightData9=[ph for ph in phasesRight9 if ~np.isnan(ph)]
maxPhaseRight9=max(phasesRightData9)
minPhaseRight9=min(phasesRightData9)

betaMiddle9=inTabObcT11U2a1b2["betaMiddle"]
betaMiddleData9=[beta for beta in betaMiddle9 if ~np.isnan(beta)]

phasesMiddle9=inTabObcT11U2a1b2["phasesMiddle"]
phasesMiddleData9=[ph for ph in phasesMiddle9 if ~np.isnan(ph)]
maxPhaseMiddle9=max(phasesMiddleData9)
minPhaseMiddle9=min(phasesMiddleData9)

maxPhase9=sorted([maxPhaseLeft9,maxPhaseRight9,maxPhaseMiddle9])[-1]
minPhase9=sorted([minPhaseLeft9,minPhaseRight9,minPhaseMiddle9])[0]
sVal=2
ax9.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(2)
         + ", $T_{1}/T_{2}=$"+str(1)+"/"+str(2)
             ,fontsize=ftSize)
ax9.scatter(betaLeftData9,phasesLeftData9,color="magenta",marker=".",s=sVal,label="left")
ax9.scatter(betaRightData9,phasesRightData9,color="cyan",marker=".",s=sVal,label="right")
ax9.scatter(betaMiddleData9,phasesMiddleData9,color="black",marker=".",s=sVal,label="bulk")
ax9.set_xlabel("$\\beta/\pi$",fontsize=ftSize)

ax9.xaxis.set_label_coords(0.5, -0.025)
ax9.set_xticks([0,2], fontsize=ftSize )
ax9.set_xlim((0,2))
ax9.set_ylim((minPhase9,maxPhase9))
ax9.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax9.yaxis.set_label_position("right")
ax9.text(-0.1, 1.15, '(i)', size=15, color='black')#numbering of figure
maxTick9=round(maxPhase9,2)
minTick9=round(minPhase9,2)
ax9.set_yticks([minTick9,0,maxTick9],fontsize=ftSize )
lgnd =ax9.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])
######################################################

######################################################
#T11U10a1b2
#spectrum
ax10=fig.add_subplot(4,3,10)
inTab10=pd.read_csv(inCsvSpectrumT11U10a1b2,header=None)
dataWithIndices10=[]
for j in range(0,len(inTab10)):
    r=j%L
    m=int((j-r)/L)
    #[betaNum=m, phiNum=r, ...]
    oneRow10=[m,r]
    oneRow10.extend(inTab10.iloc[j,:])
    dataWithIndices10.append(oneRow10)

#sort according to r, so that the first M=50 rows corrensponds to phi=0

sortedTabByPhi10=np.array(sorted(dataWithIndices10,key=lambda elem: elem[1]))
pltByBetaDataAll10=sortedTabByPhi10[:M,:]

#data serialization
pltByBeta_beta10=[]
pltByBeta_phase10=[]
for oneRow in pltByBetaDataAll10:
    m=oneRow[0]
    for onePhase in oneRow[2:]:
        pltByBeta_beta10.append(2*m/M)
        pltByBeta_phase10.append(onePhase)

minPhase10=min(pltByBeta_phase10)
maxPhase10=max(pltByBeta_phase10)

phiValStr10 = ", $\phi=$" + str(0)

sVal=2


ftSize=16
ax10.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(10)
          +", $T_{1}/T_{2}=$"+str(1)+"/"+str(2)
          +phiValStr10,fontsize=ftSize)
ax10.scatter(pltByBeta_beta10,pltByBeta_phase10,color="blue",s=sVal)
ax10.set_xlabel("$\\beta/\pi$",fontsize=ftSize)
ax10.xaxis.set_label_coords(0.5, -0.025)
ax10.text(-0.11, 0.9, '(j)', size=15, color='black')#numbering of figures
ax10.set_xticks([0,2], fontsize=ftSize )
ax10.set_xlim((0,2))
ax10.set_ylim((minPhase10-0.05,maxPhase10+0.05))
ax10.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax10.yaxis.set_label_position("right")
maxTick10=round(maxPhase10,2)
minTick10=round(minPhase10,2)
ax10.set_yticks([minTick10,0,maxTick10],fontsize=ftSize )
#T11U10a1b2 pumping
ax11=fig.add_subplot(4,3,11)
inTabPumpingT11U10a1b2Band0=pd.read_csv(inCsvPumpingT11U10a1b2Band0)
inTabPumpingT11U10a1b2Band1=pd.read_csv(inCsvPumpingT11U10a1b2Band1)
inTabPumpingT11U10a1b2Band2=pd.read_csv(inCsvPumpingT11U10a1b2Band2)
inTabPumpingT11U10a1b2Band95=pd.read_csv(inCsvPumpingT11U10a1b2Band95)
ax11.plot(inTabPumpingT11U10a1b2Band95["t"],inTabPumpingT11U10a1b2Band95["drift"]
         ,color="navy",label="band95")


ax11.plot(inTabPumpingT11U10a1b2Band1["t"],inTabPumpingT11U10a1b2Band1["drift"]
         ,color="green",label="band1")
ax11.plot(inTabPumpingT11U10a1b2Band2["t"],inTabPumpingT11U10a1b2Band2["drift"]
         ,color="aqua",label="band2")
ax11.plot(inTabPumpingT11U10a1b2Band0["t"],inTabPumpingT11U10a1b2Band0["drift"]
         ,color="red",label="band0")
ax11.set_title("$T_{1}=$"+str(1)+", $U=$"+str(10)+", $T_{1}/T_{2}=$"+str(1)+"/"+str(2),fontsize=ftSize)
ax11.set_xlabel("$t/T$",fontsize=ftSize,labelpad=0.5)
ax11.set_ylabel("pumping",fontsize=ftSize,labelpad=0.5)
ax11.set_yticks([-2,0,2,4],fontsize=ftSize)


xMax=1500
xTicks=np.linspace(0,xMax,5)
ax11.set_xticks(xTicks,fontsize=ftSize)
ax11.hlines(y=-2,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
ax11.hlines(y=4,xmin=0,xmax=xMax,linewidth=0.5,color="k",linestyles="--")
ax11.set_xlim((0,xMax))
ax11.legend(loc="best",fontsize=ftSize)
ax11.text(-85, 4.8, '(k)', size=15, color='black')#numbering of figure
#T11U10a1b2 obc

inTabObcT11U10a1b2=pd.read_csv(inCsvObcT11U10a1b2)
ax12=fig.add_subplot(4,3,12)
betaLeft12=inTabObcT11U10a1b2["betaLeft"]
betaLeftData12=[beta for beta in betaLeft12 if ~np.isnan(beta)]

phasesLeft12=inTabObcT11U10a1b2["phasesLeft"]
phasesLeftData12=[ph for ph in phasesLeft12 if ~np.isnan(ph)]
maxPhaseLeft12=max(phasesLeftData12)
minPhaseLeft12=min(phasesLeftData12)

betaRight12=inTabObcT11U10a1b2["betaRight"]
betaRightData12=[beta for beta in betaRight12 if ~np.isnan(beta)]

phasesRight12=inTabObcT11U10a1b2["phasesRight"]
phasesRightData12=[ph for ph in phasesRight12 if ~np.isnan(ph)]
maxPhaseRight12=max(phasesRightData12)
minPhaseRight12=min(phasesRightData12)

betaMiddle12=inTabObcT11U10a1b2["betaMiddle"]
betaMiddleData12=[beta for beta in betaMiddle12 if ~np.isnan(beta)]

phasesMiddle12=inTabObcT11U10a1b2["phasesMiddle"]
phasesMiddleData12=[ph for ph in phasesMiddle12 if ~np.isnan(ph)]
maxPhaseMiddle12=max(phasesMiddleData12)
minPhaseMiddle12=min(phasesMiddleData12)

maxPhase12=sorted([maxPhaseLeft12,maxPhaseRight12,maxPhaseMiddle12])[-1]
minPhase12=sorted([minPhaseLeft12,minPhaseRight12,minPhaseMiddle12])[0]
sVal=2
ax12.set_title("$T_{1}=$"+str(1)
          +", $U=$"+str(10)
         + ", $T_{1}/T_{2}=$"+str(1)+"/"+str(2)
             ,fontsize=ftSize)
ax12.scatter(betaLeftData12,phasesLeftData12,color="magenta",marker=".",s=sVal,label="left")
ax12.scatter(betaRightData12,phasesRightData12,color="cyan",marker=".",s=sVal,label="right")
ax12.scatter(betaMiddleData12,phasesMiddleData12,color="black",marker=".",s=sVal,label="bulk")
ax12.set_xlabel("$\\beta/\pi$",fontsize=ftSize)

ax12.xaxis.set_label_coords(0.5, -0.025)
ax12.set_xticks([0,2], fontsize=ftSize )
ax12.set_xlim((0,2))
ax12.set_ylim((minPhase12,maxPhase12))
ax12.set_ylabel("quasienergy$/\pi$",fontsize=ftSize,labelpad=6)
ax12.yaxis.set_label_position("right")
ax12.text(-0.1, 0.96, '(l)', size=15, color='black')#numbering of figure
maxTick12=round(maxPhase12,2)
minTick12=round(minPhase12,2)
ax12.set_yticks([minTick12,0,maxTick12],fontsize=ftSize )
lgnd =ax12.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2),fontsize=ftSize)
for handle in lgnd.legendHandles:
    handle.set_sizes([25.0])
######################################################

######################################################

plt.subplots_adjust(left=0.1,
                    bottom=0.04,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

plt.savefig(inDir+"fig4"
            +".pdf")
