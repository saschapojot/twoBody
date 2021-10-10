from floquetFuncs import *

tPumpingValsAll=[dt*j for j in range(0,Q+1)]
outDir="./pump2/"

#static part
onSite2=[[U/2,j,j] for j in range(0,N)]
onSite1=[[-U/2,j]for j in range(0,N)]
tiltedPot=[[omegaF*j,j] for j in range(0,N)]
hopConst=[[-J,j,(j+1)%N] for j in range(0,N)]
staticPart=[["n",onSite1],["n",tiltedPot],["nn",onSite2],["+-",hopConst],["-+",hopConst]]
#dynamical part

hopDynPM=[[(-1)**j,j,(j+1)%N] for j in range(0,N)]
hopDynMP=[[(-1)**j ,j,(j+1)%N] for j in range(0,N)]
stgPot=[[(-1)**j,j] for j in range(0,N)]

dynamicPart=[["+-",hopDynPM,delta,[]],["-+",hopDynMP,delta,[]],["n",stgPot,D,[]],["n",stgPot,FloquetDriving,[]]]

H=hamiltonian(staticPart,dynamicPart,static_fmt="csr",dtype=np.complex128,basis=basisAll)
tS=datetime.now()

dataAll=[psi0Vec]
for q in range(0,Q):
    psiCurr=dataAll[q]
    tStart=q*dt
    tEndList=[(q+1)*dt]
    psiNext=H.evolve(psiCurr,tStart,tEndList,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False)[:,0]
    dataAll.append(psiNext)

tEnd=datetime.now()
print("computation time = ",tEnd-tS)

xPosReal=[]
xPosImag=[]
for vecTmp in dataAll:
    xPosVal=avgPos(vecTmp)
    xPosReal.append(np.real(xPosVal))
    xPosImag.append(np.imag(xPosVal))

nT=int(tTot/T)
xTickVals=[n*T for n in range(0,nT+1)]
xTickLabels=[n for n in range(0,nT+1)]


print(np.max(np.abs(xPosImag)))
# vecLast=normalizedDataAll[:,-1]
# print("norm2 = ",vecLast.T.conj().dot(vecLast))
drift=[elem-xPosReal[0] for elem in xPosReal]
plt.figure(figsize=(20,20))
plt.plot(tAll,drift,color="black")
plt.xticks(xTickVals,xTickLabels)
plt.xlabel("time/T")
plt.ylabel("ave position")
plt.title("initial position = "+str(L)+", pumping = "+str(drift[-1]-drift[0]))
plt.savefig(outDir+"L="+str(L)+"omegaF="+str(omegaF)+"omega"+str(omega)+"sgm"+str(sgm)+".png")
plt.close()

