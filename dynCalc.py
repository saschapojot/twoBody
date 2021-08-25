from dynFuncs import *

outDir="./pump0/"
tAll=[q*dt for q in range(0,Q+1)]
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

dynamicPart=[["+-",hopDynPM,delta,[]],["-+",hopDynMP,delta,[]],["n",stgPot,Delta,[]]]
H=hamiltonian(staticPart,dynamicPart,static_fmt="csr",dtype=np.complex128,basis=basisAll)

tStart=datetime.now()
dataAll=H.evolve(psi0,0,tAll,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False)

tEnd=datetime.now()
print("computation time = ",tEnd-tStart)

xPosReal=[]
xPosImag=[]
dataAll=np.array(dataAll)
rowN,colN=dataAll.shape
for cTmp in range(0,colN):
    vecTmp=dataAll[:,cTmp]
    xPosVal=avgPos(vecTmp)
    xPosReal.append(np.real(xPosVal))
    xPosImag.append(np.imag(xPosVal))

nT=int(tTot/T)
xTickVals=[n*T for n in range(0,nT+1)]
xTickLabels=[n for n in range(0,nT+1)]


print(np.max(np.abs(xPosImag)))
drift=[elem-xPosReal[0] for elem in xPosReal]
plt.figure(figsize=(20,20))
plt.plot(tAll,drift,color="black")
plt.xticks(xTickVals,xTickLabels)
plt.xlabel("time/T")
plt.ylabel("ave position")
plt.title("initial position = "+str(L)+", pumping = "+str(drift[-1]-drift[0]))
plt.savefig(outDir+"init"+str(L)+".png")
plt.close()
