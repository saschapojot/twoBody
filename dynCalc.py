from dynFuncs import *

outDir="./pump1/"
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
# onsiteDrivingCoef1=[[A*np.cos(2*np.pi*alpha*j), j] for j in range(0,N)]#onsite driving coefficient
#quench coefficient
qHop1PM=[[1,j,(j+1)%N] for j in range(0,N)]
qHop1MP=[[1,j,(j+1)%N] for j in range(0,N)]

qHop2PM=[[1,j,(j+1)%N] for j in range(0,N)]
qHop2MP=[[-1,j,(j+1)%N] for j in range(0,N)]

dynamicPart=[["+-",hopDynPM,delta,[]],["-+",hopDynMP,delta,[]],["n",stgPot,Delta,[]]]
H=hamiltonian(staticPart,dynamicPart,static_fmt="csr",dtype=np.complex128,basis=basisAll)

tS=datetime.now()
###########
#init vec

##########
# dataAll=H.evolve(psi0,0,tAll,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False)
normalizedDataAll=[]
normalizedDataAll.append(psi0Vec)
# normalizedDataAll.append(psi0)
for q in range(0,Q):
    psiCurr=normalizedDataAll[q]
    tStart=q*dt
    tEndList=[(q+1)*dt]
    psiNext=H.evolve(psiCurr,tStart,tEndList,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False)[:,0]
    psiNextNormalized=reNormalization(psiNext)
    normalizedDataAll.append(psiNextNormalized)


tEnd=datetime.now()
print("computation time = ",tEnd-tS)

xPosReal=[]
xPosImag=[]
# dataAll=np.array(dataAll)
# rowN,colN=dataAll.shape
# normalizedDataAll=[]
# for cTmp in range(0,colN):
#     vecTmp=reNormalization(dataAll[:,cTmp])
#     normalizedDataAll.append(vecTmp)
# for cTmp in range(0,colN):
#     vecTmp=dataAll[:,cTmp]
#     xPosVal=avgPos(vecTmp)
#     xPosReal.append(np.real(xPosVal))
#     xPosImag.append(np.imag(xPosVal))
for vecTmp in normalizedDataAll:
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

psiEnd=normalizedDataAll[-1]
density_total_end=np.zeros(N)

for x in range(N):
    exec(f'density_total_end[x] = psiEnd.conj().T @ H_n_{x} @ psiEnd')

plt.figure()
plt.plot(np.arange(N), np.abs(density_total_end))
plt.savefig("tmp4.png")
plt.close()
