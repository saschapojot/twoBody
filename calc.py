from funcs import *

outDir="./pump1/"
tStart=datetime.now()
dataAll=[]# list of all wvfcnts
dataAll.append(psi0)
for q in range(0,Q):
    print("q = "+str(q))
    psiCurr=dataAll[q]
    psiNext=S2(q,psiCurr)
    psiNext=reNormalization(psiNext)
    dataAll.append(psiNext)

tEnd=datetime.now()
print("calc time : ",tEnd-tStart)
xPosAll=[]
for q in range(0,Q+1):
    vecTmp=dataAll[q]
    xTmp=avgPos(vecTmp)
    xPosAll.append(xTmp)

drift=[np.real(elem -xPosAll[0]) for elem in xPosAll]

driftImagPart=[np.imag(elem-xPosAll[0]) for elem in xPosAll]
print(np.max(np.abs(driftImagPart)))
tAll=[dt*q for q in range(0,Q+1)]
plt.figure(figsize=(20,20))
plt.plot(tAll,drift,color="black")
plt.xlabel("time")
plt.ylabel("ave position")
plt.title("initial position = "+str(L)+", pumping = "+str(drift[-1]-drift[0]))
plt.savefig(outDir+"init"+str(L)+".png")
plt.close()
