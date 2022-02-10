import numpy as np
import matplotlib.pyplot as plt
from quspin.operators import  hamiltonian
from quspin.basis import boson_basis_1d
from datetime import datetime


#reproduce results from paper

#consts
J=1
delta0=0.8
Delta0=2
U=30
phi0=0
omega=0.005
T=2*np.pi/omega

###
#lattice cell number
L=11#L is odd!
#sublattice number
q=2
N=q*L#total sublattice number
#particle number is 2
#basis
basisAll=boson_basis_1d(2*L,Nb=2)
basisAllInString=[basisAll.int_to_state(numTmp,bracket_notation=False) for numTmp in basisAll]
kValsAll=[2*np.pi*l/(q*L) for l in range(0,L+1)]#momentum values including right endpoint
def coTranslation(stateStr):
    """

    :param stateStr: state repr in string
    :return: translate 1 lattice=2 sublattices to the right
    """
    return stateStr[-q:]+stateStr[:-q]

seedStatesAll=[]
redundantStatesAll=[]
for stateStrTmp in basisAllInString:
    if stateStrTmp in redundantStatesAll:
        continue
    else:
        seedStatesAll.append(stateStrTmp)
        nextStr=stateStrTmp[:]
        for j in range(1,L):
            nextStr=coTranslation(nextStr)
            redundantStatesAll.append(nextStr)

def strToVec(stateStr):
    """

    :param stateStr: basis repr in string
    :return: vector repr for stateStr
    """
    i0=basisAll.index(stateStr)
    psi=np.zeros(basisAll.Ns,dtype=complex)
    psi[i0]=1
    return psi

def seedToMomentumEigVec(kVal,seedStr):
    """

    :param kVal: momentum value
    :param seedStr: seed state in string
    :return: momentum eigvec |k,n>
    """
    retVec=np.zeros(basisAll.Ns,dtype=complex)
    nextStr=seedStr[:]
    for j in range(0,L):
        retVec+=np.exp(1j*kVal*q*j)*strToVec(nextStr)
        nextStr=coTranslation(nextStr)
    retVec/=np.sqrt(L)
    return retVec



def delta(t):
    return delta0*np.sin(omega*t+phi0)

def Delta(t):
    return Delta0*np.cos(omega*t+phi0)


d0=delta(0)#delta(t=0)
D0=Delta(0)#Delta(t=0)
hoppping=[[-J-(-1)**j*d0,j,(j+1)%N] for j in range(0,N)]
staggered=[[(-1)**j*D0,j] for j in range(0,N)]
onSite2=[[U/2,j,j] for j in range(0,N)]
onSite1=[[-U/2,j] for j in range(0,N)]

hListStatic=[["+-",hoppping],["-+",hoppping],["n",staggered],["nn",onSite2],["n",onSite1]]
H0=hamiltonian(hListStatic,[],dtype=np.complex128,basis=basisAll)

H0Mat=H0.toarray()


def HkBlock(k):
    """

    :param k: momentum value
    :return: block Hamiltonian with momentum k
    """
    retHkMat=np.zeros((len(seedStatesAll),len(seedStatesAll)),dtype=complex)
    for i in range(0,len(seedStatesAll)):
        for j in range(0,len(seedStatesAll)):
            str1=seedStatesAll[i]
            str2=seedStatesAll[j]
            vec1=seedToMomentumEigVec(k,str1)
            vec2=seedToMomentumEigVec(k,str2)
            retHkMat[i,j]=vec1.conj().T@H0Mat@vec2
    return retHkMat

tInitStart=datetime.now()
HkBlocksAll=[]#list holding all Hk blocks
for k in kValsAll:
    HkBlocksAll.append(HkBlock(k))

eigVecsAll=[] #list holding all eigvecs for each Hk
eigValsAll=[] #list holding all eigvals for each Hk, in ascending order

for hk in HkBlocksAll:
    eigValsTmp,eigVecsTmp=np.linalg.eig(hk)
    inds=np.argsort(eigValsTmp)[::-1]
    eigValsListTmp=[eigValsTmp[ind] for ind in inds]
    eigVecsListTmp=[eigVecsTmp[:,ind] for ind in inds]
    eigVecsAll.append(eigVecsListTmp)
    eigValsAll.append(eigValsListTmp)

Ds=int(basisAll.Ns/L)

#phase smoothing
# for n in range(0,len(kValsAll)-1):
#     for j in range(0,Ds):
#         dTheta=np.angle(np.vdot(eigVecsAll[n][j],eigVecsAll[n+1][j]))
#         eigVecsAll[n+1][j]*=np.exp(-1j*dTheta)
#
# for j in range(0,Ds):
#     theta=np.vdot(eigVecsAll[0][j],eigVecsAll[-1][j])
#     S=len(kValsAll)-1
#     dTheta=theta/S
#     for n in range(0,S):
#         eigVecsAll[n][j]*=np.exp(-1j*n*dTheta)


def eigVecToFock(k,vec):
    """
    :param k: momentum value
    :param vec: eigenvector in Hk's representation
    :return: eigenvector in Fock basis representation
    """
    retVec=np.zeros(basisAll.Ns,dtype=complex)
    for n in range(0,len(vec)):
        seedStr=seedStatesAll[n]
        retVec+=vec[n]*seedToMomentumEigVec(k,seedStr)
    return retVec



#plot band
#data serialization
kPltAll=[]
EPltAll=[]
for j in range(0,len(kValsAll)-1):
    k=kValsAll[j]
    for ETmp in eigValsAll[j]:
        kPltAll.append(k/np.pi)
        EPltAll.append(ETmp)
plt.figure()
plt.scatter(kPltAll,EPltAll,color="black",s=10)
plt.savefig("E.png")
plt.close()
#construct Wannier state
wState=np.zeros(basisAll.Ns,dtype=complex)

bandNum=0
R=4

for j in range(0,len(kValsAll)-1):
    k=kValsAll[j]
    wState+=np.exp(-1j*k*q*R)*eigVecToFock(k,eigVecsAll[j][bandNum])

wState/=np.linalg.norm(wState,ord=2)

siteOpList=[]
for j in range(0,N):
    listTmp=[[1,j]]
    staticTmp=[["n",listTmp]]
    opTmp=hamiltonian(staticTmp,[],dtype=np.complex128,basis=basisAll)
    arrTmp=opTmp.toarray()
    siteOpList.append(arrTmp)


magList=[]
for arr in siteOpList:
    tmp=np.real(wState.conj().T@arr@wState)
    magList.append(tmp)

plt.plot(range(0,N),magList,color="black")
plt.savefig("ws.png")
plt.close()
tInitEnd=datetime.now()
print("initialization time: ",tInitEnd-tInitStart)
###construct time-dependent Hamiltonian
hoppingStatic=[[-J,j,(j+1)%N] for j in range(0,N)]
staticPart=[["+-",hoppingStatic],["-+",hoppingStatic],["nn",onSite2],["n",onSite1]]
#dynamical part
hoppingDyn=[[(-1)**j,j,(j+1)%N] for j in range(0,N)]
staggeredDyn=[[(-1)**j ,j] for j in range(0,N)]
dynPart=[["+-",hoppingDyn,delta,[]],["-+",hoppingDyn,delta,[]],["n",staggeredDyn,Delta,[]]]
H=hamiltonian(staticPart,dynPart,static_fmt="csr",dtype=np.complex128,basis=basisAll)

psi0=wState[:]
Q=500

datsAll=[psi0]
dt=T/Q
tEvolveStart=datetime.now()
for q in range(0,Q):
    psiCurr=datsAll[q]
    tStart=q*dt
    tEndList=[(q+1)*dt]
    psiNext=H.evolve(psiCurr,tStart,tEndList,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False)[:,0]
    datsAll.append(psiNext)
tEvolveEnd=datetime.now()
print("evolution time: ",tEvolveEnd-tEvolveStart)

#position operator in lattice
posVals=[[j,j]for j in range(0,N)]
posList=[["n",posVals]]
xOpr=hamiltonian(posList,[],basis=basisAll,dtype=np.complex128)
xMat=xOpr.toarray()/2
def avgPos(psiq):
    '''

    :param psiq: wv func
    :return: average position in lattice
    '''
    return psiq.T.conj().dot(xMat).dot(psiq)

xPos=[]
for vecTmp in datsAll:
    xPos.append(np.real(avgPos(vecTmp)))
tValsAll=[q*dt for q in range(0,Q+1)]
drift=[elem-xPos[0] for elem in xPos]
plt.figure()
plt.plot(tValsAll,drift,color="black")
plt.title("drift="+str(drift[-1]))
plt.savefig("drift.png")
plt.close()

psiEnd=datsAll[-1]
magEndList=[]
for arr in siteOpList:
    tmp=np.real(psiEnd.conj().T@arr@psiEnd)
    magEndList.append(tmp)

plt.figure()
plt.plot(range(0,N),magEndList,color="black")
plt.savefig("last.png")
