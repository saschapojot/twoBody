import subprocess
import math
from datetime import datetime



#1~10
def generateAB():
    """

    :return: coprime a, b pairs <=20
    """
    start = 1
    endPast1 = 10 + 1
    pairsAll = []
    for i in range(start, endPast1 - 1):

        for j in range(start, endPast1):
            if math.gcd(i, j) > 1:
                continue
            else:
                pairsAll.append([i, j])
    return pairsAll

# def kill(proc_pid):
#     process = psutil.Process(proc_pid)
#     for proc in process.children(recursive=True):
#         proc.kill()
#     process.kill()

abList=generateAB()
T1List=[1,2,4]
UList=[0.1,1,10,20,30]
tAllStart=datetime.now()
num=0
for onePair in abList:
    a,b=onePair
    # os.system("singularity exec /app/singularity/images/pytorch/pytorch-1.5.1-cuda10.0-cudnn7.5.simg /home/users/nus/e0385051/anaconda3/bin/python batchTorchSpectrum.py "+str(a)+" "+str(b))
    for T1 in T1List:
        for U in UList:
            pr=subprocess.Popen(["singularity exec /app/singularity/images/pytorch/pytorch-1.5.1-cuda10.0-cudnn7.5.simg /home/users/nus/e0385051/anaconda3/bin/python batchTorchSpectrum.py "+str(a)+" "+str(b)+" "+str(T1)+" "+str(U)],shell=True)
            pr.wait()
            num+=1
            # kill(pr.pid)
            # pr.kill()
            # print("a="+str(a)+", b="+str(b)+", T1="+str(T1)+", U="+str(U))

tEndAll=datetime.now()
print("all time: ",tEndAll-tAllStart)
avgTime=(tEndAll-tAllStart)/num
print("avg time: ",avgTime)