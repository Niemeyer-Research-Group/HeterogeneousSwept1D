# @Author: Anthony Walker <sokato>
# @Date:   2020-04-17T17:31:11-07:00
# @Email:  walkanth@oregonstate.edu
# @Filename: revisionPlot.py
# @Last modified by:   sokato
# @Last modified time: 2020-04-18T10:28:56-07:00


import os
import matplotlib.pyplot as plt
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
curr_file = os.path.join(curr_path,'bestRuns.csv')


with open(curr_file,'r') as f:
    line = f.readline()
    line = f.readline()
    data = []
    while line:
        data.append([float(item) for item in line.split(',')])
        line = f.readline()
    f.close()
#Normalizing data
norm_index = 0
data = np.asarray(data)
size = data[:,0]
data = data[:,1:]
for i,row in enumerate(data):
    row /= data[i,norm_index]

#Plot data
fig = plt.figure()
plt.grid(b=True)
plt.xlabel('Grid size')
plt.ylabel('Speedup')
# plt.xlim([0,4])
lstyle = ['solid','dashed','dotted']
strs = ['Classic lengthening', 'Swept flattening', 'Swept lengthening']
colors = ['red','green','blue']
for i in range(1,data.shape[1],1):
    plt.semilogx(size,data[:,i],color=colors[i-1],linestyle=lstyle[i-1],linewidth=3)
    idx = len(size)//2
    plt.text(size[idx], data[-1,i], strs[i-1], fontsize=12)
plt.savefig(os.path.join(curr_path,"speedup.pdf"))
plt.show()
