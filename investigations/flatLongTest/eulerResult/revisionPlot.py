# @Author: Anthony Walker <sokato>
# @Date:   2020-04-17T17:31:11-07:00
# @Email:  walkanth@oregonstate.edu
# @Filename: revisionPlot.py
# @Last modified by:   sokato
# @Last modified time: 2020-04-20T12:38:00-07:00


import os
import matplotlib.pyplot as plt
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
save_path = os.path.join(curr_path,"figs")
def bestRunsSpeedUp():
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
    plt.savefig(os.path.join(save_path,"best-runs-speedup.pdf"))

def fileDisect(f):
    data =[]
    line = f.readline()
    line = f.readline()[:-1]
    while line:
        lineData = [float(l) if "." in l else int(l) for l in line.split(',')]
        data.append(lineData)
        line = f.readline()[:-1]
    return data

def generateFigure(size,data,value):
    fig = plt.figure()
    plt.grid(b=True)
    plt.xlabel('Grid size')
    plt.ylabel('Speedup-{}'.format(value))
    # plt.xlim([0,4])
    lstyle = ['solid','dashed','dotted']
    strs = ['Classic lengthening', 'Swept flattening', 'Swept lengthening']
    colors = ['red','green','blue']
    for i in range(data.shape[1]):
        plt.semilogx(size,data[:,i],color=colors[i-1],linestyle=lstyle[i-1],linewidth=3)
        idx = len(size)//2
        plt.text(size[idx], data[-1,i], strs[i-1], fontsize=12)
    plt.savefig(os.path.join(save_path,"speedup-{}.pdf".format(value)))

def blockSizeSpeedUp():
    func = lambda currStr: True if ".csv" in currStr and currStr!="bestRuns.csv" else False
    files = list(filter(func,os.listdir(curr_path)))
    files.sort()
    curr_file = os.path.join(curr_path,files[0])
    with open(curr_file,'r') as f:
        rdata = fileDisect(f)
    blocks,grids,ndata = zip(*rdata)
    ndata = np.asarray(ndata)
    blocks = list(set(blocks))
    grids = list(set(grids))
    blocks.sort()
    grids.sort()
    ng = len(grids)
    nb = len(blocks)
    nf = len(files)
    norm_data = np.zeros((len(ndata),nf-1))
    for i,cfn in enumerate(files[1:]):
        curr_file = os.path.join(curr_path,cfn)
        with open(curr_file,'r') as f:
            rdata = fileDisect(f)
            cdata = np.asarray(list(zip(*rdata))[-1])/ndata
            norm_data[:,i] = cdata[:]

    average = np.zeros((ng,nf-1))
    for blk,i in enumerate(range(0,int(nb*ng),ng)):
        dataSet = norm_data[i:i+ng,:]
        average += dataSet
        generateFigure(grids,dataSet,blocks[blk])
    average /= nb #taking average
    generateFigure(grids,average,"average")
    # with open(curr_file,'r') as f:
    #     line = f.readline()
    #     data =[]
    #     while line:
    #         line = f.readline()
    #         data.append(line)
    #     print(len(data))
    #     print(data)
    #     f.close()

if __name__ == "__main__":
    # bestRunsSpeedUp()
    blockSizeSpeedUp()
