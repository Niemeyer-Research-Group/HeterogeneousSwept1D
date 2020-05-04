# @Author: Anthony Walker <walkanth>
# @Date:   2020-02-21T11:28:13-08:00
# @Email:  dev.sokato@gmail.com
# @Last modified by:   walkanth
# @Last modified time: 2020-03-06T15:19:40-08:00



#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys, os
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
# from collections.abc import Iterable
# import matplotlib.animation as animation
# from mpl_toolkits import mplot3d
# from matplotlib.patches import FancyArrowPatch
import numpy as np
from matplotlib import transforms
from itertools import cycle
dir = os.path.join(os.path.dirname(__file__),'figs')
transparency = 0.25
fsyms = ['o','s','p']
def generateSets(*args):
    """Use this function to generate sets based on block size."""
    colors,symbols,blocksize,boundaryPoints,numberOfBlocks = args
    totalPoints = blocksize+2*boundaryPoints
    nsteps = int(blocksize/(2*boundaryPoints))
    halfBlockSize = int(blocksize/2)
    domMax = blocksize*numberOfBlocks-1
    #Upsets
    upsets  = list()
    for j in range(0,nsteps):
        upsets.append([(i,j) for i in range(j*boundaryPoints,blocksize-j*boundaryPoints)])

    upsetsBlocks = (upsets,)
    for k in range(1,numberOfBlocks,1):
        tlist = list()
        for uset in upsets:
            tlist.append([(i+int(k*blocksize),j)  for i, j in uset])
        upsetsBlocks += tlist,
    #Diamond Sets
    diamondsets = list()
    for i in range(1,nsteps):
        diamondsets.append([(j+halfBlockSize,i) for j in range(halfBlockSize-(i)*boundaryPoints,halfBlockSize+(i)*boundaryPoints,1)])
    for set in upsets[1:]:
        diamondsets.append([(dx+halfBlockSize,dy+nsteps-1) for dx,dy in set])
    diamondBlocks = (diamondsets,)
    for k in range(1,numberOfBlocks,1):
        tlist = list()
        for diaset in diamondsets:
            tlist.append([(i+int(k*blocksize),j) if i+int(k*blocksize)<=domMax else (i-blocksize,j) for i, j in diaset])
        diamondBlocks += tlist,
    #Downsets
    downsets = list()
    for i in range(nsteps,2*nsteps-1):
        downsets.append([(j,i)  for j in range(halfBlockSize-(i-nsteps+1)*boundaryPoints,halfBlockSize+(i-nsteps+1)*boundaryPoints,1)])
    downsetsBlocks = (downsets,)
    for k in range(1,numberOfBlocks,1):
        tlist = list()
        for dset in downsets:
            tlist.append([(i+int(k*blocksize),j) if i+int(k*blocksize)<=domMax else (i-blocksize,j) for i, j in dset])
        downsetsBlocks += tlist,
    return upsetsBlocks,diamondBlocks,downsetsBlocks

def getEdgeBlocks(blocks,boundaryPoints,blocksize):
    """Use this function to highlight edge points."""
    edgeBlocks = tuple()
    for block in blocks:
        edgeSet = list()
        for set in block:
            edgeSet.append(set[:boundaryPoints*2]+set[-2*boundaryPoints:])
        edgeBlocks += edgeSet,
    return edgeBlocks

def consistentAxesOne(ax,blocksize,numberOfBlocks,nsteps,offset=False):
    """Use this function for consistency amongst axes."""
    #Setting axes
    yax = -0.4
    xax = -1.5
    axisColor = "black"
    axisWidth = 2
    xlabCoords = (0.4,0.1)
    ylabCoords = (-0.02,0.4)
    ax.text(xlabCoords[0],xlabCoords[1],"Spatial Points",transform=ax.transAxes)
    ax.text(ylabCoords[0],ylabCoords[1],"Timestep",rotation=90,transform=ax.transAxes)
    ax.plot((xax,blocksize*numberOfBlocks),(yax,yax),color=axisColor,linewidth=axisWidth)
    ax.plot((xax,xax),(yax,4*nsteps),color=axisColor,linewidth=axisWidth)
    ax.set_ylabel("Time Step")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_xlim(-2,numberOfBlocks*blocksize+2)
    ax.set_ylim(-2,2*nsteps)
    #Setting node barriers
    axCoords = (0.25,0.9)
    base = blocksize/2-0.5 if offset else blocksize-0.5
    # for i in range(0,numberOfBlocks):
    #     ax.plot((base+i*blocksize,base+i*blocksize),(yax,4*nsteps),linestyle="--",linewidth=axisWidth,color=axisColor)

def UpTriangle(*args):
    """Call this function to generate the first step of the process."""
    points,edges,colors,blocksize,numberOfBlocks,nsteps = args
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    consistentAxesOne(ax,blocksize,numberOfBlocks,nsteps)
    ax.set_title("Up Triangle")
    ax.axis('off')
    #Setting points
    lines = []
    for i,block in enumerate(points):
        for set in block:
            X,Y = zip(*(set))
            currLine = ax.scatter(X,Y,marker=fsyms[i],edgecolor="black",color=colors[i])
        lines.append(currLine)
    nstrs = ["Node "+str(i) for i in range(len(points))]
    ax.legend(lines,nstrs,loc="upper center",ncol=numberOfBlocks)
    plt.savefig(os.path.join(dir,"a_UpTriangle.pdf"))

def Comm1(*args):
    """Call this function to represent the communcation steps."""
    points,edges,colors,blocksize,numberOfBlocks,nsteps = args
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    consistentAxesOne(ax,blocksize,numberOfBlocks,nsteps)
    ax.set_title("Communication 1")
    ax.axis('off')
    #Setting points
    for i,block in enumerate(points):
        for set in block:
            X,Y = zip(*(set))
            ax.plot(X,Y,linestyle=" ",marker=fsyms[i],color='k',zorder=1,alpha=transparency)
    lines = []
    for i,block in enumerate(edges):
        for set in block:
            X,Y = zip(*(set))
            currLine = ax.scatter(X,Y,marker=fsyms[i],edgecolor="black",facecolor=colors[i],zorder=2)
        lines.append(currLine)
    nstrs = ["Node "+str(i) for i in range(len(points))]
    ax.legend(lines,nstrs,loc="upper center",ncol=numberOfBlocks)
    plt.savefig(os.path.join(dir,"b_Comm1.pdf"))


def Diamond(*args):
    """Call this function to generate the intermediate steps of the process."""
    points,uppoints,edges,colors,blocksize,numberOfBlocks,nsteps = args
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    consistentAxesOne(ax,blocksize,numberOfBlocks,nsteps,offset=True)
    ax.set_title("Diamond")
    ax.axis('off')
    for i,block in enumerate(uppoints):
        for set in block:
            X,Y = zip(*(set))
            ax.plot(X,Y,linestyle=" ",marker=fsyms[i],color="k",zorder=0,alpha=transparency)
    #Setting points
    lines = []
    for i,block in enumerate(points):
        for set in block:
            X,Y = zip(*(set))
            currLine = ax.scatter(X,Y,marker=fsyms[i],edgecolor="black",facecolor=colors[i],zorder=2)
        lines.append(currLine)
    nstrs = ["Node "+str(i) for i in range(len(points))]
    ax.legend(lines,nstrs,loc="upper center",ncol=numberOfBlocks)

    plt.savefig(os.path.join(dir,"c_Diamond.pdf"))

def Comm2(*args):
    """Call this function to represent the communcation steps."""
    points,uppoints,edges,colors,blocksize,numberOfBlocks,nsteps = args
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    consistentAxesOne(ax,blocksize,numberOfBlocks,nsteps,offset=True)
    ax.set_title("Communication 2")
    ax.axis('off')
    #Setting points
    for i,block in enumerate(uppoints):
        for set in block:
            X,Y = zip(*(set))
            ax.plot(X,Y,linestyle=" ",marker=fsyms[i],color='k',zorder=0,alpha=transparency)
    #Setting points
    for i,block in enumerate(points):
        for set in block:
            X,Y = zip(*(set))
            ax.plot(X,Y,linestyle=" ",marker=fsyms[i],color='k',zorder=1,alpha=transparency)
    lines = []
    for i,block in enumerate(edges):
        for set in block:
            X,Y = zip(*(set))
            currLine = ax.scatter(X,Y,marker=fsyms[i],edgecolor="black",facecolor=colors[i],zorder=2)
        lines.append(currLine)
    nstrs = ["Node "+str(i) for i in range(len(points))]
    ax.legend(lines,nstrs,loc="upper center",ncol=numberOfBlocks)
    plt.savefig(os.path.join(dir,"d_Comm2.pdf"))


def DownTriangle(*args):
    """Call this function to generate the last step of the process."""
    points,uppoints,diapoints,edges,colors,blocksize,numberOfBlocks,nsteps = args
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    consistentAxesOne(ax,blocksize,numberOfBlocks,nsteps,offset=False)
    ax.set_title("Down Triangle")
    ax.axis('off')
    #Setting points
    for i,block in enumerate(uppoints):
        for set in block:
            X,Y = zip(*(set))
            ax.plot(X,Y,linestyle=" ",marker=fsyms[i],color='k',zorder=0,alpha=transparency)

    for i,block in enumerate(diapoints):
        for set in block:
            X,Y = zip(*(set))
            ax.plot(X,Y,linestyle=" ",marker=fsyms[i],color='k',zorder=1,alpha=transparency)
    lines = []
    for i,block in enumerate(points):
        for set in block:
            X,Y = zip(*(set))
            currLine = ax.scatter(X,Y,marker=fsyms[i],edgecolor="black",facecolor=colors[i],zorder=2)
        lines.append(currLine)
    nstrs = ["Node "+str(i) for i in range(len(points))]
    ax.legend(lines,nstrs,loc="upper center",ncol=numberOfBlocks)
    plt.savefig(os.path.join(dir,"e_DownTriangle.pdf"))

def sweepPlotGen(*args):
    """Call this function to generate the swept process description plots."""
    #Generating points
    nsteps = int(blocksize/(2*boundaryPoints))
    usb,diasb,dsb = generateSets(*args)
    usbEdges = getEdgeBlocks(usb,args[3],args[2])
    dialen = int(np.shape(diasb)[1]/2-1)
    tempDias = tuple()
    for block in diasb:
        templist = list()
        for set in block[dialen:]:
            templist.append(set)
        tempDias += templist,
    diasbEdges = getEdgeBlocks(tempDias,args[3],args[2])
    dshape = np.shape(diasbEdges)
    for i in range(dshape[0]):
        for j in range(1):
            lth = len(diasbEdges[i][j])
            for k in range(dshape[2]):
                x,y = diasbEdges[i][j][k]
                if int(lth/2) > k:
                    diasbEdges[i][j][k] = x-boundaryPoints,y
                else:
                    diasbEdges[i][j][k] = x+boundaryPoints,y

    dsbEdges = getEdgeBlocks(dsb,args[3],args[2])
    #Generating figures
    UpTriangle(usb,usbEdges,args[0],args[2],args[4],nsteps)
    Comm1(usb,usbEdges,args[0],args[2],args[4],nsteps)
    Diamond(diasb,usb,usbEdges,args[0],args[2],args[4],nsteps)
    Comm2(diasb,usb,diasbEdges,args[0],args[2],args[4],nsteps)
    DownTriangle(dsb,usb,diasb,diasbEdges,args[0],args[2],args[4],nsteps)

if __name__ == "__main__":
    colors = ['#1b9e77','#d95f02','#7570b3','orange','dodgerblue','orange']
    symbols = cycle(['o','o','o','o'])
    blocksize = 16#limit to appropriately divisible cases
    boundaryPoints = 2
    numberOfBlocks = 3
    sweepPlotGen(colors,symbols,blocksize,boundaryPoints,numberOfBlocks)
