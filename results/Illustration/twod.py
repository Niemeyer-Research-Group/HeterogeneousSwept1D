from iglob import *

plt.style.use('classic')
aniname = "papier"
ext = ".pdf"
plt.rcParams['hatch.color'] = 'k'

sz = 80
szbor = 2/3*sz
fc = 0
markerz = '.'
lww = 2

def savePlot(fh, n, ip=impath):
    plotfile = op.join(ip, aniname + "-" + str(n) + ext)
    fh.tight_layout()
    fh.savefig(plotfile, dpi=200, bbox_inches="tight")