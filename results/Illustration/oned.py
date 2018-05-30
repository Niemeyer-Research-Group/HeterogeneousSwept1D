
from iglob import *

plt.style.use(style)

aniname = "single"
ext = ".pdf"

sz = 80
szbor = 2/3*sz
fc = 0
markerz = '.'
lww = 2

def savePlot(fh, n, ip=impath):
    plotfile = op.join(ip, aniname + "-" + str(n) + ext)
    fh.tight_layout()
    fh.savefig(plotfile, dpi=200, bbox_inches="tight")

class SweptCycle(object):
    def __init__(self, regions, base, freq, nodeColors, figsz=(10,8), basef=3/2):
        self.f, self.ax = plt.subplots(1, 1, figsize=figsz)
        self.base       = base
        self.freq       = freq
        self.offs       = base//2 
        self.regions    = regions
        self.nodeColors = nodeColors

        self.ax.set_xlim(0, self.base*self.regions)
        self.ax.set_ylim(-1, basef*self.base)
        self.ax.set_xlabel("Spatial Point")
        self.ax.set_ylabel("Sub-Timestep")
        self.ax.set_title("Initial Triangle (UpTriangle)")
        self.framei = 1
        self.ts     = 1

    def upTriangle(self):
        fc = 0
        ts = 0 if self.ts == 1 else self.ts
        while(self.base-2*ts > 1):
            for r in range(self.regions):
                srt         = r*self.base
                tstrt       = ts + srt
                backstrt    = srt + self.base
                self.ax.scatter(tstrt-1, ts, s=szbor, marker=markerz, edgecolor='black', linewidth=lww, c=self.nodeColors[r])
                self.ax.scatter(tstrt, ts, s=szbor, marker=markerz, edgecolor='black',linewidth=lww, c=self.nodeColors[r])
                self.ax.scatter(backstrt-(ts+1), ts, s=szbor, marker=markerz, edgecolor='black', linewidth=lww, c=self.nodeColors[r])
                if r==0: print(tstrt, ts, self.base-ts)
                self.ax.scatter(backstrt-(ts), ts, s=szbor, marker=markerz, edgecolor='black', linewidth=lww, c=self.nodeColors[r])
                
                for b in range(1+tstrt,backstrt-(ts+1)):
                    self.ax.scatter(b, ts, s=sz, marker=markerz, edgecolor='black', linewidth=0, c=self.nodeColors[r])

            ts += 1       
            fc += 1
            if (fc > self.freq):
                savePlot(self.f, self.framei)
                self.framei += 1
                fc += self.freq

        for r in range(self.regions):
            srt     = r*self.base - 1
            tstrt   = srt + ts
            self.ax.scatter(tstrt, ts, s=szbor, marker=markerz, edgecolor='black',linewidth=lww, c=self.nodeColors[r])
            self.ax.scatter(tstrt+1, ts, s=szbor, marker=markerz, edgecolor='black', linewidth=lww, c=self.nodeColors[r])

    def downTriangle(self):
        fc = 0
        while(self.base-2*ts > 1):
            pass

    def diamond(self):
        pass

    def passSide(self):
        pass



if __name__ == "__main__":
    base=16
    nodeColors = ["r", "g", "b", "c"]
    if len(sys.argv) > 1:
        nregions = 10
        j=nregions//(len(nodeColors)+1)
        nc = []
        for k in range(len(nodeColors)):
            jo = j if k else 2*j
            nc += [nodeColors[k]]*jo

        scn = SweptCycle(nregions, base, 1000, nc, (12,2.5), basef=.7)        
        scn.upTriangle()
        scn.ax.plot([1*base-.5, 1*base-.5], [0, base/1.2], '-k', [3*base-.5, 3*base-.5],[0, base/1.2], '-k', linewidth=3)
        scn.ax.text(1.5*base, base/1.4, "GPU AREA", fontweight='bold')
        scn.f.tight_layout()

    else:    
        sc = SweptCycle(4, base, 1, nodeColors)
        sc.upTriangle()
        
        # savePlot(sc.f, 0)
    
    plt.show()


