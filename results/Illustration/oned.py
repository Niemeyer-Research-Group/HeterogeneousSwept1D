
from iglob import *

aniname = "single"
ext = ".pdf"

nodeColors = [".r", ".g", ".b", ".k"]

def savePlot(fh, n):
    plotfile = op.join(impath, aniname + "-" + str(n) + ext)
    fh.savefig(plotfile, dpi=200, bbox_inches="tight")

def upTriangle(regions, base, freq, framec, framei):


if __name__ == "__main__":
    pass


