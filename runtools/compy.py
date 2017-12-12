import timing_help as th
import sys


myar = [float(k) for k in sys.argv[3:]]
argstr = "tpb {:.2f} gpuA {:.2f} nX {:.2f}".format(*myar)
rat = th.compareRuns(sys.argv[1], sys.argv[2], argstr)
print(rat)