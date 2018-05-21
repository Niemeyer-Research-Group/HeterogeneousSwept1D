
import sys

import timing_help as th

# run with ./compy.py equation scheme tpb gpuA nX

myar = [float(k) for k in sys.argv[3:]]

argstr = "tpb {:.2f} gpuA {:.2f} nX {:.2f}".format(*myar)
rat = th.compareRuns(sys.argv[1], sys.argv[2], argstr)
print(rat)
