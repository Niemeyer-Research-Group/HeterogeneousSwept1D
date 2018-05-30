import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import numpy as np
# from moviepy.video.io.bindings import mplfig_to_npimage
# import moviepy.editor as mpy

thispath = op.abspath(op.dirname(__file__))
ipath = op.dirname(thispath)
toppath = op.dirname(ipath)
pypath = op.join(toppath, "runtools")
impath = op.join(thispath, "images")

style = op.join(pypath, "swept.mplstyle")

