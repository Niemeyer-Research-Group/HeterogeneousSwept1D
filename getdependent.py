'''
    RUN THIS FIRST.  GETS AND INSTALLS OR INSERTS DEPENDENCIES.
    THANKS TO NLOHMANN FOR MODERN C++ JSON HEADER.
'''

import os
import os.path as op
import shutil
import subprocess as sp
import shlex


# DEPENDENCY 1: json.hpp

thispath = op.abspath(op.dirname(__file__))
os.chdir(thispath)
targetfile = "json.hpp"
targetpath = op.join(op.join(op.join(thispath, "src"), "utilities"), targetfile)

if not op.isfile(targetpath):
    repo = 'https://github.com/nlohmann/json.git'
    callrep = "git clone " + repo
    tokens = shlex.split(callrep)
    proc = sp.Popen(tokens)
    sp.Popen.wait(proc)
    jpath = op.join(thispath, "json")
    fpath = op.join(op.join(jpath, "src"), targetfile)
    os.rename(fpath, targetpath)
    print("json.hpp has been downloaded and is now in the utilities folder")
    shutil.rmtree(jpath)

else:
    print("json.hpp was already in the utilities folder")