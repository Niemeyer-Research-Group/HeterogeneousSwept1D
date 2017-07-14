'''
    RUN THIS FIRST.  GETS AND INSTALLS OR INSERTS DEPENDENCIES.
    THANKS TO NLOHMANN FOR MODERN C++ JSON HEADER.
'''

import os
import os.path as op
import urllib

# DEPENDENCY 1: json.hpp

targetfile = "json.hpp"
webfile = "https://github.com/nlohmann/json/blob/develop/src/json.hpp"
thispath = op.abspath(op.dirname(__file__))
local = op.join(op.join(op.join(thispath,"src"), "utilities"), targetfile)
os.chdir(thispath)

if not op.isfile(local):
    jfile = urllib.urlretrieve(webfile, targetfile)
    os.rename(op.join(thispath, targetfile), local)
    print("json.hpp has been downloaded and is now in the utilities folder\n")
else:
    print("json.hpp was already in the utilities folder\n")