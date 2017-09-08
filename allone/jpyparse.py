
import json as j
import pandas as p

def parseJson(fpath):
    fi = open(fpath)
    fr = fi.read()
    jdict = j.loads(fr)
    for jk in jdict.keys():
        jdict[jk] = p.DataFrame(jdict[jk]) 
        
    return jdict