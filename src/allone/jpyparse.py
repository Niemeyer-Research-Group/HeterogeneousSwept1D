"""
    Parse the json and make some output.
"""
import json as j
import pandas as p
import matplotlib.pyplot as plt

def parseJ(fpath):
    fi = open(fpath)
    fr = fi.read()
    jdict = j.loads(fr)
    return jdict

def parseJdf(fb):
    if isinstance(fb,str):
        jdict = parseJ(fb)
    elif isinstance(fb, dict):
        jdict=fb
    else:
        print("What is this?")
        return 0
    
    for jk in jdict.keys():
        jdict[jk] = p.DataFrame(jdict[jk]) 
        
    return jdict

def normJdf(fpath, xlist):
    jdict = parseJ(fpath)
    jd = dict()
    for jk in jdict.keys():

        jd[jk] = dict()
        for jkk in jdict[jk].keys():
            jknt, idx = min([(abs(int(jkk)-x), i) for i, x in enumerate(xlist)])
            
            jkn = xlist[idx]
            print(jknt, idx, jkk, jkn)
            
            if jkn not in jd[jk].keys():
                jd[jk][jkn] = dict()
                
                
            jd[jk][jkn].update(jdict[jk][jkk])
        
            
    return jd

def nicePlot(df, ptitle, ltitle):
    df.plot(grid=True, logy=True, title=ptitle)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, title=ltitle, borderaxespad=0.)
    #Saveit as pdf?    
    return True
                
if __name__ == "__main__":
    f = "timer.json"
    x0 = [2**k for k in range(11,21,2)]
    jd = normJdf(f, x0)
    dfj = parseJdf(jd)
    dfj.index = p.to_numeric(dfj.index)
    
                