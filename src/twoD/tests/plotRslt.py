

import os
import os.path as op
import json
import matplotlib.pyplot as plt
import pandas as pd

fol = "testResult"
fi = os.listdir(fol)
fi = [op.join(fol, ft) for ft in fi]

for fn in fi:
    with open(fn, 'r') as thisf:
        fd = json.load(thisf)
    
    kf = list(fd.keys())[0]
    ext = fd[kf]
    ke = list(ext.keys())
    print(ke)
    print(ext[ke[0]].values() == ext[ke[1]].values()) 


# # dict_of_df = {k: pd.DataFrame(v) for k,v in dictionary.items()}
# df = pd.concat(dict_of_df, axis=1)


