
import runatom as ra

rsltfolder = "eulerResult"
root = "./bin/Euler"
algo = ["Atomic", "Array"]

tpb = [32*k for k in range(2, 24, 2)]
nx  = [2**k for k in range(11, 21)]

times = " 1e-7 .01 20"

for ia in algo:
    for ib in range(2):
        rt = root + ia
        for tp in tpb:
            for x in nx:
                strnia = rt + "{:d} {:d}".format(x, tp) + times
                
                exstr = shlex.split(strnia)
			    proc = sp.Popen(exstr)
			    sp.Popen.wait(proc)
			    time.sleep(3)

lst = [k for k in os.listdir(rsltfolder) if k.endswith('.csv')]

# take the csvs and merge them into a dataframe then toss that dataframe to the function and viola.

