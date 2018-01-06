import os
import os.path as op
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import interpolate
import itertools

def cartProd(y):
    return pd.DataFrame(list(itertools.product(*y))).values

def formu(cList):
    xo = cList[:-1]
    form = xo[:]
    for i, x in enumerate(xo):
        for xa in xo[i:]:
            if x == xa:
                form.append("I(" + xa + "**2)")
            else:
                form.append(x + ":" + xa)

    return cList[-1] + " ~ " + " + ".join(form)

a = pd.read_csv("tEulerS.csv")
ac = list(a.columns.values)

tp, gp = a.tpb.unique(), a.gpuA.unique()

nxlim = [a.groupby(ac[:2]).min()['nX'].max(), a.groupby(ac[:2]).max()['nX'].min()]
nx = np.logspace(*(np.log2(nxlim)), base=2, num=100)
it = list(itertools.product(tp, gp, nx))
frameC = []
dff = pd.DataFrame(index=nx)

## STATSMODELS METHOD

# fm = formu(ac)
# smod = smf.ols(formula=fm, data=a).fit()
# print(smod.summary())

# newpd = pd.DataFrame(it, columns=ac[:-1])
# newpd['time'] = smod.predict(newpd)

## INTERPOLANT METHOD

for k, g in a.groupby('tpb'):
    for kk, gg in g.groupby('gpuA'):
        dff = dff.assign(tpb=k, gpuA=kk)
        itn = interpolate.interp1d(gg.nX, gg.time, kind='cubic')
        frameC.append(dff.assign(time=itn(nx)))

newpda = pd.concat(frameC).reset_index()
newpd = newpda.rename(columns={'index': 'nX'})

## PLOTTING

ga = pd.DataFrame()
gb = pd.DataFrame()
ad={}

for i in range(len(tp)//4):
    f, ai = plt.subplots(2,2)
    ap = ai.ravel()
    for aa, t in zip(ap, tp[i::2]):
        ad[t] = aa

for k, g in newpd.groupby('tpb'):
    gg = g.pivot(*ac[1:]).T
    gg.plot(ax = ad[k], logy=True, logx=True, grid=True, title=k)
    gbest = gg.min(axis=1)
    gpbest = gg.idxmin(axis=1)
    ga[k] = gbest
    gb[k] = gpbest

gb.reset_index(drop=True, inplace=True)
gtot = ga.min(axis=1)
gtpb = ga.idxmin(axis=1)
baff = []
trutpb = list(gtpb)

for i, g in enumerate(trutpb):
    baff.append(gb.loc[i, g])

trubest = pd.DataFrame(np.array((trutpb, baff)).T, columns=ac[:2], index=nx)

trubest.plot(logx=True, grid=True)
f = plt.figure()
aax = gtot.plot(logx=True, grid=True, label="Best Run")
a.plot(*ac[-2:], c='gpuA', ax=aax, kind='scatter', logx=True, grid=True)
aax.set_xlim([round(nx[0]/1.5, -5), round(nx[-1]*1.05, -5)])
plt.title("Best interpolated run vs observation")

# plt.show()

##AGAIN
frameC = []
dff = pd.DataFrame(index=nx)
gpuas = np.linspace(0,20,100)

for k, g in newpd.groupby('tpb'):
    for kk, gg in g.groupby('nX'):
        dff = dff.assign(tpb=k, nX=kk)
        itn = interpolate.interp1d(gg.gpuA, gg.time, kind='cubic')
        frameC.append(dff.assign(time=itn(gpuas)))

newpda = pd.concat(frameC).reset_index()
newpd = newpda.rename(columns={'index': 'nX'})

## PLOTTING

# ga = pd.DataFrame()
# gb = pd.DataFrame()
# ad={}

# for i in range(len(tp)//4):
#     f, ai = plt.subplots(2,2)
#     ap = ai.ravel()
#     for aa, t in zip(ap, tp[i::2]):
#         ad[t] = aa

# for k, g in newpd.groupby('tpb'):
#     gbest = gg.min(axis=1)
#     gpbest = gg.idxmin(axis=1)
#     ga[k] = gbest
#     gb[k] = gpbest