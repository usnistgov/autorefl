from json import load
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from plot_entropy import combinedata
#from autorefl import *

tscale ='log'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['lines.markersize'] = 8

def speedup(avexp, avctrl):
    # unpack, throwing away first point (t = 0 typically for an experiment)
    tctrl, Hctrl, _, Hctrl_marg, _ = avctrl
    t, H, dH, H_marg, dH_marg = [a[1:] for a in avexp]
    ti = np.interp(H, Hctrl, tctrl)
    ti_marg = np.interp(H_marg, Hctrl_marg, tctrl)

    return t, ti / t, ti_marg / t

exps = glob.glob('eta0.[2-7,9]*')
exps.append('eta0.80_npoints1_repeats1_20220115T194944')
exps.sort()
expctrl = ['control_20220117T143247']
colors = ['C%i' % i for i in range(10)]

avctrl, rawctrl = combinedata(expctrl, controls=True)

fig, (axm, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(8, 10))

legend_elements = []

min_t_av = 1e3

for exppath, color in zip(exps, colors):
    explist = glob.glob(exppath + '/' + '*.pickle')
    print(explist)
    avdata, rawdata = combinedata(explist)
    t, rat, ratmarg = speedup(avdata, avctrl)
    axm.plot(t, ratmarg, 'o', alpha=0.4, color=color)
    axm.axvline(np.mean(ratmarg[t>min_t_av]), linestyle='--', color=color)
    ax.plot(t, rat, 'o', alpha=0.4, color=color)
    ax.axvline(np.mean(rat[t>min_t_av]), linestyle='--', color=color)
    legend_elements.append(Line2D([0],[0], color=color, marker='o', alpha=0.4, label=exppath))

ax.set_xlabel('Time (s)')
ax.set_ylabel(r'speedup')
axm.set_ylabel(r'speedup')
tscale = tscale if tscale in ['linear', 'log'] else 'log'
ax.set_xscale(tscale)
ax.tick_params(axis='x', direction='inout', which='both', top=True, bottom=True)
ax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
if tscale == 'log':
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                    numticks=100)
    ax.xaxis.set_minor_locator(locmin)

axm.legend(handles=legend_elements, loc=0, fontsize='smaller')

fig.tight_layout()
plt.show()
