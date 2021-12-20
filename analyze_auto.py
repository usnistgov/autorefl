import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from autorefl import *

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 16
plt.rcParams['lines.markersize'] = 8

paths = sys.argv[1:]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'hspace': 0})

colors = [['C0', 'C1'], ['C2', 'C3'], ['C4', 'C5'], ['C6', 'C7']]
legend_elements = list()

for path, cs in zip(paths, colors):
    all_ts = list()
    all_dHs = list()
    all_dHs_marg = list()
    all_types = list()

    if path[-1] != '/':
        path += '/'
    flist = glob.glob(path + '*timevars*.txt')
    legend_elements.append(Line2D([0],[0], color=cs[0], marker='o', alpha=0.4, label='auto: alpha' + flist[0].split('-timevars')[0].split('alpha')[2]))
    legend_elements.append(Line2D([0],[0], color=cs[1], marker='o', alpha=0.4, label='even: alpha' + flist[0].split('-timevars')[0].split('alpha')[2]))
    for f in flist:
        lbl = f.split('timevars')[1].split('.txt')[0]
        t, dHs, dHs_marg  = np.loadtxt(f, usecols=[0, 3, 4], unpack=True)
        all_ts.append(t)
        all_dHs.append(dHs)
        all_dHs_marg.append(dHs_marg)
        iseven = True if 'noselect' in lbl else False
        all_types.append(iseven)
        print(f, 'npts=%i' % len(t))
        color = cs[1] if 'noselect' in lbl else cs[0]
        ax[0].semilogx(t, -dHs_marg, 'o', label=lbl, color=color, alpha=0.4)
        ax[1].semilogx(t, -dHs, 'o', label=lbl, color=color, alpha=0.4)

    ts = np.array([t for ts in all_ts for t in ts])
    tsort = np.argsort(ts)
    ts = ts[tsort]
    n = 0
    neven = 0
    av_dHs = list()
    av_dHs_even = list()
    av_dHs_marg = list()
    av_dHs_marg_even = list()
    for t, dH, dH_marg, iseven in zip(all_ts, all_dHs, all_dHs_marg, all_types):
        if iseven:
            neven += 1
            av_dHs_even.append(np.interp(ts, t, dH))
            av_dHs_marg_even.append(np.interp(ts, t, dH_marg))
        else:
            n += 1
            av_dHs.append(np.interp(ts, t, dH))
            av_dHs_marg.append(np.interp(ts, t, dH_marg))

    av_dHs = np.array(av_dHs)
    av_dHs_even = np.array(av_dHs_even)
    av_dHs_marg = np.array(av_dHs_marg)
    av_dHs_marg_even = np.array(av_dHs_marg_even)

    for q, axnum, c, lbl in zip([av_dHs, av_dHs_even, av_dHs_marg, av_dHs_marg_even], [1, 1, 0, 0], [cs[0], cs[1], cs[0], cs[1]], ['av_dHs', 'av_dHs_even', 'av_dHs_marg', 'av_dHs_marg_even']):
        if len(q):
            av = -np.mean(q, axis=0)
            stdev = np.std(q, axis=0)
            ax[axnum].plot(ts, av, linewidth=4, alpha=0.7, color=c)
            ax[axnum].fill_between(ts, av - stdev, av + stdev, color=c, alpha=0.3)

            goodts = ts>1e3

            print('coef of %s: ' % lbl, np.polyfit(np.log(ts[goodts]), -np.mean(q, axis=0)[goodts], 1))


ax[0].set_xlabel('Time (s)')
ax[1].set_xlabel('Time (s)')
ax[0].set_ylabel('Marginal entropy change')
ax[1].set_ylabel('Total entropy change')
ax[0].set_xscale('log')
ax[0].legend(handles=legend_elements, loc=0, fontsize='smaller')
ax[1].legend(handles=legend_elements, loc=0, fontsize='smaller')

fig.tight_layout()

plt.show()
