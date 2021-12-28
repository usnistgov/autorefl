import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
#from autorefl import *

plotscale ='log'
combine_even = True

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['lines.markersize'] = 8

paths = sys.argv[1:]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'hspace': 0})
figvars = None

colors = [['C0', 'C1'], ['C2', 'C3'], ['C4', 'C5'], ['C6', 'C7']]

def plotter(ts, plotzip, axs, fit=True):
    for q, axnum, c, lbl in plotzip:
        q = np.array(q)
        if len(q):
            av = np.mean(q, axis=0)
            stdev = np.std(q, axis=0)
            axs[axnum].plot(ts, av, linewidth=4, alpha=0.7, color=c)
            axs[axnum].fill_between(ts, av - stdev, av + stdev, color=c, alpha=0.3)
            if fit:
                goodts = ts>1e3
                print('coef of %s: ' % lbl, np.polyfit(np.log(ts[goodts]), -np.mean(q, axis=0)[goodts], 1))

def format_plot(ax):

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t$ (s)')

    ax.set_xticks(10.**np.arange(0, 6))
    ax.set_xlim([10, 5e5])
    ax.tick_params(axis='x', direction='inout', which='both', top=True, bottom=True)
    ax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                    numticks=100)
    ax.xaxis.set_minor_locator(locmin)
    plt.draw()
    #plt.savefig('plots/asyn_' + model['name'] + '.pdf')
    #plt.show()

    return None

legend_elements = list()

if combine_even:
    ts_even = list()
    all_dHs_even = list()
    all_dHs_marg_even = list()
    all_parvars_even = list()


for path, cs in zip(paths, colors):
    all_ts = list()
    all_dHs = list()
    all_dHs_marg = list()
    all_parvars = list()
    all_types = list()
    av_dHs = list()
    av_dHs_marg = list()
    av_parvars = list()
    if not combine_even:
        av_dHs_even = list()
        av_dHs_marg_even = list()
        av_parvars_even = list()

    if path[-1] != '/':
        path += '/'
    flist = glob.glob(path + '*timevars*.txt')
    if len(flist) == 0:
        continue
    legend_elements.append(Line2D([0],[0], color=cs[0], marker='o', alpha=0.4, label='auto: alpha' + flist[0].split('-timevars')[0].split('alpha')[2]))
    if not combine_even:
        legend_elements.append(Line2D([0],[0], color=cs[1], marker='o', alpha=0.4, label='even: alpha' + flist[0].split('-timevars')[0].split('alpha')[2]))
    for f in flist:
        lbl = f.split('timevars')[1].split('.txt')[0]
        t, dHs, dHs_marg  = np.loadtxt(f, usecols=[0, 3, 4], unpack=True)
        parvars = np.loadtxt(f)[:, 7:]
        all_parvars.append(parvars)
        all_ts.append(t)
        all_dHs.append(-dHs)
        all_dHs_marg.append(-dHs_marg)
        if 'noselect' in lbl:
            iseven = True
            color = cs[1]
        else:
            iseven = False
            color = cs[0]
        all_types.append(iseven)
        print(f, 'npts=%i' % len(t))
        if figvars is None:
            npars = parvars.shape[1]
            nmax = int(np.ceil(np.sqrt(npars)))
            nmin = int(np.ceil(npars/nmax))
            figvars, axvars = plt.subplots(ncols=nmin, nrows=nmax, sharex=True, figsize=(10,8), gridspec_kw={'hspace': 0})
            for axvar in axvars.flatten()[npars:]:
                axvar.axis('off')
        if not iseven or not combine_even:
            ax[0].semilogx(t, -dHs_marg, 'o', label=lbl, color=color, alpha=0.4)
            ax[1].semilogx(t, -dHs, 'o', label=lbl, color=color, alpha=0.4)
#            for i, axvar in enumerate(axvars.flatten()[:npars]):
#                axvar.loglog(t, np.sqrt(parvars[:,i]), 'o', color=color, alpha=0.4)


    ts = np.array([t for ts in all_ts for t in ts])
    tsort = np.argsort(ts)
    ts = ts[tsort]
    n = 0
    neven = 0
    for t, dH, dH_marg, parvars, iseven in zip(all_ts, all_dHs, all_dHs_marg, all_parvars, all_types):
        if iseven:
            neven += 1
            if not combine_even:
                av_dHs_even.append(np.interp(ts, t, dH))
                av_dHs_marg_even.append(np.interp(ts, t, dH_marg))
                av_parvars_item = list()
                for i, parvar in enumerate(parvars.T):
                    av_parvars_item.append(np.interp(ts, t, parvar))
                av_parvars_even.append(av_parvars_item)
            else:
                ts_even.append(t)
                all_dHs_even.append(dH)
                all_dHs_marg_even.append(dH_marg)
                all_parvars_even.append(parvars)
        else:
            n += 1
            av_dHs.append(np.interp(ts, t, dH))
            av_dHs_marg.append(np.interp(ts, t, dH_marg))
            av_parvars_item = list()
            for i, parvar in enumerate(parvars.T):
                av_parvars_item.append(np.interp(ts, t, parvar))
            av_parvars.append(av_parvars_item)

    plotzip = zip([av_dHs, av_dHs_even, av_dHs_marg, av_dHs_marg_even], [1, 1, 0, 0], [cs[0], cs[1], cs[0], cs[1]], ['av_dHs', 'av_dHs_even', 'av_dHs_marg', 'av_dHs_marg_even']) \
            if not combine_even else zip([av_dHs, av_dHs_marg], [1, 0], [cs[0], cs[0]], ['av_dHs', 'av_dHs_marg'])
    plotter(ts, plotzip, ax)

    av_parvars = np.array(av_parvars)
    plotzip = zip([np.sqrt(av_parvars[:,i,:]) for i in range(npars)], list(range(npars)), [cs[0]] * npars, ['par%i' % i for i in range(npars)])
    plotter(ts, plotzip, axvars.flatten(), fit=False)
    if not combine_even:
        av_parvars_even = np.array(av_parvars_even)
        plotzip = zip([np.sqrt(av_parvars_even[:,i,:]) for i in range(npars)], list(range(npars)), [cs[1]] * npars, ['par%i_even' % i for i in range(npars)])
        plotter(ts, plotzip, axvars.flatten(), fit=False)


if combine_even:
    evencolor = '0.1'
    ts = np.array([t for ts in ts_even for t in ts])
    tsort = np.argsort(ts)
    ts = ts[tsort]
    legend_elements.append(Line2D([0],[0], color=evencolor, marker='o', alpha=0.4, label='averaged uniform'))
    av_dHs_even = list()
    av_dHs_marg_even = list()
    av_parvars_even = list()
    for t, dH, dH_marg, parvars in zip(ts_even, all_dHs_even, all_dHs_marg_even, all_parvars_even):
        av_dHs_even.append(np.interp(ts, t, dH))
        av_dHs_marg_even.append(np.interp(ts, t, dH_marg))
        av_parvars_item = list()
        for i, parvar in enumerate(parvars.T):
            av_parvars_item.append(np.interp(ts, t, parvar))
        av_parvars_even.append(av_parvars_item)
    
    plotzip = zip([av_dHs_even, av_dHs_marg_even], [1, 0], [evencolor] * 2, ['av_dHs_even', 'av_dHs_marg_even'])
    plotter(ts, plotzip, ax)

    if len(av_parvars_even):
        av_parvars_even = np.array(av_parvars_even)
        #print(av_parvars_even.shape)
        plotzip = zip([np.sqrt(av_parvars_even[:,i,:]) for i in range(npars)], list(range(npars)), [evencolor] * npars, ['par%i' % i for i in range(npars)])
        plotter(ts, plotzip, axvars.flatten(), fit=False)

for axvar in axvars.flatten()[:npars]:
    axvar.set_yscale('log')
    axvar.set_xscale(plotscale)
    format_plot(axvar)
    for item in ([axvar.title, axvar.xaxis.label, axvar.yaxis.label] +
                axvar.get_xticklabels() + axvar.get_yticklabels()):
        item.set_fontsize(12)    



ax[0].set_xlabel('Time (s)')
ax[1].set_xlabel('Time (s)')
ax[0].set_ylabel('Marginal entropy change')
ax[1].set_ylabel('Total entropy change')
ax[0].set_xscale(plotscale)
ax[0].legend(handles=legend_elements, loc=0, fontsize='smaller')
ax[1].legend(handles=legend_elements, loc=0, fontsize='smaller')

fig.tight_layout()
figvars.tight_layout()

plt.show()
