from json import load
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from simexp import SimReflExperiment, SimReflExperimentControl, load_entropy
#from autorefl import *

tscale ='log'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['lines.markersize'] = 8

exps = []
expctrls = []
colors = ['C%i' % i for i in range(10)]

def combinedata(explist, controls=False):

    allts = []
    allHs = []
    allHs_marg = []

    ctype = SimReflExperiment if not controls else SimReflExperimentControl

    for expname in explist:
        exp = ctype.load(expname)
        allt, allH, allH_marg = load_entropy(exp.steps[:-1]) if not controls else load_entropy(exp.steps)
        allts = allts.append(allt)
        allHs = allHs.append(allH)
        allHs_marg = allHs_marg.append(allH_marg)

    t = np.unique(np.array(allt).flatten())
    allHi = list()
    allHi_marg = list()
    for allt, allH, allH_marg in zip(allts, allHs, allHs_marg):
        allHi.append(np.interp(t, allt, allH))
        allHi_marg.append(np.interp(t, allt, allH_marg))

    allHi = np.array(allHi)
    allHi_marg = np.array(allHi_marg)
    meanHi, stdHi = np.mean(allHi, axis=0), np.std(allHi, axis=0)
    meanHi_marg, stdHi_marg = np.mean(allHi_marg, axis=0), np.std(allHi_marg, axis=0)

    return (t, meanHi, stdHi, meanHi_marg, stdHi_marg), (allts, allHs, allHs_marg)

def plot_data(av, raw, axm, ax, color=None):

    for allt, allH, allH_marg in zip(raw):
        axm.plot(allt, allH_marg, 'o', alpha=0.4, color=color)
        ax.plot(allt, allH, 'o', alpha=0.4, color=color)

    t, meanHi, stdHi, meanHi_marg, stdHi_marg = av
    axm.plot(t, meanHi_marg, linewidth=4, alpha=0.7, color=color)
    axm.fill_between(t, meanHi_marg - stdHi_marg, meanHi_marg + stdHi_marg, color=color, alpha=0.3)
    ax.plot(t, meanHi, linewidth=4, alpha=0.7, color=color)
    ax.fill_between(t, meanHi_marg - stdHi, meanHi + stdHi, color=color, alpha=0.3)


fig, (axm, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})

for exppath, color in zip(exps, colors):
    explist = glob.glob(exppath + '/' + '*.pickle')
    avdata, rawdata = combinedata(explist)
    plot_data(avdata, rawdata, axm, ax, color=color)

for exppath in zip(expctrls):
    explist = glob.glob(exppath + '/' + '*.pickle')
    avdata, rawdata = combinedata(explist)
    plot_data(avdata, rawdata, axm, ax, color='0.1')

ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\Delta H_{total}$ (nats)')
axm.set_ylabel(r'$\Delta H_{marg}$ (nats)')
tscale = tscale if tscale in ['linear', 'log'] else 'log'
ax.set_xscale(tscale)
ax.tick_params(axis='x', direction='inout', which='both', top=True, bottom=True)
ax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
if tscale == 'log':
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                    numticks=100)
    ax.xaxis.set_minor_locator(locmin)

