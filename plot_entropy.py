import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from bumps.plotutil import next_color
from simexp import SimReflExperiment, SimReflExperimentControl, load_entropy
import argparse

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 1.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 16

parser = argparse.ArgumentParser()

# list of experiment folders or pickle files to plot
parser.add_argument('experiments', type=str, nargs='+')

# only one control allowed for calculating speedup (for other controls, add them as experiments)
# results of wildcards for controls are automatically combined
parser.add_argument('--control', type=str)

# time scale to use in plot ('linear' or 'log')
parser.add_argument('--tscale', type=str, default='log', choices=['linear', 'log'])

# smallest time to use for comparing speedup
parser.add_argument('--min_time', type=float, default=1e3)

#
parser.add_argument('--labels', type=str, nargs='+')
parser.add_argument('--combine', action='store_true')
parser.add_argument('--savename', type=str, nargs='+', default=[])
args = parser.parse_args()

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['lines.markersize'] = 8

def combinedata(explist, controls=False):

    allts = []
    allHs = []
    allHs_marg = []

    ctype = SimReflExperiment if not controls else SimReflExperimentControl

    for expname in explist:
        exp = ctype.load(expname)
        print(exp.problem.chisq_str())
        allt, allH, allH_marg = load_entropy(exp.steps[:-1]) if not controls else load_entropy(exp.steps, control=True)
        allts.append(allt)
        allHs.append(allH)
        allHs_marg.append(allH_marg)

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

    for allt, allH, allH_marg in zip(*raw):
        axm.plot(allt, allH_marg, 'o', alpha=0.4, color=color)
        ax.plot(allt, allH, 'o', alpha=0.4, color=color)

    t, meanHi, stdHi, meanHi_marg, stdHi_marg = av
    axm.plot(t, meanHi_marg, linewidth=4, alpha=0.7, color=color)
    axm.fill_between(t, meanHi_marg - stdHi_marg, meanHi_marg + stdHi_marg, color=color, alpha=0.3)
    ax.plot(t, meanHi, linewidth=4, alpha=0.7, color=color)
    ax.fill_between(t, meanHi - stdHi, meanHi + stdHi, color=color, alpha=0.3)

def speedup(avexp, avctrl):
    # unpack, throwing away first point (t = 0 typically for an experiment)
    tctrl, Hctrl, _, Hctrl_marg, _ = avctrl
    t, H, dH, H_marg, dH_marg = [a[1:] for a in avexp]
    ti = np.interp(H, Hctrl, tctrl, left=np.nan, right=np.nan)
    ti_marg = np.interp(H_marg, Hctrl_marg, tctrl, left=np.nan, right=np.nan)
    rat = ti / t
    trat = t[~np.isnan(rat)]
    rat = rat[~np.isnan(rat)]
    rat_marg = ti_marg / t
    trat_marg = t[~np.isnan(rat_marg)]
    rat_marg = rat_marg[~np.isnan(rat_marg)]

    return trat, rat, trat_marg, rat_marg

def load_path(exppath):
    if not 'pickle' in exppath:
        explist = glob.glob(exppath + '/' + '*_resume.pickle')
        if not len(explist):
            explist = glob.glob(exppath + '/' + '*.pickle')
    else:
        explist = [exppath]
    
    return explist

def plot_entropy(tscale=None, min_time=None, control=None, experiments=[], labels=None, combine=False, savename=[]):
    # TODO: Add documentation!!
    
    tscale = tscale
    min_t_av = min_time

    if control is not None:
        fig, all_ax = plt.subplots(2, 2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(12, 10))
        axm, ax = all_ax[:,0]
        caxm, cax = all_ax[:,1]
    else:
        fig, all_ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(6, 10))
        axm, ax = all_ax

    legend_elements = []
    clegend_elements = []
    if labels is None:
        alllabels = [None for _ in range(len(experiments))]
    else:
        alllabels = labels

    if control is not None:
        explist = load_path(control)
        print(explist)
        avctrl, rawctrl = combinedata(explist, controls=True)
        plot_data(avctrl, rawctrl, axm, ax, color='0.1')
        legend_elements.append(Line2D([0],[0], color='0.1', marker='o', alpha=0.4, label='control'))

    for exppath, label in zip(experiments, alllabels):
        explists = load_path(exppath)
        explists = [explists] if combine else [[e] for e in explists]
        for explist in explists:
            print(explist)
            color = next_color()
            avdata, rawdata = combinedata(explist)
            plot_data(avdata, rawdata, axm, ax, color=color)
            legend_elements.append(Line2D([0],[0], color=color, marker='o', alpha=0.4, label=label))
            if control is not None:
                t, rat, tmarg, ratmarg = speedup(avdata, avctrl)
                caxm.plot(tmarg, ratmarg, 'o', alpha=0.4, color=color)
                caxm.axhline(np.mean(ratmarg[tmarg>min_t_av]), linestyle='--', color=color)
                cax.plot(t, rat, 'o', alpha=0.4, color=color)
                cax.axhline(np.mean(rat[t>min_t_av]), linestyle='--', color=color)
                clegend_elements.append(Line2D([0],[0], color=color, marker='o', alpha=0.4, label=label))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\Delta H_{total}$ (nats)')
    axm.set_ylabel(r'$\Delta H_{marg}$ (nats)')
    ax.set_xscale(tscale)
    ax.tick_params(axis='x', direction='inout', which='both', top=True, bottom=True)
    ax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
    if tscale == 'log':
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                        numticks=100)
        ax.xaxis.set_minor_locator(locmin)

    if control is not None:
        cax.set_xlabel('Time (s)')
        cax.set_ylabel(r'speedup')
        caxm.set_ylabel(r'speedup')
        cax.set_xscale(tscale)
        cax.tick_params(axis='x', direction='inout', which='both', top=True, bottom=True)
        cax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
        if tscale == 'log':
            cax.xaxis.set_minor_locator(locmin)
        if labels is not None:
            caxm.legend(handles=clegend_elements, loc=0, fontsize='smaller')

    if labels is not None:
        axm.legend(handles=legend_elements, loc=0, fontsize='smaller')

    fig.tight_layout()
    for name in savename:
        plt.savefig(name, dpi=300)

    return fig, all_ax

if __name__ == '__main__':

    fig, ax = plot_entropy(**args.__dict__)
    
    plt.show()
