"""Tools for analyzing simulated AutoRefl experiments"""

import numpy as np
import copy
from typing import Union, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.gridspec import GridSpec

Color = Union[Tuple[float, float, float] , str]

import datatools as ar
from simexp import SimReflExperiment, SimReflExperimentControl, ExperimentStep, data_tuple


def get_steps_time(steps: List[ExperimentStep], control: bool = False) -> np.ndarray:
    """Finds the total time associated a list of steps, depending on whether or not 
        it was a control experiment

    Args:
        steps (List[ExperimentStep]): list of steps for which to calculate timing
        control (bool, optional): True if a control experiment. Defaults to False.

    Returns:
        np.ndarray: vector of total times
    """

    if not control:
        allt = np.cumsum([step.meastime() + step.movetime() for step in steps])
    else:
        # assume all movement was done only once
        allt = np.cumsum([step.meastime() for step in steps]) + np.array([step.movetime() for step in steps])

    return allt

def load_entropy(steps: List[ExperimentStep], control: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the total and marginalized entropy associated with a list of steps

    Args:
        steps (List[ExperimentStep]): list of steps for calculation
        control (bool, optional): True if a control experiment. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: time, entropy, marginalized entropy vectors
    """

    allt = get_steps_time(steps, control)
    allH = [step.dH for step in steps]
    allH_marg = [step.dH_marg for step in steps]

    return allt, allH, allH_marg

def get_parameter_variance(steps: List[ExperimentStep], control: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the variance in each parameter at each of a list of steps

    Args:
        steps (List[ExperimentStep]): list of steps for calculation
        control (bool, optional): True if a control experiment. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: time vector, array of parameter variances in time
    """
    allt = get_steps_time(steps, control)
    allvars = np.array([np.var(step.draw.points, axis=0) for step in steps]).T

    return allt, allvars

def plot_qprofiles(Qth: np.ndarray,
                   qprofs: np.ndarray,
                   logps: np.ndarray,
                   data: Union[data_tuple, None] = None,
                   ax: Union[Axis, None] = None,
                   exclude_from: int = 0,
                   power: int = 4) -> Tuple[Figure, Tuple[Axis, Axis, Axis]]:
    """Plot data and distribution of profiles in Q

    Args:
        Qth (np.ndarray): Q vector, length Nq
        qprofs (np.ndarray): Q profile array, size Np x Nq
        logps (np.ndarray): array of log likelihood functions associated with each array, size Np
        data (Union[data_tuple, None], optional): Tuple of data (see DataPoint). Defaults to None.
        ax (Union[Axis, None], optional): axis on which to generate plot. Defaults to None; in this
            case a new axis is created in the figure.
        exclude_from (int, optional): exclude data before this index. Defaults to 0.
        power (int, optional): Q scaling of profile (R x Q^power). Defaults to 4.

    Returns:
        Tuple[Figure, Tuple[Axis, Axis, Axis]]: figure, axis, axis=None, axis=None
    """
    #Qs, Rs, dRs = problem.fitness.probe.Q[exclude_from:], problem.fitness.probe.R[exclude_from:], problem.fitness.probe.dR[exclude_from:]
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    else:
        fig = ax.figure.canvas

    if data is not None:
        _, _, _, _, Rs, dRs, Qs, _ = ar.compile_data_N(Qth, *data)
        #print('plot_qprofiles: ', len(Qs), Qs)
        if len(Qs) > 0:
            ax.errorbar(Qs[exclude_from:], (Rs*Qs**power)[exclude_from:], (dRs*Qs**power)[exclude_from:], fmt='o', color='k', markersize=10, alpha=0.4, capsize=8, linewidth=3, zorder=100)

    cmin, cmax = np.median(logps) + 2 * np.std(logps) * np.array([-1,1])
    colornorm = colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = cm.ScalarMappable(norm=colornorm, cmap=cm.jet)

    for qp, logp in zip(qprofs, logps):
        ax.plot(Qth, qp*Qth**power, '-', alpha=0.3, color=cmap.to_rgba(logp))

    ax.set_yscale('log')
    if 0:
        ax2 = ax.inset_axes([1.1, 0, 0.08, 1], transform=ax.transAxes)
        plt.colorbar(cmap, ax=ax, cax=ax2)
        ax2.tick_params(axis='y', right=True, left=True, labelright=False, labelleft=True)
        ax3 = ax2.inset_axes([1,0,2,1], transform=ax2.transAxes)
        h = np.histogram(logps, bins=int(round(len(logps)/10)), range=[cmin, cmax])
        ax3.plot(h[0], 0.5 * (h[1][1:] + h[1][:-1]), linewidth=3)
        ax3.fill_betweenx(0.5 * (h[1][1:] + h[1][:-1]), h[0], alpha=0.4)
        xlims = ax3.get_xlim()
        ax3.set_xlim([0, max(xlims)])
        ylims = ax2.get_ylim()
        ax3.set_ylim(ylims)
        ax3.tick_params(axis='y', labelleft=False)
        ax3.set_ylabel('log likelihood')
        ax3.set_xlabel('N')
        ax.set_xlabel(r'$Q_z$ (' + u'\u212b' + r'$^{-1}$)')
        ax.set_ylabel(r'$R \times Q_z^%i$ (' % power + u'\u212b' + r'$^{-4}$)')
    else:
        ax2 = None
        ax3 = None

    return fig, (ax, ax2, ax3)

def parameter_error_plot(exp: SimReflExperiment,
                         ctrl: Union[SimReflExperimentControl, None] = None,
                         fig: Union[Figure, None] = None,
                         tscale: str = 'log',
                         yscale: str = 'log',
                         color: Union[Color, None] = None) -> Tuple[Figure, List[Axis]]:
    """Plot time course of all the errors (std dev) of the parameters in the simulated experiment

    Args:
        exp (SimReflExperiment): experiment simulation
        ctrl (Union[SimReflExperimentControl, None], optional): control experiment. Defaults to None.
        fig (Union[Figure, None], optional): figure. Defaults to None, in which case a new figure
            is created and returned
        tscale (str, optional): Time scaling, 'linear' or 'log'. Defaults to 'log'.
        yscale (str, optional): Data scaling, 'linear' or 'log'. Defaults to 'log'.
        color (Union[Color, None], optional): color. Defaults to None.

    Returns:
        Tuple[Figure, List[Axis]]: figure, list of axes (one for each parameter)
    """

    import matplotlib.ticker

    npars = exp.npars
    labels = exp.problem.labels()

    # set up figure
    nmax = int(np.ceil(np.sqrt(npars)))
    nmin = int(np.ceil(npars/nmax))
    if fig is None:
        fig, axvars = plt.subplots(ncols=nmin, nrows=nmax, sharex=True, figsize=(4+2*nmin,4+nmax), gridspec_kw={'hspace': 0})
        # remove any extra axes, but only for a new figure!
        for axvar in axvars.flatten()[npars:]:
            axvar.axis('off')
        newplot = True
    else:
        axvars = np.array(fig.get_axes())
        axtitles = [ax.get_title() for ax in axvars]
        axsort = [axtitles.index(label) for label in labels]
        axvars = axvars[axsort]
        newplot = False

    # plot simulated data
    allt, allvars = get_parameter_variance(exp.steps[:-1])
    for var, ax, label in zip(allvars, axvars.flatten(), labels):
        y = np.sqrt(var)
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        ax.plot(allt, y, 'o', alpha=0.4, color=color)
        if newplot:
            #ax.set_ylabel(r'$\sigma$')
            ax.set_title(label, y=0.5, x=1.05, va='center', ha='left', rotation=-90, fontsize='smaller')

            # Format log-scaled axes
            if tscale == 'log':
                ax.set_xscale('log')
                ax.set_xticks(10.**np.arange(np.floor(np.log10(allt[1])), np.ceil(np.log10(allt[-1])) + 1))
                ax.tick_params(axis='x', direction='inout', which='both', top=True, bottom=True)
                ax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
                locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                    numticks=200)
                ax.xaxis.set_minor_locator(locmin)
            
            if yscale == 'log':
                ax.set_yscale('log')
                ax.set_yticks(10.**np.arange(np.floor(np.log10(min(y))), np.ceil(np.log10(max(y)))))
                ax.tick_params(axis='y', direction='inout', which='both', left=True, right=True)
                ax.tick_params(axis='y', which='major', labelleft=True, labelright=False)
                locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                    numticks=200)
                ax.yaxis.set_minor_locator(locmin)
        else:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

    # plot control if present
    if ctrl is not None:
        ctrlt, ctrlvars = get_parameter_variance(ctrl.steps)
        for var, ax in zip(ctrlvars, axvars.flatten()):
            ax.plot(ctrlt, np.sqrt(var), '-', color='0.1')

    if newplot:
        # set x axis label only on bottom row
        for ax in axvars.flatten()[(npars - nmax):npars]:
            ax.set_xlabel(r'$t$ (s)')

        # turn off x axis tick labels for all but the bottom row
        for ax in axvars.flatten()[:(npars - nmax)]:
            ax.tick_params(axis='x', labelbottom=False, labeltop=False)

        # must call tight_layout before drawing boxes
        fig.tight_layout()

        # draw boxes around selected parameters
        if (exp.sel is not None):
            for ax in axvars.flatten()[exp.sel]:
                # from https://stackoverflow.com/questions/62375119/is-it-possible-to-add-border-or-frame-around-individual-subplots-in-matplotlib
                bbox = ax.axes.get_window_extent(fig.canvas.get_renderer())
                x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
                # slightly increase the very tight bounds:
                xpad = 0.0 * width
                ypad = 0.0 * height
                fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor='red', linewidth=3, fill=False, alpha=0.5))

    return fig, axvars

def snapshot(exp: SimReflExperiment,
             stepnumber: int,
             fig: Union[Figure, None] = None,
             power: int = 4,
             tscale: str = 'log') -> Tuple[Figure, Tuple[List[Axis], List[Axis], Axis, Axis]]:
    """Plot snapshot of the simulation at a particular step number.

    Args:
        exp (SimReflExperiment): simulated experiment
        stepnumber (int): index of step number to plot
        fig (Union[Figure, None], optional): figure for the plot. Defaults to None, in which case a
            new figure is created
        power (int, optional): scaling of R(Q), i.e. R(Q) * Q^power. Defaults to 4.
        tscale (str, optional): Time scaling, 'linear' or 'log'. Defaults to 'log'.

    Returns:
        Tuple[Figure, Tuple[List[Axis], List[Axis], Axis, Axis]]: figure, top axes, bottom axes,
            entropy axis, marginalized entropy axis
    """

    allt, allH, allH_marg = load_entropy(exp.steps[:-1])

    if fig is None:
        fig = plt.figure(figsize=(8 + 4 * exp.nmodels, 8))
    gsright = GridSpec(2, exp.nmodels + 1, hspace=0, wspace=0.4)
    gsleft = GridSpec(2, exp.nmodels + 1, hspace=0.2, wspace=0)
    j = stepnumber
    step = exp.steps[j]

    steptimes = [sum([step.meastime(modelnum=i) for step in exp.steps[:(j+1)]]) for i in range(exp.nmodels)]
    movetimes = [sum([step.movetime(modelnum=i) for step in exp.steps[:(j+1)]]) for i in range(exp.nmodels)]

    axtopright = fig.add_subplot(gsright[0,-1])
    axbotright = fig.add_subplot(gsright[1,-1], sharex=axtopright)
    axtopright.plot(allt, allH_marg, 'o-')
    axbotright.plot(allt, allH, 'o-')
    if (j + 1) < len(exp.steps):
        axtopright.plot(allt[j], allH_marg[j], 'o', markersize=15, color='red', alpha=0.4)
        axbotright.plot(allt[j], allH[j], 'o', markersize=15, color='red', alpha=0.4)
    axbotright.set_xlabel('Time (s)')
    axbotright.set_ylabel(r'$\Delta H_{total}$ (nats)')
    axtopright.set_ylabel(r'$\Delta H_{marg}$ (nats)')
    tscale = tscale if tscale in ['linear', 'log'] else 'log'
    axbotright.set_xscale(tscale)
    if tscale == 'linear':
        axbotright.set_xlim([0, min(max(allt[min(2, len(allt) - 1)], 3 * allt[j]), max(allt))])

    axtops = [fig.add_subplot(gsleft[0, i]) for i in range(exp.nmodels)]
    axbots = [fig.add_subplot(gsleft[1, i]) for i in range(exp.nmodels)]

    #print(np.array(step.qprofs).shape, step.draw.logp.shape)
    foms = step.foms if step.foms is not None else [np.full_like(np.array(x), np.nan) for x in exp.x]
    qprofs = step.qprofs if step.qprofs is not None else [np.full_like(np.array(measQ), np.nan) for measQ in exp.measQ]
    for i, (measQ, qprof, x, fom, axtop, axbot) in enumerate(zip(exp.measQ, qprofs, exp.x, foms, axtops, axbots)):
        plotpoints = [pt for step in exp.steps[:(j+1)] if step.use for pt in step.points if pt.model == i]
        #print(*[[getattr(pt, attr) for pt in plotpoints] for attr in exp.attr_list])
        #idata = [[getattr(pt, attr) for pt in plotpoints] for attr in exp.attr_list]
        idata = [[val for pt in plotpoints for val in getattr(pt, attr)] for attr in exp.attr_list]
        plot_qprofiles(copy.copy(measQ), qprof, step.draw.logp, data=idata, ax=axtop, power=power)
        axtop.set_title(f'meas t = {steptimes[i]:0.0f} s\nmove t = {movetimes[i]:0.0f} s', fontsize='larger')
        axbot.plot(x, fom, linewidth=3, color='C0')
        if (j + 1) < len(exp.steps):
            newpoints = [pt for pt in exp.steps[j+1].points if ((pt.model == i) & (pt.merit is not None))]
            for newpt in newpoints:
                axbot.plot(newpt.x, newpt.merit, 'o', alpha=0.5, markersize=12, color='C1')
        ##axbot.set_xlabel(axtop.get_xlabel())
        ##axbot.set_ylabel('figure of merit')

    all_top_ylims = [axtop.get_ylim() for axtop in axtops]
    new_top_ylims = [min([ylim[0] for ylim in all_top_ylims]), max([ylim[1] for ylim in all_top_ylims])]
    all_bot_ylims = [axbot.get_ylim() for axbot in axbots]
    new_bot_ylims = [min([ylim[0] for ylim in all_bot_ylims]), max([ylim[1] for ylim in all_bot_ylims])]

    for axtop, axbot in zip(axtops, axbots):
        #axtop.sharex(axbot)
        axtop.sharey(axtops[0])
        axbot.sharey(axbots[0])
        axbot.set_xlabel(exp.instrument.xlabel)
        axtop.set_xlabel(r'$Q_z$ (' + u'\u212b' + r'$^{-1}$)')
        axtop.tick_params(labelleft=False, labelbottom=True, top=True, bottom=True, left=True, right=True, direction='in')
        axbot.tick_params(labelleft=False, top=True, bottom=True, left=True, right=True, direction='in')

    axtops[0].set_ylim(new_top_ylims)
    axbots[0].set_ylim(new_bot_ylims)
    rlabel = 'R' if power == 0 else r'$R \times Q_z^%i$ (' % power + u'\u212b' + r'$^{-%i}$)' % power 
    axtops[0].set_ylabel(rlabel)
    axbots[0].set_ylabel('figure of merit')
    axtops[0].tick_params(labelleft=True)
    axbots[0].tick_params(labelleft=True)

    fig.suptitle(f'measurement time = {sum(steptimes):0.0f} s\nmovement time = {sum(movetimes):0.0f} s', fontsize='larger', fontweight='bold')

    return fig, (axtops, axbots, axtopright, axbotright)

def makemovie(exp: SimReflExperiment,
              outfilename: str,
              expctrl: Union[SimReflExperimentControl, None] = None,
              fps: int = 1,
              fmt: str = 'gif',
              power: int = 4,
              tscale: str = 'log') -> None:
    """ Makes a GIF or MP4 movie from a SimReflExperiment object"""

    fig = plt.figure(figsize=(8 + 4 * exp.nmodels, 8))

    frames = list()

    for j in range(len(exp.steps[0:-1])):

        fig, (_, _, axtopright, axbotright) = snapshot(exp, j, fig=fig, power=power, tscale=tscale)

        if expctrl is not None:
            allt, allH, allH_marg = load_entropy(expctrl.steps, control=True)
            axtopright.plot(allt, allH_marg, 'o-', color='0.1')
            axbotright.plot(allt, allH, 'o-', color='0.1')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        axbotright.set_xscale('linear')
        fig.clf()

    if fmt == 'gif':
        import imageio
        imageio.mimsave(outfilename + '.' + fmt, frames, fps=fps)
    elif fmt == 'mp4':
        import skvideo.io
        skvideo.io.vwrite(outfilename + '.' + fmt, frames, outputdict={'-r': '%0.1f' % fps, '-crf': '20', '-profile:v': 'baseline', '-level': '3.0', '-pix_fmt': 'yuv420p'})
