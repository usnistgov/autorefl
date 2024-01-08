""" Command line interface for running simulated reflectometry measurements

Example:
python -m autorefl.runauto_cli ssblm.py --pars ssblm_tosb0.par --name both_forecast_q025
    --sel 10 11 12 13 14 --eta 0.50 --meas_bkg 3e-6 3e-5 --maxtime 43.2e3 --burn 1000 --steps 500
    --pop 8 --instrument MAGIK --qmax 0.25 --qstep_max 0.0024

Required arguments:

model -- a Refl1D model script (.py)

Optional arguments:

--pars -- a file containing a parameter file (e.g. a Refl1D .par output file) containing ground
            truth values for each parameter

--name -- suffix to the simulation storage directory. Directory name has format:
            {instrument}_npoints{npoints}_nrepeats{nrepeats}_{datetime}_{name}

--control -- flags this as a control measurement

--nctrlpts -- number of control points to use (only used if --control is present)

--nomovie -- does not create a movie after simulation is complete

--sel -- list of parameter indices to use for figure of merit calculation

--meas_bkg -- measurement background (float). Single value or list, one for each model in the
                Refl1D script

--model_weights -- weights for the models (only used with --control). Does not have to be
                    normalized to unity

--eta -- confidence interval parameter; default 0.68. Ignored for controls.

--npoints -- number of points to simulate/measure at a time (uses forecasting)

--nrepeats -- number of times to repeat the simulation/measurement

--maxtime -- stopping criterion: maximum total time (measurement + movement)

--penalty -- penalty for switching models (simexp.switch_penalty)

--timepenalty -- time penalty for switching models (simexp.switch_time_penalty)

--entropy_method -- method used to calculate entropy. Default is mvn_fast; other options are
                    'mvn' or 'gmm'

--gmm_n_components -- if 'gmm' is the entropy method, number of components to use. '1' is equivalent
                        to 'mvn'. If not specified, default 5*sqrt(n_parameters) is used

--scale_samples -- scale the parameter values before doing figure of merit calculation. May improve
                    performance of gmm calculation. Largely untested.

--burn -- max number of burn steps for Refl1D fit

--steps -- number of steps for Refl1D fit

--pop -- population scale factor for Refl1D fit

--alpha -- convergence criterion for Refl1D fit

--init -- initialization for Refl1D fit. Default 'lhs'

--resume -- specify a previous fit to resume. Must give full path to pickle file. File name will
            be the same with _resume.pickle added.

--instrument -- which reflectometry instrument to use. Default 'MAGIK'. 'CANDOR' also supported

--oversampling -- oversampling level in Refl1D calculations

--qmin -- minimum Q value for Q measurement space; default 0.008

--qmax -- maximum Q value for Q measurement space; default 0.250

--qstep -- Q step size at minimum Q value; default 0.0005

--qstep_max -- Q step size at maximum Q value; if not provided, qstep value is used over
                the whole range, otherwise step size increases linearly from qstep
                to qstep_max

"""

import datetime
import numpy as np
import copy
import os
from bumps.cli import load_model
import matplotlib.pyplot as plt
from .simexp import SimReflExperiment, SimReflExperimentControl
from .analysis import makemovie
import autorefl.instrument as instrument
import argparse

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 1.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 16

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str)
parser.add_argument('--pars', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--control', action='store_true')
parser.add_argument('--nctrlpts', type=int, default=21)
parser.add_argument('--nomovie', action='store_true')
parser.add_argument('--sel', type=int, nargs='+')
parser.add_argument('--meas_bkg', type=float, nargs='+')
parser.add_argument('--model_weights', type=float, nargs='+')
parser.add_argument('--eta', type=float, default=0.68)
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--npoints', type=int, default=1)
parser.add_argument('--nrepeats', type=int, default=1)
parser.add_argument('--maxtime', type=float, default=21.6e3)
parser.add_argument('--penalty', type=float, default=1.0)
parser.add_argument('--timepenalty', type=float, default=0.0)
parser.add_argument('--entropy_method', type=str, default='mvn_fast')
parser.add_argument('--gmm_n_components', type=int)
parser.add_argument('--scale_samples', action='store_true')
parser.add_argument('--burn', type=int, default=1000)
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--pop', type=int, default=8)
parser.add_argument('--init', type=str, default='lhs')
parser.add_argument('--resume', type=str)
parser.add_argument('--instrument', type=str, default='MAGIK')
parser.add_argument('--oversampling', type=int, default=11)
parser.add_argument('--qmin', type=float, default=0.008)
parser.add_argument('--qmax', type=float, default=0.25)
parser.add_argument('--qstep', type=float, default=0.0005)
parser.add_argument('--qstep_max', type=float, default=None)
args = parser.parse_args()

# define fit options dictionary
fit_keys = ['burn', 'pop', 'steps', 'init', 'alpha']
fit_options = dict([(k, getattr(args, k)) for k in fit_keys])
entropy_options = {'method': args.entropy_method, 'n_components': args.gmm_n_components, 'scale': args.scale_samples}
print(entropy_options)
#fit_options = {'burn': 1000, 'pop': 8, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

if __name__ == '__main__':

    if args.resume is None:

        if (args.instrument == 'MAGIK') | (args.instrument is None):
            instr = instrument.MAGIK()
#            instr._mon0 = 0.0
        elif args.instrument == 'CANDOR':
            instr = instrument.CANDOR()
        else:
            raise ValueError('instrument must be MAGIK or CANDOR')

        fprefix = '%s_eta%0.2f_npoints%i_repeats%i' % (instr.name, args.eta, args.npoints, args.nrepeats) \
                    if not args.control else instr.name + '_control'
        fsuffix = '' if args.name is None else args.name

        # define file name and create results directory based on timestamp
        fn = copy.copy(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        pathname = fprefix + '_' + fn + '_' + fsuffix
        os.mkdir(pathname)
        #fn = './' + fn + fsuffix + '/' + fn

        # define calculation probe from model file (runs calculation for all models in model file)
        modelfile = args.model
        model = load_model(modelfile)

        bestpars = args.pars
        bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()]) if bestpars is not None else None

        # measurement background
        meas_bkg = args.meas_bkg if args.meas_bkg is not None else np.full(len(list(model.models)), 1e-5)

        # condition selection array
        sel = np.array(args.sel) if args.sel is not None else None

        # set measQ
        qstep_max = args.qstep if args.qstep_max is None else args.qstep_max
        dq = np.linspace(args.qstep, qstep_max, int(np.ceil(2 * (args.qmax - args.qmin) / (qstep_max + args.qstep))))
        measQ = (args.qmin-args.qstep) + np.cumsum(dq)
        #measQ = [m.fitness.probe.Q for m in model.models]

        # simulated experiment
        if not args.control:

            for kk in range(args.nrepeats):
                exp = SimReflExperiment(model, measQ, instrument=instr, eta=args.eta, fit_options=fit_options, oversampling=args.oversampling, bestpars=bestp, select_pars=sel, meas_bkg=meas_bkg, switch_penalty=args.penalty, npoints=args.npoints, entropy_options=entropy_options)
                exp.switch_time_penalty = args.timepenalty # takes time to switch models

                # Generate appropriate x vectors depending on the instrument
                if args.instrument == 'MAGIK':
                    exp.x = exp.measQ
                elif args.instrument == 'CANDOR':
                    for i, measQ in enumerate(exp.measQ):
                        x = list()
                        overlap = 0.90
                        xrng = exp.instrument.qrange2xrange([min(measQ), max(measQ)])
                        x.append(xrng[0])
                        while x[-1] < xrng[1]:
                            curq = exp.instrument.x2q(x[-1])
                            curminq, curmaxq = np.min(curq), np.max(curq)
                            newrng = exp.instrument.qrange2xrange([curminq + (curmaxq - curminq) * (1 - overlap), max(measQ)])
                            x.append(newrng[0])
                        x[-1] = xrng[1]
                        x = np.array(x)
                        exp.x[i] = x

                # add the initial step
                exp.add_initial_step()

                # run the simulation until the maximum simulation time is reached
                total_t = 0.0
                k = 0
                while total_t < args.maxtime:
                    total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
                    print('Rep: %i, Step: %i, Total time so far: %0.1f' % (kk, k, total_t))
                    
                    # fit the previous step
                    exp.fit_step()

                    # to turn off movement penalty
                    #exp.instrument.x = None

                    # take the next step
                    exp.take_step(allow_repeat=False)

                    # save the simulation
                    exp.save(pathname + '/exp%i.pickle' % kk)
                    k += 1

                # make a movie
                if not args.nomovie:
                    try:
                        makemovie(exp, pathname + '/exp%i' % kk, fmt='gif', fps=3)
                    except RuntimeError:
                        plt.switch_backend('agg')
                        makemovie(exp, pathname + '/exp%i' % kk, fmt='gif', fps=3)

        # control measurement
        else:
            # generate the appropriate number of measurement times
            meastimes = np.diff(np.insert(np.logspace(1, np.log10(args.maxtime), args.nctrlpts, endpoint=True), 0, 0))

            # condition model weights
            if args.model_weights is None:
                model_weights = np.ones(len(list(model.models)))
            else:
                assert len(args.model_weights) == len(model.models), 'model_weights must have one per model'
                model_weights = args.model_weights

            for kk in range(args.nrepeats):
                exp = SimReflExperimentControl(model, measQ, instrument=instr, model_weights=model_weights, eta=args.eta, fit_options=fit_options, oversampling=args.oversampling, bestpars=bestp, select_pars=sel, meas_bkg=meas_bkg, entropy_options=entropy_options)

                # Generate appropriate x vectors depending on the instrument
                if args.instrument == 'CANDOR':
                    for i, measQ in enumerate(exp.measQ):
                        x = list()
                        overlap = 0.90
                        xrng = exp.instrument.qrange2xrange([min(measQ), max(measQ)])
                        x.append(xrng[0])
                        while x[-1] < xrng[1]:
                            curq = exp.instrument.x2q(x[-1])
                            curminq, curmaxq = np.min(curq), np.max(curq)
                            newrng = exp.instrument.qrange2xrange([curminq + (curmaxq - curminq) * (1 - overlap), max(measQ)])
                            x.append(newrng[0])
                        x[-1] = xrng[1]
                        x = np.array(x)
                        exp.x[i] = x

                    model_weights = np.array(model_weights) / np.sum(model_weights)

                    exp.meastimeweights = list()
                    for x, weight in zip(exp.x, model_weights):
                        exp.meastimeweights.append(weight * np.array(x)**2 / np.sum(np.array(x)**2))

                    print(exp.x, len(exp.x[0]))

                # run the simulation once for each measurement time
                total_t = 0.0
                k = 0
                for meastime in meastimes:

                    # take a step
                    exp.take_step(meastime)
                    total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
                    print('Rep: %i, Step: %i, Total time so far: %0.1f' % (kk, k, total_t))

                    # fit the previous step
                    exp.fit_step()

                    # save the result
                    exp.save(pathname + '/expctrl%i.pickle' % kk)
                    k += 1

    else:
        
        pathname = os.path.dirname(args.resume)
        filename = os.path.basename(args.resume)
        basename = filename.split('.pickle')[0]
        print('Resuming %s from %s...' % (basename, pathname))

        # resuming simulated experiment (can't resume control measurement)
        exp = SimReflExperiment.load(args.resume)

        total_t = sum([step.meastime() + step.movetime() for step in exp.steps[:-1]])
        k = len(exp.steps) - 1
        while total_t < args.maxtime:
            total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
            print('Resumed, Step: %i, Total time so far: %0.1f' % (k, total_t))
            exp.fit_step()
            #exp.instrument.x = None # to turn off movement penalty
            exp.take_step(allow_repeat=False)
            exp.save(pathname + '/' + basename + '_resume.pickle')
            k += 1

        if not args.nomovie:
            try:
                makemovie(exp, pathname + '/' + basename + '_resume', fmt='gif', fps=3)
            except RuntimeError:
                plt.switch_backend('agg')
                makemovie(exp, pathname + '/' + basename + '_resume', fmt='gif', fps=3)
