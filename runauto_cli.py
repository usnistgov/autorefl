#%%
import datetime
import numpy as np
import copy
import os
from bumps.cli import load_model
import matplotlib.pyplot as plt
from simexp import SimReflExperiment, SimReflExperimentControl, makemovie
import instrument
import argparse
#import imageio
#import skvideo.io

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
parser.add_argument('--eta', type=float, default=0.8)
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--npoints', type=int, default=1)
parser.add_argument('--nrepeats', type=int, default=1)
parser.add_argument('--maxtime', type=float, default=21.6e3)
parser.add_argument('--penalty', type=float, default=1.0)
parser.add_argument('--timepenalty', type=float, default=0.0)
parser.add_argument('--burn', type=int, default=1000)
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--pop', type=int, default=8)
parser.add_argument('--resume', type=str)
parser.add_argument('--instrument', type=str, default='MAGIK')
parser.add_argument('--oversampling', type=int, default=11)
args = parser.parse_args()

# define fit options dictionary
fit_keys = ['burn', 'pop', 'steps', 'init', 'alpha']
fit_options = dict([(k, getattr(args, k)) for k in fit_keys])
#fit_options = {'burn': 1000, 'pop': 8, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

if __name__ == '__main__':

    if args.resume is None:

        if (args.instrument == 'MAGIK') | (args.instrument is None):
            instr = instrument.MAGIK()
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

        #%%
        # define calculation probe from model file (runs calculation for all models in model file)
        modelfile = args.model
        model = load_model(modelfile)

        bestpars = args.pars
        bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()]) if bestpars is not None else None

        # measurement background
        meas_bkg = np.full(len(model.models), 1e-5) if args.meas_bkg is not None else args.meas_bkg

        # simulated experiment
        if not args.control:

            # calculation vector
            # TODO: figure out how to specify this in the cli
            measQ = np.linspace(0.008, 0.25, 201)

            for kk in range(args.nrepeats):
                exp = SimReflExperiment(model, measQ, instrument=args.instr, eta=args.eta, fit_options=fit_options, oversampling=args.oversampling, bestpars=bestp, select_pars=args.sel, meas_bkg=meas_bkg, switch_penalty=args.penalty, npoints=args.npoints)
                exp.switch_time_penalty = args.switch_time_penalty # takes time to switch models
                exp.add_initial_step()
                total_t = 0.0
                k = 0
                while total_t < args.maxtime:
                    total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
                    print('Rep: %i, Step: %i, Total time so far: %0.1f' % (kk, k, total_t))
                    exp.fit_step()
                    #exp.instrument.x = None # to turn off movement penalty
                    exp.take_step()
                    exp.save(pathname + '/exp%i.pickle' % kk)
                    k += 1

                if not args.nomovie:
                    try:
                        makemovie(exp, pathname + '/exp%i' % kk, fmt='gif', fps=3)
                    except RuntimeError:
                        plt.switch_backend('agg')
                        makemovie(exp, pathname + '/exp%i' % kk, fmt='gif', fps=3)

        # control measurement
        else:
            # calculation vector
            measQ = [m.fitness.probe.Q for m in model.models]  

            # meastimes
            meastimes = np.diff(np.insert(np.logspace(1, np.log10(args.maxtime), args.nctrlpts, endpoint=True), 0, 0))

            # condition model weights
            if args.model_weights is None:
                model_weights = np.ones(len(model.models))
            else:
                assert len(args.model_weights) == len(model.models), 'model_weights must have one per model'
                model_weights = args.model_weights

            for kk in range(args.nrepeats):
                exp = SimReflExperimentControl(model, measQ, instrument=instr, model_weights=model_weights, eta=args.eta, fit_options=fit_options, oversampling=args.oversampling, bestpars=bestp, select_pars=args.sel, meas_bkg=meas_bkg)
                total_t = 0.0
                k = 0
                for meastime in meastimes:
                    exp.take_step(meastime)
                    total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
                    print('Rep: %i, Step: %i, Total time so far: %0.1f' % (kk, k, total_t))
                    exp.fit_step()
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
            exp.take_step()
            exp.save(pathname + '/' + basename + '_resume.pickle')
            k += 1

        if not args.nomovie:
            try:
                makemovie(exp, pathname + '/' + basename + '_resume', fmt='gif', fps=3)
            except RuntimeError:
                plt.switch_backend('agg')
                makemovie(exp, pathname + '/' + basename + '_resume', fmt='gif', fps=3)
