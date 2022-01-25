#%%
import datetime
from multiprocessing.sharedctypes import Value
import numpy as np
import copy
import os
from bumps.cli import load_model
import matplotlib.pyplot as plt
from simexp import SimReflExperiment, makemovie
import instrument
import argparse
#import imageio
#import skvideo.io

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 1.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 16

parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--npoints', type=int)
parser.add_argument('--nrepeats', type=int)
parser.add_argument('--maxtime', type=float)
parser.add_argument('--penalty', type=float)
parser.add_argument('--burn', type=int)
parser.add_argument('--steps', type=int)
parser.add_argument('--resume', type=str)
parser.add_argument('--instrument', type=str)
args = parser.parse_args()

# define fit options dictionary
fit_options = {'burn': 1000, 'pop': 8, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

if __name__ == '__main__':

    # plot
    movieyn = True

    #%%
    eta = 0.8 if args.eta is None else args.eta
    npoints = 1 if args.npoints is None else args.npoints
    maxtime = 21.6e3 if args.maxtime is None else args.maxtime
    nrepeats = 1 if args.nrepeats is None else args.nrepeats
    penalty = 1.0 if args.penalty is None else args.penalty
    fit_options['alpha'] = 0.001 if args.alpha is None else args.alpha
    fit_options['burn'] = 1000 if args.burn is None else args.burn
    fit_options['steps'] = 500 if args.steps is None else args.steps

    if args.resume is None:

        if (args.instrument == 'MAGIK') | (args.instrument is None):
            instr = instrument.MAGIK()
        elif args.instrument == 'CANDOR':
            instr = instrument.CANDOR()
        else:
            raise ValueError('instrument must be MAGIK or CANDOR')

        fprefix = '%s_eta%0.2f_npoints%i_repeats%i' % (instr.name, eta, npoints, nrepeats)

        # define file name and create results directory based on timestamp
        fn = copy.copy(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
        pathname = fprefix + '_' + fn
        os.mkdir(pathname)
        #fn = './' + fn + fsuffix + '/' + fn

        #%%
        # define calculation probe from model file (runs calculation for all models in model file)
        modelfile = 'ssblm.py'
        bestpars = 'ssblm.par'

        model = load_model(modelfile)
        bestp = np.array([float(line.split(' ')[-1]) for line in open(bestpars, 'r').readlines()])

        # calculation vector
        measQ = np.linspace(0.008, 0.25, 201)

        # parameters to use for marginalization
        sel = np.array([10, 11, 12, 13, 14])

        for kk in range(nrepeats):
            exp = SimReflExperiment(model, measQ, instrument=instr, eta=eta, fit_options=fit_options, oversampling=11, bestpars=bestp, select_pars=sel, meas_bkg=[3e-6, 3e-5], switch_penalty=penalty)
            exp.switch_time_penalty = 0.0 # takes 5 minutes to switch models
            exp.add_initial_step()
            total_t = 0.0
            k = 0
            while total_t < maxtime:
                total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
                print('Rep: %i, Step: %i, Total time so far: %0.1f' % (kk, k, total_t))
                exp.fit_step()
                #exp.instrument.x = None # to turn off movement penalty
                exp.take_step()
                exp.save(pathname + '/exp%i.pickle' % kk)
                k += 1

            try:
                makemovie(exp, pathname + '/exp%i' % kk, fmt='gif', fps=3)
            except RuntimeError:
                plt.switch_backend('agg')
                makemovie(exp, pathname + '/exp%i' % kk, fmt='gif', fps=3)

    else:
        
        pathname = os.path.dirname(args.resume)
        filename = os.path.basename(args.resume)
        basename = filename.split('.pickle')[0]
        print('Resuming %s from %s...' % (basename, pathname))
        exp = SimReflExperiment.load(args.resume)

        total_t = sum([step.meastime() + step.movetime() for step in exp.steps[:-1]])
        k = len(exp.steps) - 1
        while total_t < maxtime:
            total_t += exp.steps[-1].meastime() + exp.steps[-1].movetime()
            print('Resumed, Step: %i, Total time so far: %0.1f' % (k, total_t))
            exp.fit_step()
            #exp.instrument.x = None # to turn off movement penalty
            exp.take_step()
            exp.save(pathname + '/' + basename + '_resume.pickle')
            k += 1

        try:
            makemovie(exp, pathname + '/' + basename + '_resume', fmt='gif', fps=3)
        except RuntimeError:
            plt.switch_backend('agg')
            makemovie(exp, pathname + '/' + basename + '_resume', fmt='gif', fps=3)




