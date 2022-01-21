#%%
import datetime
import numpy as np
import copy
import os
from bumps.cli import load_model
import matplotlib.pyplot as plt
from simexp import SimReflExperimentControl, makemovie
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

    if (args.instrument == 'MAGIK') | (args.instrument is None):
        instr = instrument.MAGIK()
    elif args.instrument == 'CANDOR':
        instr = instrument.CANDOR()
    else:
        raise ValueError('instrument must be MAGIK or CANDOR')

    fprefix = instr.name + '_control'

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
    measQ = [m.fitness.probe.Q for m in model.models]  

    # parameters to use for marginalization
    sel = np.array([10, 11, 12, 13, 14])

    # meastimes
    meastimes = np.diff(np.insert(np.logspace(1, np.log10(maxtime), 31, endpoint=True), 0, 0))

    for kk in range(nrepeats):
        exp = SimReflExperimentControl(model, measQ, instrument=instr, model_weights=[2., 1.], eta=eta, fit_options=fit_options, oversampling=11, bestpars=bestp, select_pars=sel, meas_bkg=[3e-6, 3e-5])
        total_t = 0.0
        k = 0
        for meastime in meastimes:
            exp.take_step(meastime)
            total_t += exp.steps[-1].meastime()
            print('Rep: %i, Step: %i, Total time so far: %0.1f' % (kk, k, total_t))
            exp.fit_step()
            exp.save(pathname + '/expctrl%i.pickle' % kk)
            k += 1

#        try:
#            makemovie(exp, pathname + '/expctrl%i' % kk, fmt='gif')
#        except RuntimeError:
#            plt.switch_backend('agg')
#            makemovie(exp, pathname + '/expctrl%i' % kk, fmt='gif')


