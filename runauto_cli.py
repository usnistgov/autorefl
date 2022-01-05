#%%
import datetime
import numpy as np
import copy
import os
import time
from bumps.cli import load_model, load_best
from bumps.fitters import ConsoleMonitor, _fill_defaults, StepMonitor
from refl1d.names import FitProblem
import matplotlib.pyplot as plt
from bumps.mapper import MPMapper
from autorefl import *
import argparse
import imageio
#import skvideo.io

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 1.5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 16

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float)
parser.add_argument('--npoints', type=int)
parser.add_argument('--nrepeats', type=int)
parser.add_argument('--maxtime', type=float)
parser.add_argument('--burn', type=int)
parser.add_argument('--steps', type=int)
args = parser.parse_args()

# define fit options dictionary
fit_options = {'burn': 1000, 'pop': 8, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

# plot
movieyn = True


#%%
alpha = 0.5 if args.alpha is None else args.alpha
npoints = 1 if args.npoints is None else args.npoints
maxtime = 21.6e3 if args.maxtime is None else args.maxtime
nrepeats = 10 if args.nrepeats is None else args.nrepeats
fit_options['burn'] = 1000 if args.burn is None else args.burn
fit_options['steps'] = 500 if args.steps is None else args.steps

fsuffix = '_alpha%0.2f_npoints%i_repeats%i' % (alpha, npoints, nrepeats)


# define file name and create results directory based on timestamp
fn = copy.copy(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
os.mkdir(fn + fsuffix)
fn = './' + fn + fsuffix + '/' + fn
print(fn)

#%%
# define calculation probe from model file (runs calculation for all models in model file)
modelfile = 'ssblm.py'
bestpars = 'ssblm.par'

model = load_model(modelfile)
models = [model] if hasattr(model, 'fitness') else list(model.models)
nmodels = len(models)
npars = len(model.getp())
oversampling = 11

# make dedicated calculation model for "ground truth" reflectivity
calcmodel = load_model(modelfile)
load_best(calcmodel, bestpars)
calcmodels = [calcmodel] if hasattr(calcmodel, 'fitness') else list(calcmodel.models)

# make a copy of the model for dynamic updating
# define parameter scales.
newmodels = [m.fitness for m in models]
par_scale = np.diff(model.bounds(), axis=0)
calcprobes = [m.fitness for m in calcmodels]

# add in background
bkg = 1e-6
for c in calcprobes:
    c.probe.background.value = bkg

# set Q range
minQ = 0.008
maxQ = 0.25

# TODO: simulate real data with real background levels!

#%%
def run_cycle(fitnesslist, measQ, newQs, datalist, use_entropy=True, restart_pop=None, outfid=None):
    
    for fitness, data in zip(fitnesslist, datalist):
        mT, mdT, mL, mdL, mR, mdR, mQ, mdQ = compile_data_N(measQ, *data)

        fitness.probe._set_TLR(mT, mdT, mL, mdL, mR, mdR, dQ=None)
        fitness.probe.oversample(oversampling)
        fitness.update()

    newproblem = FitProblem(fitnesslist)
    newproblem.model_reset()
    newproblem.chisq_str()
    mapper = MPMapper.start_mapper(newproblem, None, cpus=0)
    if outfid is not None:
        monitor = StepMonitor(newproblem, outfid)
    else:
        monitor = ConsoleMonitor(newproblem)
    fitter = DreamFitPlus(newproblem)
    options=_fill_defaults(fit_options, fitter.settings)
    result = fitter.solve(mapper=mapper, monitors=[monitor], initial_population=restart_pop, **options)

    _, chains, _ = fitter.state.chains()
    new_pop = chains[-1, :, :]

    fitter.state.keep_best()
    fitter.state.mark_outliers()
    d = fitter.state.draw(thin=fit_options['steps']*5)
    best_logp = fitter.state.best()[1]
    newproblem.setp(fitter.state.best()[0])
    final_chisq = newproblem.chisq_str()

    if use_entropy:
        qprofs, qbkgs = calc_qprofiles(newproblem, d.points, [measQ] * len(fitnesslist), oversampling)
        #plot_qprofiles(newproblem, measQ, qprof, d.logp)

        foms, meas_times = calc_foms(newproblem, d.points, measQ, qprofs, qbkgs, eta=alpha, select_pars=sel)

        newQ, meas_time_Q, newfoms = select_new_points(measQ, foms, meas_times, npoints=npoints, switch_penalty=None)
    else:
        newQ = newQs
        meas_time_Q = meas_time
        foms = None
        qprofs = None
        qbkgs = None
        newfoms = None

    return newQ, meas_time_Q, new_pop, best_logp, final_chisq, d, newfoms, qprofs, qbkgs, foms

#%%

# parameters to use for marginalization
sel = np.array([10, 11, 12, 13, 14])

# calc initial entropy
Hproblem = FitProblem(newmodels)
Hs0 = calc_init_entropy(Hproblem)
Hs0_marg = calc_init_entropy(Hproblem, select_pars=sel)
print('initial entropy, marginalized: %f, %f' % (Hs0, Hs0_marg))


# open a reporter file (note write, not append). Use time stamp to avoid overwriting.
fid = open(fn + fsuffix + '_auto_report.txt', 'w')

# define global list of entropies and marginalized entropies
all_t = list()
all_Hs = list()
all_Hs_marg = list()
all_best_logps = list()
all_median_logps = list()
all_foms = list()

# generate initial data set
# Note: following line should work, but doesn't. Each model needs more data points than fitting parameters, apparently.
nQs = [((npars + 1) // nmodels) + 1 if i < ((npars + 1) % nmodels) else ((npars + 1) // nmodels) for i in range(nmodels)]
#nQs = [npars + 1] * nmodels
newQs = list()
new_meastimes = list()
for nQ in nQs:
    newQs.append(np.linspace(minQ, maxQ, nQ, endpoint=True))
    new_meastimes.append(np.zeros_like(newQs[-1]))

if movieyn:
    # initialize frames list for movies
    fig = plt.figure(figsize=(12, 8))
    axtop, axbot = fig.subplots(2, nmodels, sharex=True, sharey='row', gridspec_kw={'hspace': 0})
    fig.tight_layout()
    fig.canvas.draw()
    fig.clf()
# Run nrepeats versions of the calculation cycle.
for kk in range(nrepeats):
    fid.write('Auto iteration %i\n' % kk)
    iteration_start = time.time()
    k=0
    t=[]
    total_t = 0.0
    Hs = list()
    Hs_marg = list()
    meastimes = [[] for _ in range(nmodels)]
    best_logps = list()
    median_logps = list()
    iter_foms = [[] for _ in range(nmodels)]
    varXs = list()
    last_foms = [None for _ in range(nmodels)]
    restart_pop=None

    frames = list()        

    # define space of possible measurements. Same space is used for all models
    measQ = np.linspace(minQ, maxQ, 201, endpoint=True)

    # generate initial data
    for m in newmodels:
        m.probe.oversample(oversampling)
        m.update()
    newproblem = FitProblem(newmodels)
    initpts = generate(newproblem, init='lhs', pop=fit_options['pop'], use_point=False)
    iqprof, iqbkg = calc_qprofiles(newproblem, initpts, newQs)
    data = create_init_data_N(newQs, iqprof, dRoR=10.0)
#    print(newQs, meastimes, new_meastimes, data, calcprobes)
    while total_t < maxtime:
        starttime = time.time()
        print('Now on cycle %i' % k, flush=True)
        print('Total time so far: %f' % total_t)
        fid.write('Cycle: %i\n' % k)
        for i, (newQ, meas_time, new_meastime, idata, calcprobe) in enumerate(zip(newQs, meastimes, new_meastimes, data, calcprobes)):
            fid.write(('Q[%i]: ' % i) + ', '.join(map(str, newQ)) + '\n')
            fid.write(('Time[%i]: ' % i) + ', '.join(map(str, new_meastime)) + '\n')
            print(('newQ[%i]: ' % i), newQ)
            print(('Time[%i]: ' % i) + ', '.join(map(str, new_meastime)))
            meas_time.append(new_meastime)
            if (k > 0) & (len(newQ) > 0):
                newvars = gen_new_variables(newQ)
                calcR = calc_expected_R(calcprobe, *newvars, oversampling=oversampling)
                data[i] = append_data_N(newQ, calcR, new_meastime, bkg, *idata)

        total_t = sum([sum(m) for mt in meastimes for m in mt])
        t.append(total_t)

        newQs, new_meastimes, restart_pop, best_logp, final_chisq, d, newfoms, qprofs, qbkgs, foms = run_cycle(newmodels, measQ, newQs, data, use_entropy=True, restart_pop=restart_pop, outfid=None)

        # impose a minimum 10 s measurement time
        new_meastimes = [np.maximum(m, 10.0 * np.ones_like(m)) for m in new_meastimes]

        Hs.append(calc_entropy(d.points / par_scale))
        Hs_marg.append(calc_entropy(d.points / par_scale, select_pars=sel))
        best_logps.append(best_logp)
        median_logps.append(np.median(d.logp))
        for iter_fom, fom in zip(iter_foms, foms):
            iter_fom.append(fom)
        varXs.append(np.std(d.points, axis=0)**2)
        fid.write('final chisq: %s\n' % final_chisq)
        fid.write('entropy: %f\nmarginalized entropy: %f\nbest_logp: %f\nmedian_logp: %f\n' % (Hs[-1], Hs_marg[-1], best_logps[-1], median_logps[-1]))
        fid.write('calculation wall time (s): %f\n' % (time.time() - starttime))
        print('entropy, marginalized entropy: %f, %f' % (Hs[-1], Hs_marg[-1]))
        fid.flush()

        if movieyn:
            axtops, axbots = fig.subplots(2, nmodels, sharex=True, sharey='row', squeeze=False, gridspec_kw={'hspace': 0})
            total_meastime = 0.0
            for newQ, meas_time, idata, qprof, cur_fom, last_fom, newfom, axtop, axbot in zip(newQs, meastimes, data, qprofs, foms, last_foms, newfoms, axtops, axbots):
                plotdata = tuple([v[len(meas_time[0]):] for v in idata]) if k > 0 else None
#                print(meas_time, idata[0])
                plot_qprofiles(measQ, qprof, d.logp, data=plotdata, ax=axtop)
                meas_time_sum = np.sum([sum(mt) for mt in meas_time])
                axtop.set_title('t = %0.0f s' % meas_time_sum, fontsize='larger')
                total_meastime += meas_time_sum
                if last_fom is not None:
                    axbot.semilogy(measQ, last_fom, linewidth=2, alpha=0.4, color='C0')
                last_fom = cur_fom
                axbot.semilogy(measQ, cur_fom, linewidth=3, color='C0')
                axbot.plot(newQ, newfom, 'o', alpha=0.5, markersize=12, color='C1')
                axbot.set_xlabel(axtop.get_xlabel())
                axbot.set_ylabel('figure of merit')
            fig.suptitle('t = %0.0f s' % total_meastime, fontweight='bold', fontsize='larger')
            fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            fig.clf()

        k += 1
        #np.savetxt(fn + fsuffix + '-fom%i.%i.txt' % (kk, k), np.vstack((measQ, np.array(foms))))

    if movieyn:
    #    skvideo.io.vwrite(fn + fsuffix + '-movie%i.mp4' % kk, frames, outputdict={'-r': '2', '-crf': '20', '-profile:v': 'baseline', '-level': '3.0', '-pix_fmt': 'yuv420p'})
        imageio.mimsave(fn + fsuffix + '-movie%i.gif' % kk, frames, fps=1)

    all_t.append(t)
    all_Hs.append(Hs)
    all_Hs_marg.append(Hs_marg)
    all_best_logps.append(best_logps)
    all_median_logps.append(median_logps)
    all_foms.append(iter_foms)

    fid.write('iteration wall time (s): %f\n' % (time.time() - iteration_start))
    fid.flush()

    np.savetxt(fn + fsuffix + '-timevars%i.txt' % kk, np.vstack((t, Hs, Hs_marg, Hs0-np.array(Hs), Hs0_marg - np.array(Hs_marg), best_logps, median_logps, np.array(varXs).T)).T, header='t, Hs, Hs_marg, dHs, dHs_marg, best_logps, median_logps, nxparameter_variances')
#    for mnum, (idata, meas_time, fom) in enumerate(zip(data, meastimes, iter_foms)):
#        idata = np.array(idata)
#        cycletime = np.array([(i, val) for i, m in enumerate(meas_time) for val in m])
    #    print(data.shape, cycletime.shape, data[0,:][None,:].shape, data[:,0][:,None].shape)
#        np.savetxt(fn + fsuffix + '-data%i_m%i.txt' % (kk, mnum), np.hstack((a2q(idata[0,:], idata[2,:])[:,None], cycletime, idata.T)), header='Q, cycle, meas_time, T, dT, L, dL, Nspecular, Nbackground, Nincident')
#        np.savetxt(fn + fsuffix + '-foms%i_m%i.txt' % (kk, mnum), np.vstack((measQ, np.array(fom))).T, header='Q, figure_of_merit')

#np.savetxt(fn + fsuffix + '-timevars_all.txt', np.vstack((t, Hs, Hs_marg, best_logps, median_logps)), header='t, Hs, Hs_marg, best_logps, median_logps')
fid.close()

#%%
all_t2 = list()
all_Hs2 = list()
all_Hs2_marg = list()
all_best_logps2 = list()
all_median_logps2 = list()

newQs = models[modelnum].fitness.probe.Q
measQ2 = newQs
meastimeweight = newQs**2
meastimeweight /= np.sum(meastimeweight)
data2 = ([], [], [], [], [], [], [])
restart_pop2=None

fid = open(fn + fsuffix+ '_auto_report_noselect.txt', 'w')

# Run nrepeats versions of the calculation cycle.
for kk in range(nrepeats):
    fid.write('Auto iteration %i\n' % kk)
    iteration_start = time.time()
    Hs2 = list()
    Hs2_marg = list()
    best_logps2 = list()
    median_logps2 = list()
    varXs2 = list()
    t = all_t[kk]
    meastimes2 = np.diff(t) # ends up being one smaller than the initial, because t=0 point is meaningless here
    restart_pop=None
    all_meastimes2 = list()

    # generate data
    data2 = ([], [], [], [], [], [], [])

    # reset model
    newmodel = copy.deepcopy(models[modelnum].fitness)

    for k in range(len(meastimes2)):
        starttime = time.time()
        meas_time = meastimes2[k]*meastimeweight
        print('Now on cycle %i of %i' % (k, len(meastimes2)-1), flush=True)
        fid.write('Cycle: %i\n' % k)
        fid.write('Q: ' + ', '.join(map(str, newQs)) + '\n')
        fid.write('Time: ' + ', '.join(map(str, meas_time)) + '\n')
        print('newQ: ', newQs)
        all_meastimes2.append(meas_time)

        newvars = gen_new_variables(newQs)
        calcR = calc_expected_R(calcprobe, *newvars, oversampling=oversampling)
        data2 = append_data_N(newQs, calcR, meas_time, bkg, *data2)

        _, _, restart_pop2, best_logp, final_chisq, d, _, _, _, _ = run_cycle(newmodel, measQ, newQs, data2, use_entropy=False, restart_pop=restart_pop2, outfid=None)


        Hs2.append(calc_entropy(d.points/par_scale))
        Hs2_marg.append(calc_entropy(d.points/par_scale, select_pars=sel))
        best_logps2.append(best_logp)
        median_logps2.append(np.median(d.logp))
        varXs2.append(np.std(d.points, axis=0)**2)
        fid.write('final chisq: %s\n' % final_chisq)        
        fid.write('entropy: %f\nmarginalized entropy: %f\nbest_logp: %f\nmedian_logp: %f\n' % (Hs2[-1], Hs2_marg[-1], best_logps2[-1], median_logps2[-1]))
        fid.write('calculation wall time (s): %f\n' % (time.time() - starttime))
        fid.flush()

    all_t2.append(t[1:])
    all_Hs2.append(Hs2)
    all_Hs2_marg.append(Hs2_marg)
    all_best_logps2.append(best_logps2)
    all_median_logps2.append(median_logps2)

    fid.write('iteration wall time (s): %f\n' % (time.time() - iteration_start))
    fid.flush()

    data2 = np.array(data2)
    cycletime = np.array([(i, val) for i, m in enumerate(all_meastimes2) for val in m])
#    print(data2.shape, cycletime.shape, data2[0,:][None,:].shape, data2[:,0][:,None].shape)
    np.savetxt(fn + fsuffix + '-data%i_noselect.txt' % kk, np.hstack((a2q(data2[0,:], data2[2,:])[:,None], cycletime, data2.T)), header='Q, cycle, meas_time, T, dT, L, dL, Nspecular, Nbackground, Nincident')
    np.savetxt(fn + fsuffix + '-timevars%i_noselect.txt' % kk, np.vstack((t[1:], Hs2, Hs2_marg, Hs0-np.array(Hs2), Hs0_marg - np.array(Hs2_marg), best_logps2, median_logps2, np.array(varXs2).T)).T, header='t, Hs, Hs_marg, dHs, dHs_marg, best_logps, median_logps, nxparameter_variances')

#np.savetxt(fn + fsuffix + '-timevars_all_noselect.txt', np.vstack((all_t2, Hs2, Hs2_marg, best_logps2, median_logps2)), header='t, Hs, Hs_marg, best_logps, median_logps')
fid.write('Done!')
fid.close()
print('Done!')
