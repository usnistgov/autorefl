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
parser.add_argument('--model', type=int)
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
modelnum = 0 if args.model is None else args.model

fsuffix = '_alpha%0.2f_npoints%i_repeats%i_m%i' % (alpha, npoints, nrepeats, modelnum)


# define file name and create results directory based on timestamp
fn = copy.copy(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
os.mkdir(fn + fsuffix)
fn = './' + fn + fsuffix + '/' + fn
print(fn)

#%%
# define calculation probe from model file
modelfile = 'ssblm.py'
bestpars = 'ssblm.par'

model = load_model(modelfile)
models = list(model.models)
oversampling = 11

# make a copy of the model for dynamic updating
newmodel = copy.deepcopy(models[modelnum].fitness)

# define parameter scale. Note that this will have to be done for each model
# separately if using independent models
par_scale = np.diff(models[modelnum].bounds(), axis=0)

# make dedicated calculation model for "ground truth" reflectivity
calcmodel = load_model(modelfile)
load_best(calcmodel, bestpars)
calcmodels = list(calcmodel.models)
calcprobe = copy.deepcopy(calcmodels[modelnum].fitness)

# simulates real data with high background level
#calcprobe.probe.background.value *= 10.0

# define background level
bkg = 1e-6
calcprobe.probe.background.value = bkg

#%%
def run_cycle(fitness, measQ, newQs, data, use_entropy=True, restart_pop=None, outfid=None):
    
    mT, mdT, mL, mdL, mR, mdR, mQ, mdQ = compile_data_N(measQ, *data)

    fitness.probe._set_TLR(mT, mdT, mL, mdL, mR, mdR, dQ=None)
    fitness.probe.oversample(oversampling)
    fitness.update()
    newproblem = FitProblem(fitness)
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
    d = fitter.state.draw(thin=fit_options['steps'])
    best_logp = fitter.state.best()[1]
    newproblem.setp(fitter.state.best()[0])
    final_chisq = newproblem.chisq_str()

    if use_entropy:
        qprof, qbkg = calc_qprofiles(newproblem, d.points, measQ, oversampling)
        #plot_qprofiles(newproblem, measQ, qprof, d.logp)

        newQ, newfoms, meas_time_Q, fom = select_new_points(newproblem, d.points, measQ, qprof, qbkg, alpha=alpha, npoints=npoints, select_pars=sel)
    else:
        newQ = newQs
        meas_time_Q = meas_time
        fom = None
        qprof = None
        qbkg = None
        newfoms = None

    return newQ, meas_time_Q, new_pop, best_logp, final_chisq, d, newfoms, qprof, qbkg, fom

#%%

# parameters to use for marginalization
sel = np.array([10, 11, 12, 13, 14])

# calc initial entropy
Hproblem = FitProblem(newmodel)
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

if movieyn:
    # initialize frames list for movies
    fig = plt.figure(figsize=(12, 8))
    axtop, axbot = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
    fig.tight_layout()
    fig.canvas.draw()
    fig.clf()
# Run nrepeats versions of the calculation cycle.
for kk in range(nrepeats):
    fid.write('Auto iteration %i\n' % kk)
    iteration_start = time.time()
    k=0
    t=[0]
    Hs = list()
    Hs_marg = list()
    meastimes = list()
    best_logps = list()
    median_logps = list()
    foms = list()
    varXs = list()
    last_fom = None
    restart_pop=None

    frames = list()        
    # generate initial data set
    newQs = np.linspace(0.008, 0.25, 17, endpoint=True)
    meas_time = np.zeros_like(newQs) # 10 seconds per point

    # define space of possible measurements.
    measQ = np.linspace(newQs[0], newQs[-1], 201)

    # reset model
    newmodel = copy.deepcopy(models[modelnum].fitness)

    # generate initial data
    newmodel.probe.oversample(oversampling)
    newmodel.update()
    newproblem = FitProblem(newmodel)
    initpts = generate(newproblem, init='lhs', pop=fit_options['pop'], use_point=False)
    iqprof, iqbkg = calc_qprofiles(newproblem, initpts, newQs)
    data = create_init_data_N(newQs, iqprof, dRoR=10.0)

    while t[-1] < maxtime:
        starttime = time.time()
        print('Now on cycle %i' % k, flush=True)
        print('Total time so far: %f' % t[-1])
        fid.write('Cycle: %i\n' % k)
        fid.write('Q: ' + ', '.join(map(str, newQs)) + '\n')
        fid.write('Time: ' + ', '.join(map(str, meas_time)) + '\n')
        print('newQ: ', newQs)
        print('Time: ' + ', '.join(map(str, meas_time)))
        meastimes.append(meas_time)
        t = np.cumsum(np.array([sum(m) for m in meastimes]))
        if k > 0:
            newvars = gen_new_variables(newQs)
            calcR = calc_expected_R(calcprobe, *newvars, oversampling=oversampling)
            data = append_data_N(newQs, calcR, meas_time, bkg, *data)

        newQs, meas_time, restart_pop, best_logp, final_chisq, d, newfoms, qprof, qbkg, fom = run_cycle(newmodel, measQ, newQs, data, use_entropy=True, restart_pop=restart_pop, outfid=None)

        # impose a minimum 10 s measurement time
        meas_time = np.max(np.vstack((meas_time, 10.0 * np.ones_like(meas_time))), axis=0)

        Hs.append(calc_entropy(d.points / par_scale))
        Hs_marg.append(calc_entropy(d.points / par_scale, select_pars=sel))
        best_logps.append(best_logp)
        median_logps.append(np.median(d.logp))
        foms.append(fom)
        varXs.append(np.std(d.points, axis=0)**2)
        fid.write('final chisq: %s\n' % final_chisq)
        fid.write('entropy: %f\nmarginalized entropy: %f\nbest_logp: %f\nmedian_logp: %f\n' % (Hs[-1], Hs_marg[-1], best_logps[-1], median_logps[-1]))
        fid.write('calculation wall time (s): %f\n' % (time.time() - starttime))
        print('entropy, marginalized entropy: %f, %f' % (Hs[-1], Hs_marg[-1]))
        fid.flush()

        if movieyn:
            axtop, axbot = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0})
            plotdata = tuple([v[len(meastimes[0]):] for v in data]) if k > 0 else None
            plot_qprofiles(measQ, qprof, d.logp, data=plotdata, ax=axtop)
            axtop.set_title('t = %0.0f s' % t[-1], fontweight='bold', fontsize='larger')
            if last_fom is not None:
                axbot.semilogy(measQ, last_fom, linewidth=2, alpha=0.4, color='C0')
            last_fom = fom
            axbot.semilogy(measQ, fom, linewidth=3, color='C0')
            axbot.plot(newQs, newfoms, 'o', alpha=0.5, markersize=12, color='C1')
            axbot.set_xlabel(axtop.get_xlabel())
            axbot.set_ylabel('figure of merit')
            fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            fig.clf()

        k += 1
        #np.savetxt(fn + fsuffix + '-fom%i.%i.txt' % (kk, k), np.vstack((measQ, np.array(foms))))

    if movieyn:
#        skvideo.io.vwrite(fn + fsuffix + '-movie%i.mp4' % kk, frames, outputdict={'-r': '2', '-crf': '20', '-profile:v': 'baseline', '-level': '3.0', '-pix_fmt': 'yuv420p'})
        imageio.mimsave(fn + fsuffix + '-movie%i.gif' % kk, frames, fps=1)

    all_t.append(t)
    all_Hs.append(Hs)
    all_Hs_marg.append(Hs_marg)
    all_best_logps.append(best_logps)
    all_median_logps.append(median_logps)
    all_foms.append(foms)

    fid.write('iteration wall time (s): %f\n' % (time.time() - iteration_start))
    fid.flush()

    np.savetxt(fn + fsuffix + '-timevars%i.txt' % kk, np.vstack((t, Hs, Hs_marg, Hs0-np.array(Hs), Hs0_marg - np.array(Hs_marg), best_logps, median_logps, np.array(varXs).T)).T, header='t, Hs, Hs_marg, dHs, dHs_marg, best_logps, median_logps, nxparameter_variances')
    data = np.array(data)
    cycletime = np.array([(i, val) for i, m in enumerate(meastimes) for val in m])
#    print(data.shape, cycletime.shape, data[0,:][None,:].shape, data[:,0][:,None].shape)
    np.savetxt(fn + fsuffix + '-data%i.txt' % kk, np.hstack((a2q(data[0,:], data[2,:])[:,None], cycletime, data.T)), header='Q, cycle, meas_time, T, dT, L, dL, Nspecular, Nbackground, Nincident')
    np.savetxt(fn + fsuffix + '-foms%i.txt' % (kk), np.vstack((measQ, np.array(foms))).T, header='Q, figure_of_merit')

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
