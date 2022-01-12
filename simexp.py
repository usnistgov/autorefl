import numpy as np
import copy
import time
import dill
#from bumps.cli import load_model, load_best
from bumps.fitters import DreamFit, ConsoleMonitor, _fill_defaults, StepMonitor
from bumps.initpop import generate
#from bumps.dream.state import load_state
#from refl1d.names import FitProblem, Experiment
from refl1d.resolution import TL2Q, dTdL2dQ
import matplotlib.pyplot as plt
from matplotlib import cm, colors
#from bumps.mapper import can_pickle, SerialMapper
from sklearn.linear_model import LinearRegression
#from scipy.stats import poisson
import autorefl as ar

def magik_intensity(Q, modelnum=None):
    # gives counts / second as a function of Q for MAGIK
    ps1 = np.array([ 1.35295366e+01, -9.99016840e-04])
    p_intens = np.array([ 5.56637543e+02,  7.27944632e+04,  2.13479802e+02, -4.37052050e+01])
    news1 = np.polyval(ps1, Q)
    incident_neutrons = np.polyval(p_intens, news1)
    
    return incident_neutrons

class DataPoint(object):
    """ Container object for single data point"""
    def __init__(self, meastime, modelnum, data, merit=None):
        self.model = modelnum
        self.t = meastime
        self.merit = merit
        self._data = None
        self.data = data

    @property
    def data(self):
        """ data is a length-7 tuple with fields T, dT, L, dL, Nspec, Nbkg, Nincident"""
        return self._data

    @data.setter
    def data(self, newdata):
        self._data = newdata
        self.T, self.dT, self.L, self.dL, self.N, self.Nbkg, self.Ninc = newdata

    def Q(self):
        return TL2Q(self.T, self.L)

class ExperimentStep(object):
    """ Container object for experiment step"""
    def __init__(self, points, use=True) -> None:
        self.points = points
        self.H = None
        self.dH = None
        self.H_marg = None
        self.dH_marg = None
        self.foms = None
        self.scaled_foms = None
        self.meastimes = None
        self.qprofs = None
        self.qbkgs = None
        self.best_logp = None
        self.final_chisq = None
        self.draw = None
        self.chain_pop = None
        self.use = use

    def getdata(self, attr, modelnum):
        # returns all data of type "attr" for a specific model
        if self.use:
            return [getattr(pt, attr) for pt in self.points if pt.model == modelnum]
        else:
            return []

    def meastime(self):
        return sum([pt.t for pt in self.points])


class SimReflExperiment(object):

    def __init__(self, problem, Q, f_intensity=magik_intensity, bestpars=None, fit_options=None, oversampling=11, bkg=1e-6, startmodel=0, min_meas_time=10, select_pars=None) -> None:
        # running list of options: oversampling, background x nmodels, minQ, maxQ, fit_options, startmodel, wavelength
        # more options: eta, npoints, (nrepeats not necessary because multiple objects can be made and run), switch_penalty, min_meas_time
        # problem is the FitProblem object to simulate
        # Q is a single Q vector or a list of measurement Q vectors, one for each model in problem
        # f_intensity is a function that gives the incident intensity as a function of Q. TODO: function of theta for candor and TOF?
        
        self.attr_list = ['T', 'dT', 'L', 'dL', 'N', 'Nbkg', 'Ninc']

        # Analysis options
        self.eta = 0.8
        self.npoints = int(1.0)
        self.switch_penalty = 1.0
        self.min_meas_time = min_meas_time

        # Initialize
        self.problem = problem
        models = [problem] if hasattr(problem, 'fitness') else list(problem.models)
        self.models = models
        self.nmodels = len(models)
        self.curmodel = startmodel
        self.oversampling = oversampling
        self.fit_options = fit_options
        for m in self.models:
            m.fitness.probe.oversample(oversampling)
            m.fitness.update()

        self.intensity = f_intensity
        self.L = 5.0
        self.dL = 0.01648374 * self.L

        if len(Q.shape) == 1:
            self.measQ = np.broadcast_to(Q, (self.nmodels, len(Q)))
        elif len(Q.shape) == 2:
            assert (Q.shape[0]==self.nmodels), "Q array must be a single vector or have first dimension equal to the number of models in problem"
            self.measQ = Q
        else:
            raise Exception('Bad Q shape')

        self.npars = len(problem.getp())
        self.orgQ = [list(m.fitness.probe.Q) for m in models]
        calcmodel = copy.deepcopy(problem)
        self.calcmodels = [calcmodel] if hasattr(calcmodel, 'fitness') else list(calcmodel.models)
        if bestpars is not None:
            calcmodel.setp(bestpars)

        if not isinstance(bkg, (list, np.ndarray)):
            self.bkg = np.full(self.nmodels, bkg)

        # add in background
        for c, bval in zip(self.calcmodels, self.bkg):
            c.fitness.probe.background.value = bval

        self.newmodels = [m.fitness for m in models]
        self.par_scale = np.diff(problem.bounds(), axis=0)

        if select_pars is None:
            self.sel = np.arange(self.npars)
        else:
            self.sel = np.array(select_pars, ndmin=1)

        self.init_entropy = ar.calc_init_entropy(problem)
        self.init_entropy_marg = ar.calc_init_entropy(problem, select_pars=select_pars)

        self.steps = []
        self.restart_pop = None

    def start_mapper(self):

        setattr(self.problem, 'calcQs', self.measQ)
        setattr(self.problem, 'oversampling', self.oversampling)

        self.mapper, self.mappercalc = ar.MPMapper.start_mapper(self.problem, None, cpus=0)

    def stop_mapper(self):

        ar.MPMapper.pool.terminate()
        
        # allow start_mapper call again
        ar.MPMapper.pool = None

    def get_all_points(self, modelnum):
        return [pt for step in self.steps for pt in step.points if pt.model == modelnum]

    def getdata(self, attr, modelnum):
        # returns all data of type "attr" for a specific model
        return [getattr(pt, attr) for pt in self.get_all_points(modelnum)]

    def compile_datapoints(self, Qbasis, points):

        idata = [[getattr(pt, attr) for pt in points] for attr in self.attr_list]

        return ar.compile_data_N(copy.copy(Qbasis), *idata)

    def add_initial_step(self, dRoR=10.0):

        # generate initial data set. This is only necessary because of the requirement that dof > 0
        # in Refl1D (probably not strictly required for DREAM fit)
        nQs = [((self.npars + 1) // self.nmodels) + 1 if i < ((self.npars + 1) % self.nmodels) else ((self.npars + 1) // self.nmodels) for i in range(self.nmodels)]
        newQs = [np.linspace(min(Qvec), max(Qvec), nQ) for nQ, Qvec in zip(nQs, self.measQ)]
        new_meastimes = [np.zeros_like(newQ) for newQ in newQs]
        newfoms = [np.zeros_like(newQ) for newQ in newQs]

        initpts = generate(self.problem, init='lhs', pop=self.fit_options['pop'], use_point=False)
        init_qprof, _ = ar.calc_qprofiles(self.problem, initpts, self.newQs)
    
        points = []

        for mnum, (newQ, mtime, mfom, qprof, bkg) in enumerate(zip(newQs, new_meastimes, newfoms, init_qprof, self.bkg)):
            newR, newdR = np.mean(qprof, axis=0), dRoR * np.std(qprof, axis=0)
            targetN = (newR / newdR) ** 2
            target_incident_neutrons = targetN / newR
            Ns, Nbkgs, Nincs = ar.sim_data_N(newR, target_incident_neutrons, background=bkg)
            #print(newR, target_incident_neutrons, Ns, Nbkgs, Nincs)
            Ts = ar.q2a(newQ, self.L)
            # TODO: Replace with resolution function
            dTs = np.polyval(np.array([ 2.30358547e-01, -1.18046955e-05]), newQ)
            Ls = np.ones_like(newQ)*self.L
            dLs = np.ones_like(newQ)*self.dL
            for t, T, dT, L, dL, N, Nbkg, Ninc, fom in zip(mtime, Ts, dTs, Ls, dLs, Ns, Nbkgs, Nincs, mfom):
                points.append(DataPoint(t, mnum, (T, dT, L, dL, N, Nbkg, Ninc), merit=fom))

        self.add_step(points, use=False)

    def update_models(self):

        for i, (m, measQ) in enumerate(zip(self.models, self.measQ)):
            mT, mdT, mL, mdL, mR, mdR, mQ, mdQ = self.compile_datapoints(measQ, self.get_all_points(i))
            m.fitness.probe._set_TLR(mT, mdT, mL, mdL, mR, mdR, dQ=None)
            m.fitness.probe.oversample(self.oversampling)
            m.fitness.update()
        
        self.problem.model_reset()
        self.problem.chisq_str()

    def calc_qprofiles(self, drawpoints, mappercalc):
        # this version is limited to calculating profiles with measQ, cannot be used with initial calculation
        res = mappercalc(drawpoints)

        qprofs = list()
        qbkgs = list()
        for i in range(self.nmodels):
            qprofs.append(np.array([r[0][i] for r in res]))
            qbkgs.append(np.array([r[1][i] for r in res]))

        return qprofs, qbkgs

    def fit_step(self, outfid=None):
        """Analyzes most recent step"""
        
        self.update_models()

        setattr(self.problem, 'calcQs', self.measQ)
        setattr(self.problem, 'oversampling', self.oversampling)

        mapper, mappercalc = ar.MPMapper.start_mapper(self.problem, None, cpus=0)

        if outfid is not None:
            monitor = StepMonitor(self.problem, outfid)
        else:
            monitor = ConsoleMonitor(self.problem)
        fitter = ar.DreamFitPlus(self.problem)
        options=_fill_defaults(self.fit_options, fitter.settings)
        result = fitter.solve(mapper=mapper, monitors=[monitor], initial_population=self.restart_pop, **options)

        _, chains, _ = fitter.state.chains()
        self.restart_pop = chains[-1, : ,:]

        fitter.state.keep_best()
        fitter.state.mark_outliers()

        step = self.steps[-1]
        step.chain_pop = chains[-1, :, :]
        step.draw = fitter.state.draw(thin=self.fit_options['steps']*2)
        step.best_logp = fitter.state.best()[1]
        self.problem.setp(fitter.state.best()[0])
        step.final_chisq = self.problem.chisq_str()
        step.H = ar.calc_entropy(step.draw.points)
        step.dH = self.init_entropy - step.H
        step.H_marg = ar.calc_entropy(step.draw.points, select_pars=self.sel)
        step.dH_marg = self.init_entropy_marg - step.H_marg

        print('Calculating %i Q profiles:' % (step.draw.points.shape[0]))
        init_time = time.time()
        step.qprofs, step.qbkgs = self.calc_qprofiles(step.draw.points, mappercalc)
        print('Calculation time: %f' % (time.time() - init_time))

        ar.MPMapper.stop_mapper(mapper)
        ar.MPMapper.pool = None

    def take_step(self):

        step = self.steps[-1]
        
        step.foms, step.meastimes = self.calc_foms(step)

        points = []
        
        for i in range(self.npoints):
            # all models incur switch penalty except the current one
            spenalty = np.full((self.nmodels, 1), self.switch_penalty)
            spenalty[self.curmodel] = 1.0

            step.scaled_foms = np.array(step.foms) / spenalty

            newpoint = self.select_new_point(step, start=i)
            if newpoint is not None:
                points.append(newpoint)
                self.curmodel = newpoint.model
            else:
                break

        self.add_step(points)

    def add_step(self, points, use=True):

        self.steps.append(ExperimentStep(points, use=use))

    def calc_foms(self, step):
    
        # define parameter numbers to select
        pts = step.draw.points[:,self.sel]

        # alternative approach
        foms = list()
        meas_times = list()
        for mnum, (m, Qth, qprof, qbkg) in enumerate(zip(self.models, self.measQ, step.qprofs, step.qbkgs)):

            incident_neutrons = self.intensity(Qth, modelnum=mnum)

            # define signal to background. For now, this is just a scaling factor on the effective rate
            sbr = (qprof - qbkg[:,None]) / qbkg[:,None]
            refl_rate = incident_neutrons * np.mean((qprof - qbkg[:,None])/(1+2/sbr), axis=0)
            refl_rate = np.maximum(refl_rate, np.zeros_like(refl_rate))

            # q-dependent noise. Use the minimum of the actual spread in Q and the expected spread from the nearest points.
            # TODO: Is this really the right thing to do? Should probably just be the actual spread; the problem is that if
            # the spread doesn't constrain the variables very much, then we just keep measuring at the same point over and over.
            minstd = np.min(np.vstack((np.std(qprof, axis=0), np.interp(Qth, m.fitness.probe.Q, m.fitness.probe.dR))), axis=0)
            totalrate = refl_rate * (minstd/np.mean(qprof, axis=0))**4
            #totalrate = refl_rate * (np.std(qprof, axis=0)/np.mean(qprof, axis=0))**4

            reg = LinearRegression(fit_intercept=True)
            reg.fit(step.draw.points/np.std(step.draw.points, axis=0), qprof/np.std(qprof, axis=0))
            reg_marg = LinearRegression(fit_intercept=True)
            reg_marg.fit(pts[:,:]/np.std(pts[:,:], axis=0), qprof/np.std(qprof, axis=0))
            J = reg.coef_.T
            J_marg = reg_marg.coef_.T
            covX = np.cov((step.draw.points/np.std(step.draw.points, axis=0)).T)
            covX_marg = np.cov((pts/np.std(pts, axis=0)).T)
            df2s = np.zeros_like(Qth)
            df2s_marg = np.zeros_like(Qth)    
            for j in range(len(Qth)):
                Jj = J[:,j][:,None]
                df2s[j] = np.squeeze(Jj.T @ covX @ Jj)

                Jj = J_marg[:,j][:,None]
                df2s_marg[j] = np.squeeze(Jj.T @ covX_marg @ Jj)

            fom = df2s_marg/df2s*totalrate
            foms.append(fom)

            meas_time = (1-self.eta) / (self.eta**2 * refl_rate * (minstd/np.mean(qprof, axis=0))**2)
            meas_times.append(meas_time)

        return foms, meas_times

    def select_new_point(self, step, start=0):
        # finds a single point to measure
        maxQs = []
        maxidxs = []
        maxfoms = []

        # find maximum positions in each model

        for fom, Qth in zip(step.scaled_foms, self.measQ):
            
            # a. calculate whether gradient is > 0
            dfom = np.sign(np.diff(np.append(np.insert(fom, 0, 0),0))) < 0
            # b. find zero crossings
            xings = np.diff(dfom.astype(float))
            maxidx = np.where(xings>0)[0]
            maxfoms.append(fom[maxidx])
            maxQs.append(Qth[maxidx])
            maxidxs.append(maxidx)

        maxidxs_m = [[fom, m, idx] for m, (idxs, mfoms) in enumerate(zip(maxidxs, maxfoms)) for idx, fom in zip(idxs, mfoms)]
        #print(maxidxs_m)
        # select top point
        top_n = sorted(maxidxs_m, reverse=True)[start:min(start+1, len(maxidxs_m))][0]

        if len(top_n):

            # generate a DataPoint object with the maximum point
            _, mnum, idx = top_n
            maxfom = step.foms[mnum][idx]       # use unscaled version for plotting
            newQ = self.measQ[mnum][idx]
            new_meastime = max(step.meastimes[mnum][idx], self.min_meas_time)

            newvars = ar.gen_new_variables(newQ)
            calcR = ar.calc_expected_R(self.calcmodels[mnum].fitness, *newvars, oversampling=self.oversampling)
            #print('expected R:', calcR)
            incident_neutrons = self.intensity(newQ, modelnum=mnum) * new_meastime
            N, Nbkg, Ninc = ar.sim_data_N(calcR, incident_neutrons, background=self.bkg[mnum])
            #print(newR, target_incident_neutrons, N, Nbkg, Ninc)
            T = ar.q2a(newQ, self.L)
            dT = np.polyval(np.array([ 2.30358547e-01, -1.18046955e-05]), newQ)
            L = np.ones_like(newQ)*self.L
            dL = np.ones_like(newQ)*self.dL        
            t = max(self.min_meas_time, new_meastime)

            return DataPoint(t, mnum, (T, dT, L, dL, N, Nbkg, Ninc), merit=maxfom)
        
        else:

            return None

    def save(self, fn):

        with open(fn, 'wb') as f:
            dill.dump(self, f, recurse=True)

    @classmethod
    def load(cls, fn):

        with open(fn, 'rb') as f:
            return dill.load(f)

def makemovie(exp, outfilename, fps=1, fmt='gif', power=4):
    import autorefl as ar
    """ Makes a GIF or MP4 movie from a SimReflExperiment object"""

    allt = np.cumsum([step.meastime() for step in exp.steps[:-1]])
    allH = [step.dH for step in exp.steps[:-1]]
    allH_marg = [step.dH_marg for step in exp.steps[:-1]]

    total_t = 0.0
    steptimes = np.zeros(exp.nmodels)
    fig = plt.figure(figsize=(8 + 4 * exp.nmodels, 8))
    gsright = GridSpec(2, exp.nmodels + 1, hspace=0, wspace=0.4)
    gsleft = GridSpec(2, exp.nmodels + 1, hspace=0, wspace=0)
    frames = list()
    for j, step in enumerate(exp.steps[0:-1]):
        #axtops, axbots = fig.subplots(2, exp.nmodels + 1, squeeze=False)
        axtopright = fig.add_subplot(gsright[0,-1])
        axbotright = fig.add_subplot(gsright[1,-1], sharex=axtopright)
        axtopright.plot(allt, allH_marg, 'o-')
        axbotright.plot(allt, allH, 'o-')
        axtopright.plot(allt[j], allH_marg[j], 'o', markersize=15, color='red', alpha=0.4)
        axbotright.plot(allt[j], allH[j], 'o', markersize=15, color='red', alpha=0.4)
        axbotright.set_xlabel('Time (s)')
        axbotright.set_ylabel(r'$\Delta H_{marg}$ (nats)')
        axtopright.set_ylabel(r'$\Delta H_{total}$ (nats)')

        axtops = [fig.add_subplot(gsleft[0, i]) for i in range(exp.nmodels)]
        axbots = [fig.add_subplot(gsleft[1, i]) for i in range(exp.nmodels)]

        for axtop, axbot in zip(axtops, axbots):
            axtop.sharex(axbot)
            axtop.sharey(axtops[0])
            axbot.sharey(axbots[0])
            axbot.set_xlabel(r'$Q_z$ (' + u'\u212b' + r'$^{-1}$)')
            axtop.tick_params(labelleft=False, top=True, bottom=True, left=True, right=True, direction='in')
            axbot.tick_params(labelleft=False, top=True, bottom=True, left=True, right=True, direction='in')

        axtops[0].set_ylabel(r'$R \times Q_z^%i$ (' % power + u'\u212b' + r'$^{-4}$)')
        axbots[0].set_ylabel('figure of merit')
        axtops[0].tick_params(labelleft=True)
        axbots[0].tick_params(labelleft=True)


        total_t += step.meastime()
        #print(np.array(step.qprofs).shape, step.draw.logp.shape)
        allnewpoints = exp.steps[j+1].points
        for i, (measQ, qprof, fom, axtop, axbot) in enumerate(zip(exp.measQ, step.qprofs, step.foms, axtops, axbots)):
            steptimes[i] += sum(step.getdata('t', i))
            plotpoints = [pt for step in exp.steps[:(j+1)] if step.use for pt in step.points if pt.model == i]
            #print(*[[getattr(pt, attr) for pt in plotpoints] for attr in exp.attr_list])
            idata = [[getattr(pt, attr) for pt in plotpoints] for attr in exp.attr_list]
            ar.plot_qprofiles(copy.copy(measQ), qprof, step.draw.logp, data=idata, ax=axtop, power=power)
            axtop.set_title('t = %0.0f s' % steptimes[i], fontsize='larger')
            axbot.semilogy(measQ, fom, linewidth=3, color='C0')
            newpoints = [pt for pt in exp.steps[j+1].points if pt.model == i]
            for newpt in newpoints:
                axbot.plot(newpt.Q(), newpt.merit, 'o', alpha=0.5, markersize=12, color='C1')
            ##axbot.set_xlabel(axtop.get_xlabel())
            ##axbot.set_ylabel('figure of merit')
        fig.suptitle('t = %0.0f s' % total_t, fontsize='larger', fontweight='bold')
        #fig.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        fig.clf()

    if format == 'gif':
        import imageio
        imageio.mimsave(outfilename + '.' + fmt, frames, fps=fps)
    elif format == 'mp4':
        import skvideo.io
        skvideo.io.vwrite(outfilename + '.' + fmt, frames, outputdict={'-r': '%0.1f' % fps, '-crf': '20', '-profile:v': 'baseline', '-level': '3.0', '-pix_fmt': 'yuv420p'})
