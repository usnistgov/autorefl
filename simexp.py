import numpy as np
import copy
import time
import dill
#from bumps.cli import load_model, load_best
from bumps.fitters import DreamFit, ConsoleMonitor, _fill_defaults, StepMonitor
from bumps.initpop import generate
from bumps.mapper import MPMapper
#from bumps.dream.state import load_state
#from refl1d.names import FitProblem, Experiment
from refl1d.resolution import TL2Q, dTdL2dQ
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
#from bumps.mapper import can_pickle, SerialMapper
from sklearn.linear_model import LinearRegression
#from scipy.stats import poisson
import autorefl as ar
import instrument

fit_options = {'pop': 10, 'burn': 1000, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

class DataPoint(object):
    """ Container object for single data point. All inputs should be as lists or ndarrays with min dimension 1"""
    def __init__(self, x, meastime, modelnum, data, merit=None):
        self.model = modelnum
        self.t = meastime
        self.merit = merit
        self.x = x
        self._data = None
        self.data = data

    def __repr__(self):

        try:
            reprq = 'Q: %0.4f Ang^-1' % self.Q()
        except TypeError:
            reprq = 'Q: ' + ', '.join('{:0.4f}'.format(q) for q in self.Q()) + ' Ang^-1'
        
        return ('Model: %i\t' % self.model) + reprq + ('\tTime: %0.1f s' %  self.t)

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

    def meastime(self, modelnum=None):
        if modelnum is None:
            return sum([pt.t for pt in self.points])
        else:
            return sum([pt.t for pt in self.points if pt.model == modelnum])


class SimReflExperiment(object):

    def __init__(self, problem, Q, instrument=instrument.MAGIK(), eta=0.8, npoints=1, switch_penalty=1, bestpars=None, fit_options=fit_options, oversampling=11, meas_bkg=1e-6, startmodel=0, min_meas_time=10, select_pars=None) -> None:
        # running list of options: oversampling, background x nmodels, minQ, maxQ, fit_options, startmodel, wavelength
        # more options: eta, npoints, (nrepeats not necessary because multiple objects can be made and run), switch_penalty, min_meas_time
        # problem is the FitProblem object to simulate
        # Q is a single Q vector or a list of measurement Q vectors, one for each model in problem
        # f_intensity is a function that gives the incident intensity as a function of Q. TODO: function of theta for candor and TOF?
        
        self.attr_list = ['T', 'dT', 'L', 'dL', 'N', 'Nbkg', 'Ninc']

        # Load instrument
        self.instrument = instrument

        # Analysis options
        self.eta = eta
        self.npoints = int(npoints)
        self.switch_penalty = switch_penalty
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
            m.fitness.probe.resolution = self.instrument.resolution
            m.fitness.update()

        if isinstance(Q, np.ndarray):
            if len(Q.shape) == 1:
                self.measQ = np.broadcast_to(Q, (self.nmodels, len(Q)))
            elif len(Q.shape) == 2:
                assert (Q.shape[0]==self.nmodels), "Q array must be a single vector or have first dimension equal to the number of models in problem"
                self.measQ = Q
            else:
                raise Exception('Bad Q shape')
        else:
            if any(isinstance(i, (list, np.ndarray)) for i in Q): # is a nested list
                assert (len(Q) == self.nmodels), "Q array must be a single vector or a list of vectors with length equal to the number of models in problem"
                self.measQ = Q
            else:
                self.measQ = [Q for _ in range(self.nmodels)]

        # define measurement space. Contains same number of points per model as self.measQ
        self.x = list()
        for Q in self.measQ:
            minx, maxx = self.instrument.qrange2xrange([min(Q), max(Q)])
            self.x.append(np.linspace(minx, maxx, len(Q), endpoint=True))

        self.npars = len(problem.getp())
        self.orgQ = [list(m.fitness.probe.Q) for m in models]
        calcmodel = copy.deepcopy(problem)
        self.calcmodels = [calcmodel] if hasattr(calcmodel, 'fitness') else list(calcmodel.models)
        if bestpars is not None:
            calcmodel.setp(bestpars)

        # deal with inherent measurement background
        if not isinstance(meas_bkg, (list, np.ndarray)):
            self.meas_bkg = np.full(self.nmodels, meas_bkg)
        else:
            self.meas_bkg = np.array(meas_bkg)

        # add residual background
        self.resid_bkg = np.array([c.fitness.probe.background.value for c in self.calcmodels])

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

        self.mapper = MPMapper.start_mapper(self.problem, None, cpus=0)

    def stop_mapper(self):

        MPMapper.pool.terminate()
        
        # allow start_mapper call again
        MPMapper.pool = None

    def get_all_points(self, modelnum):
        return [pt for step in self.steps for pt in step.points if pt.model == modelnum]

    def getdata(self, attr, modelnum):
        # returns all data of type "attr" for a specific model
        return [getattr(pt, attr) for pt in self.get_all_points(modelnum)]

    def compile_datapoints(self, Qbasis, points):

        idata = [[val for pt in points for val in getattr(pt, attr)] for attr in self.attr_list]

        return ar.compile_data_N(Qbasis, *idata)

    def add_initial_step(self, dRoR=10.0):

        # generate initial data set. This is only necessary because of the requirement that dof > 0
        # in Refl1D (probably not strictly required for DREAM fit)
        nQs = [((self.npars + 1) // self.nmodels) + 1 if i < ((self.npars + 1) % self.nmodels) else ((self.npars + 1) // self.nmodels) for i in range(self.nmodels)]
        newQs = [np.linspace(min(Qvec), max(Qvec), nQ) for nQ, Qvec in zip(nQs, self.measQ)]

        initpts = generate(self.problem, init='lhs', pop=self.fit_options['pop'], use_point=False)
        init_qprof, _ = ar.calc_qprofiles(self.problem, initpts, newQs)
    
        points = []

        for mnum, (newQ, qprof, meas_bkg, resid_bkg) in enumerate(zip(newQs, init_qprof, self.meas_bkg, self.resid_bkg)):
            newR, newdR = np.mean(qprof, axis=0), dRoR * np.std(qprof, axis=0)
            targetN = (newR / newdR) ** 2
            target_incident_neutrons = targetN / newR
            Ns, Nbkgs, Nincs = ar.sim_data_N(newR, target_incident_neutrons, resid_bkg=resid_bkg, meas_bkg=meas_bkg)
            #print(newR, target_incident_neutrons, Ns, Nbkgs, Nincs)
            Ts = ar.q2a(newQ, 5.0)
            # Resolution function doesn't matter here at all because these points don't have any effect
            dTs = np.polyval(np.array([ 2.30358547e-01, -1.18046955e-05]), newQ)
            Ls = np.ones_like(newQ)*5.0
            dLs = np.ones_like(newQ)*0.01648374 * 5.0
            points.append(DataPoint(0.0, 0.0, mnum, (Ts, dTs, Ls, dLs, Ns, Nbkgs, Nincs)))

        self.add_step(points, use=False)

    def update_models(self):

        for i, (m, measQ) in enumerate(zip(self.models, self.measQ)):
            mT, mdT, mL, mdL, mR, mdR, mQ, mdQ = self.compile_datapoints(measQ, self.get_all_points(i))
            m.fitness.probe._set_TLR(mT, mdT, mL, mdL, mR, mdR, dQ=None)
            m.fitness.probe.oversample(self.oversampling)
            m.fitness.probe.resolution = self.instrument.resolution
            m.fitness.update()
        
        self.problem.model_reset()
        self.problem.chisq_str()

    def calc_qprofiles(self, drawpoints, mappercalc):
        # this version is limited to calculating profiles with measQ, cannot be used with initial calculation
        res = mappercalc(drawpoints)

        qprofs = list()
        for i in range(self.nmodels):
            qprofs.append(np.array([r[i] for r in res]))

        return qprofs

    def fit_step(self, outfid=None):
        """Analyzes most recent step"""
        
        self.update_models()

        setattr(self.problem, 'calcQs', self.measQ)
        setattr(self.problem, 'oversampling', self.oversampling)
        setattr(self.problem, 'resolution', self.instrument.resolution)

        mapper = MPMapper.start_mapper(self.problem, None, cpus=0)

        mappercalc = lambda points: MPMapper.pool.map(_MP_calc_qprofile, ((MPMapper.problem_id, p) for p in points))

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
        step.qprofs = self.calc_qprofiles(step.draw.points, mappercalc)
        print('Calculation time: %f' % (time.time() - init_time))

        MPMapper.stop_mapper(mapper)
        MPMapper.pool = None

    def take_step(self):

        step = self.steps[-1]
        
        step.foms, step.meastimes = self.calc_foms(step)

        points = []
        
        for i in range(self.npoints):
            # all models incur switch penalty except the current one
            spenalty = [1.0 if j == self.curmodel else self.switch_penalty for j in range(self.nmodels)]
            step.scaled_foms = [fom / pen for fom, pen in zip(step.foms, spenalty)]

            newpoint = self.select_new_point(step, start=i)
            if newpoint is not None:
                points.append(newpoint)
                print('New data point:\t' + repr(newpoint))
                self.curmodel = newpoint.model
            else:
                break
        
        self.add_step(points)

    def add_step(self, points, use=True):

        self.steps.append(ExperimentStep(points, use=use))

    def calc_foms(self, step):
    
        # define parameter numbers to select
        pts = step.draw.points[:,self.sel]

        foms = list()
        meas_times = list()
        for mnum, (m, xs, Qth, qprof, qbkg) in enumerate(zip(self.models, self.x, self.measQ, step.qprofs, self.meas_bkg)):

            incident_neutrons = self.instrument.intensity(xs)

            # define signal to background. For now, this is just a scaling factor on the effective rate
            sbr = qprof / qbkg
            refl = np.mean(qprof/(1+2/sbr), axis=0)
            refl = np.maximum(refl, np.zeros_like(refl))

            # q-dependent noise. Use the minimum of the actual spread in Q and the expected spread from the nearest points.
            # TODO: Is this really the right thing to do? Should probably just be the actual spread; the problem is that if
            # the spread doesn't constrain the variables very much, then we just keep measuring at the same point over and over.
            minstd = np.min(np.vstack((np.std(qprof, axis=0), np.interp(Qth, m.fitness.probe.Q, m.fitness.probe.dR))), axis=0)
            normrefl = refl * (minstd/np.mean(qprof, axis=0))**4

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

            qfom_norm = df2s_marg/df2s*normrefl
            fom = list()
            meas_time = list()
            for x, intens in zip(xs, incident_neutrons):
                q = self.instrument.x2q(x)
                xrefl = intens * np.interp(q, Qth, refl * (minstd/np.mean(qprof, axis=0))**2)
                # TODO: check this. Should it be the average of xrefl, or the sum?
                meas_time.append(np.mean((1-self.eta) / (self.eta**2 * xrefl)))
                fom.append(np.sum(intens * np.interp(q, Qth, qfom_norm)))
            foms.append(np.array(fom))
            meas_times.append(np.array(meas_time))

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
            newx = self.x[mnum][idx]
            new_meastime = max(step.meastimes[mnum][idx], self.min_meas_time)

            T = self.instrument.T(newx)[0]
            dT = self.instrument.dT(newx)[0]
            L = self.instrument.L(newx)[0]
            dL = self.instrument.dL(newx)[0]
            #print(T, dT, L, dL)
            calcR = ar.calc_expected_R(self.calcmodels[mnum].fitness, T, dT, L, dL, oversampling=self.oversampling, resolution=self.instrument.resolution)
            #print('expected R:', calcR)
            incident_neutrons = self.instrument.intensity(newx) * new_meastime
            N, Nbkg, Ninc = ar.sim_data_N(calcR, incident_neutrons.T, resid_bkg=self.resid_bkg[mnum], meas_bkg=self.meas_bkg[mnum])
            
            t = max(self.min_meas_time, new_meastime)

            return DataPoint(newx, t, mnum, (T, dT, L, dL, N[0], Nbkg[0], Ninc[0]), merit=maxfom)
        
        else:

            return None

    def save(self, fn):

        with open(fn, 'wb') as f:
            dill.dump(self, f, recurse=True)

    @classmethod
    def load(cls, fn):

        with open(fn, 'rb') as f:
            return dill.load(f)

class SimReflExperimentControl(SimReflExperiment):

    def __init__(self, problem, Q, model_weights=None, instrument=instrument.MAGIK(), eta=0.8, npoints=1, switch_penalty=1, bestpars=None, fit_options=fit_options, oversampling=11, meas_bkg=0.000001, startmodel=0, min_meas_time=10, select_pars=None) -> None:
        super().__init__(problem, Q, instrument=instrument, eta=eta, npoints=npoints, switch_penalty=switch_penalty, bestpars=bestpars, fit_options=fit_options, oversampling=oversampling, meas_bkg=meas_bkg, startmodel=startmodel, min_meas_time=min_meas_time, select_pars=select_pars)

        if model_weights is None:
            model_weights = np.ones(self.nmodels)
        else:
            assert (len(model_weights) == self.nmodels), "weights must have same length as number of models"
        
        model_weights = np.array(model_weights) / np.sum(model_weights)

        self.meastimeweights = list()
        for x, weight in zip(self.x, model_weights):
            self.meastimeweights.append(weight * np.array(x)**2 / np.sum(np.array(x)**2))

    def take_step(self, total_time):
        points = list()
        #TODO: Make this into a (Q to points) function
        for mnum, (newx, mtimeweight, meas_bkg, resid_bkg) in enumerate(zip(self.x, self.meastimeweights, self.meas_bkg, self.resid_bkg)):

            Ts = self.instrument.T(newx)
            dTs = self.instrument.dT(newx)
            Ls = self.instrument.L(newx)
            dLs = self.instrument.dL(newx)
            #print(T, dT, L, dL)
            incident_neutrons = self.instrument.intensity(newx)
            for x, t, T, dT, L, dL, intens in zip(newx, total_time * mtimeweight, Ts, dTs, Ls, dLs, incident_neutrons):
                calcR = ar.calc_expected_R(self.calcmodels[mnum].fitness, T, dT, L, dL, oversampling=self.oversampling)
            #print('expected R:', calcR)
                N, Nbkg, Ninc = ar.sim_data_N(calcR, intens.T * t, resid_bkg=resid_bkg, meas_bkg=meas_bkg)
                points.append(DataPoint(x, t, mnum, (T, dT, L, dL, N, Nbkg, Ninc)))
        
        self.add_step(points)

def _MP_calc_qprofile(problem_point_pair):
    # given a problem and a sample draw and a Q-vector, calculate the profiles associated with each sample
    problem_id, point = problem_point_pair
    if problem_id != MPMapper.problem_id:
        #print(f"Fetching problem {problem_id} from namespace")
        # Problem is pickled using dill when it is available
        try:
            import dill
            MPMapper.problem = dill.loads(MPMapper.namespace.pickled_problem)
        except ImportError:
            MPMapper.problem = MPMapper.namespace.problem
        MPMapper.problem_id = problem_id
    return _calc_qprofile(MPMapper.problem, point)

def _calc_qprofile(calcproblem, point):
    mlist = [calcproblem] if hasattr(calcproblem, 'fitness') else list(calcproblem.models)
    newvars = [ar.gen_new_variables(Q) for Q in calcproblem.calcQs]
    qprof = list()
    for m, newvar in zip(mlist, newvars):
        calcproblem.setp(point)
        calcproblem.chisq_str()
        Rth = ar.calc_expected_R(m.fitness, *newvar, oversampling=calcproblem.oversampling, resolution=calcproblem.resolution)
        qprof.append(Rth)

    return qprof


def load_entropy(steps):

    allt = np.cumsum([step.meastime() for step in steps])
    allH = [step.dH for step in steps]
    allH_marg = [step.dH_marg for step in steps]

    return allt, allH, allH_marg


def snapshot(exp, stepnumber, fig=None, power=4, tscale='log'):

    allt, allH, allH_marg = load_entropy(exp.steps[:-1])

    if fig is None:
        fig = plt.figure(figsize=(8 + 4 * exp.nmodels, 8))
    gsright = GridSpec(2, exp.nmodels + 1, hspace=0, wspace=0.4)
    gsleft = GridSpec(2, exp.nmodels + 1, hspace=0.2, wspace=0)
    j = stepnumber
    step = exp.steps[j]

    steptimes = [sum([step.meastime(modelnum=i) for step in exp.steps[:(j+1)]]) for i in range(exp.nmodels)]

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
        ar.plot_qprofiles(copy.copy(measQ), qprof, step.draw.logp, data=idata, ax=axtop, power=power)
        axtop.set_title('t = %0.0f s' % steptimes[i], fontsize='larger')
        axbot.semilogy(x, fom, linewidth=3, color='C0')
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

    fig.suptitle('t = %0.0f s' % sum(steptimes), fontsize='larger', fontweight='bold')

    return fig, (axtops, axbots, axtopright, axbotright)

def makemovie(exp, outfilename, expctrl=None, fps=1, fmt='gif', power=4, tscale='log'):
    """ Makes a GIF or MP4 movie from a SimReflExperiment object"""

    fig = plt.figure(figsize=(8 + 4 * exp.nmodels, 8))

    frames = list()

    for j in range(len(exp.steps[0:-1])):

        fig, (_, _, axtopright, axbotright) = snapshot(exp, j, fig=fig, power=power, tscale=tscale)

        if expctrl is not None:
            allt, allH, allH_marg = load_entropy(expctrl.steps)
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
