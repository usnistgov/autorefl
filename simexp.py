import numpy as np
import copy
import time
import dill
#from bumps.cli import load_model, load_best
from bumps.fitters import DreamFit, ConsoleMonitor, _fill_defaults, StepMonitor
from bumps.initpop import generate
from bumps.mapper import MPMapper
from bumps.dream.stats import credible_interval
#from bumps.dream.state import load_state
#from refl1d.names import FitProblem, Experiment
from refl1d.resolution import TL2Q, dTdL2dQ
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
#from bumps.mapper import can_pickle, SerialMapper
from sklearn.linear_model import LinearRegression
#from scipy.stats import poisson
from scipy.interpolate import interp1d
import autorefl as ar
import instrument

fit_options = {'pop': 10, 'burn': 1000, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

class DataPoint(object):
    """ Container object for a single data point.

    A "single data point" normally corresponds to a single instrument configuration.
    Note that for polychromatic and time-of-flight instruments, this may involve multiple
    Q values. As a result, all of the "data" fields (described below) are stored 
    as lists or numpy.ndarrays.

    Required attributes:
    model -- the index of the bumps.FitProblem model with which the data point
             is associated
    t -- the total measurement time 
    movet -- the total movement time. Note that this varies depending on what the 
            previous point was.
    x -- a description of the instrument configuration, usually as a single number
        whose interpretation is determined by the instrument class (e.g. Q for MAGIK,
        Theta for CANDOR)
    merit -- if calculated, the figure of merit of this data point. Mainly used for plotting.

    Data attributes. When initializing, these are required as the argument "data" 
    in a tuple of lists or arrays.
    T -- theta array
    dT -- angular resolution array
    L -- wavelength array
    dL -- wavelength uncertainty array
    N -- neutron counts at this instrument configuration
    Nbkg -- background neutron counts at this instrument configuration
    Ninc -- incident neutron counts at this instrument configuration

    Methods:
    Q -- returns an array of Q points corresponding to T and L.
    """

    def __init__(self, x, meastime, modelnum, data, merit=None, movet=0.0):
        self.model = modelnum
        self.t = meastime
        self.movet = movet
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
        """ gets the internal data variable"""
        return self._data

    @data.setter
    def data(self, newdata):
        """populates T, dT, L, dL, N, Nbkg, Ninc.
            newdata is a length-7 tuple of lists"""
        self._data = newdata
        self.T, self.dT, self.L, self.dL, self.N, self.Nbkg, self.Ninc = newdata

    def Q(self):
        return TL2Q(self.T, self.L)

class ExperimentStep(object):
    """ Container object for a single experiment step.

        Attributes:
        points -- a list of DataPoint objects
        H -- MVN entropy in all parameters
        dH -- change in H from the initial step (with no data and calculated
                only from the bounds of the model parameters)
        H_marg -- MVN entropy from selected parameters (marginalized entropy)
        dH_marg -- change in dH from the initial step
        foms -- list of the figures of merit for each model
        scaled_foms -- figures of merit after various penalties are applied. Possibly
                        not useful
        meastimes -- list of the measurement time proposed for each Q value of each model
        qprofs -- list of Q profile arrays calculated from each sample from the MCMC posterior
        qbkgs -- not used
        best_logp -- best nllf after fitting
        final_chisq -- final chi-squared string (including uncertainty) after fitting
        draw -- an MCMCDraw object containing the best fit results
        chain_pop -- MCMC chain heads for use in DreamFitPlus for initializing the MCMC
                     fit. Useful for restarting fits from an arbitrary step.
        use -- a flag for whether the step contains real data and should be used in furthur
                analysis.
        
        TODO: do not write draw.state, which is inflating file sizes!

        Methods:
        getdata -- returns all data of type "attr" for data points from a specific model
        meastime -- returns the total measurement time or the time from a specific model
        movetime -- returns the total movement time or the time from a specific model
    """

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

    def movetime(self, modelnum=None):
        if modelnum is None:
            return sum([pt.movet for pt in self.points])
        else:
            return sum([pt.movet for pt in self.points if pt.model == modelnum])


class SimReflExperiment(object):
    """
    A simulated reflectometry experiment.

    Contains methods for defining the experiment (via a bumps.FitProblem) object,
    simulating data from a specific instrument (via a ReflectometerBase-d object from
    the instrument module), fitting simulated data (via Refl1D), and determining the
    next optimal measurement point. Also allows saving and loading.

    Typical workflow:
        exp = SimReflExperiment(...)
        exp.add_initial_step()
        while (condition):
            exp.fit_step()
            exp.take_step()
    """

    def __init__(self, problem, Q, instrument=instrument.MAGIK(), eta=0.8, npoints=1, switch_penalty=1, bestpars=None, fit_options=fit_options, oversampling=11, meas_bkg=1e-6, startmodel=0, min_meas_time=10, select_pars=None) -> None:
        # running list of options: oversampling, background x nmodels, minQ, maxQ, fit_options, startmodel, wavelength
        # more options: eta, npoints, (nrepeats not necessary because multiple objects can be made and run), switch_penalty, min_meas_time
        # problem is the FitProblem object to simulate
        # Q is a single Q vector or a list of measurement Q vectors, one for each model in problem
        
        self.attr_list = ['T', 'dT', 'L', 'dL', 'N', 'Nbkg', 'Ninc']

        # Load instrument
        self.instrument = instrument

        # Analysis options
        self.eta = eta
        self.npoints = int(npoints)
        self.switch_penalty = switch_penalty
        self.switch_time_penalty = 0.0          # turn into parameter later?
        self.min_meas_time = min_meas_time

        # Initialize the fit problem
        self.problem = problem
        models = [problem] if hasattr(problem, 'fitness') else list(problem.models)
        self.models = models
        self.nmodels = len(models)
        self.curmodel = startmodel
        self.oversampling = oversampling
        for m in self.models:
            m.fitness.probe.oversample(oversampling)
            m.fitness.probe.resolution = self.instrument.resolution
            m.fitness.update()

        # Condition Q vector to a list of arrays, one for each model
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
        # measurement space is instrument specific (e.g. for MAGIK x=Q but for polychromatic
        # or TOF instruments x = Theta). In principle x can be anything that can be mapped
        # to a specific instrument configuration; this is defined in the instrument module.
        # TODO: Make separate measurement list. Because Q is used for rebinning, it should
        # have a different length from "x"
        self.x = list()
        for Q in self.measQ:
            minx, maxx = self.instrument.qrange2xrange([min(Q), max(Q)])
            self.x.append(np.linspace(minx, maxx, len(Q), endpoint=True))

        # Create a copy of the problem for calculating the "true" reflectivity profiles
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

        # these are not used
        self.newmodels = [m.fitness for m in models]
        self.par_scale = np.diff(problem.bounds(), axis=0)

        # set and condition selected parameters for marginalization; use all parameters
        # if none are specified
        if select_pars is None:
            self.sel = np.arange(self.npars)
        else:
            self.sel = np.array(select_pars, ndmin=1)

        # calculate initial MVN entropy in the problem
        self.init_entropy = ar.calc_init_entropy(problem)
        self.init_entropy_marg = ar.calc_init_entropy(problem, select_pars=select_pars)

        # initialize objects required for fitting
        self.fit_options = fit_options
        self.steps = []
        self.restart_pop = None

    def start_mapper(self):
        # deprecated: the call to "self" in self.mapper really slows down multiprocessing
        setattr(self.problem, 'calcQs', self.measQ)
        setattr(self.problem, 'oversampling', self.oversampling)

        self.mapper = MPMapper.start_mapper(self.problem, None, cpus=0)

    def stop_mapper(self):
        # terminates the multiprocessing mapper pool
        MPMapper.pool.terminate()
        
        # allow start_mapper call again
        MPMapper.pool = None

    def get_all_points(self, modelnum):
        # returns all data points associated with model with index modelnum
        return [pt for step in self.steps for pt in step.points if pt.model == modelnum]

    def getdata(self, attr, modelnum):
        # returns all data of type "attr" for a specific model
        return [getattr(pt, attr) for pt in self.get_all_points(modelnum)]

    def compile_datapoints(self, Qbasis, points):
        # bins all of the data from a list "points" onto a Q-space "Qbasis"
        idata = [[val for pt in points for val in getattr(pt, attr)] for attr in self.attr_list]

        return ar.compile_data_N(Qbasis, *idata)

    def add_initial_step(self, dRoR=10.0):
        # generate initial data set. This is only necessary because of the requirement that dof > 0
        # in Refl1D (probably not strictly required for DREAM fit)
        # dRoR is the target uncertainty relative to the average of the reflectivity and
        # determines the "measurement time" for the initial data set. This should be larger
        # than about 3 so as not to constrain the parameters before collecting any real data.

        # evenly spread the Q points over the models in the problem
        nQs = [((self.npars + 1) // self.nmodels) + 1 if i < ((self.npars + 1) % self.nmodels) else ((self.npars + 1) // self.nmodels) for i in range(self.nmodels)]
        newQs = [np.linspace(min(Qvec), max(Qvec), nQ) for nQ, Qvec in zip(nQs, self.measQ)]

        # generate an initial population and calculate the associated q-profiles
        initpts = generate(self.problem, init='lhs', pop=self.fit_options['pop'], use_point=False)
        init_qprof, _ = ar.calc_qprofiles(self.problem, initpts, newQs)
    
        points = []

        # simulate data based on the q profiles. The uncertainty in the parameters is estimated
        # from the dRoR paramter
        for mnum, (newQ, qprof, meas_bkg, resid_bkg) in enumerate(zip(newQs, init_qprof, self.meas_bkg, self.resid_bkg)):
            # calculate target mean and uncertainty from unconstrained profiles
            newR, newdR = np.mean(qprof, axis=0), dRoR * np.std(qprof, axis=0)

            # calculate target number of measured neutrons to give the correct uncertainty with
            # Poisson statistics
            targetN = (newR / newdR) ** 2

            # calculate the target number of incident neutrons to give the target reflectivity
            target_incident_neutrons = targetN / newR

            # simulate the data
            Ns, Nbkgs, Nincs = ar.sim_data_N(newR, target_incident_neutrons, resid_bkg=resid_bkg, meas_bkg=meas_bkg)

            # Calculate T, dT, L, dL. Note that because these data don't constrain the model at all,
            # these values are brought in from MAGIK (not instrument-specific) because they don't need
            # to be.
            Ts = ar.q2a(newQ, 5.0)
            # Resolution function doesn't matter here at all because these points don't have any effect
            dTs = np.polyval(np.array([ 2.30358547e-01, -1.18046955e-05]), newQ)
            Ls = np.ones_like(newQ)*5.0
            dLs = np.ones_like(newQ)*0.01648374 * 5.0

            # Append the data points with zero measurement time
            points.append(DataPoint(0.0, 0.0, mnum, (Ts, dTs, Ls, dLs, Ns, Nbkgs, Nincs)))

        # Add the step with the new points
        self.add_step(points, use=False)

    def update_models(self):
        # Update the models in the fit problem with new data points. Should be run every time
        # new data are to be incorporated into the model
        for i, (m, measQ) in enumerate(zip(self.models, self.measQ)):
            mT, mdT, mL, mdL, mR, mdR, mQ, mdQ = self.compile_datapoints(measQ, self.get_all_points(i))
            m.fitness.probe._set_TLR(mT, mdT, mL, mdL, mR, mdR, dQ=mdQ)
            m.fitness.probe.oversample(self.oversampling)
            m.fitness.probe.resolution = self.instrument.resolution
            m.fitness.update()
        
        # Triggers recalculation of all models
        self.problem.model_reset()
        self.problem.chisq_str()

    def calc_qprofiles(self, drawpoints, mappercalc):
        # q-profile calculator using multiprocessing for speed
        # this version is limited to calculating profiles with measQ, cannot be used with initial calculation
        res = mappercalc(drawpoints)

        # condition output of mappercalc to a list of q-profiles for each model
        qprofs = list()
        for i in range(self.nmodels):
            qprofs.append(np.array([r[i] for r in res]))

        return qprofs

    def fit_step(self, outfid=None):
        """Analyzes most recent step"""
        
        # Update models
        self.update_models()

        # Set attributes of "problem" for passing into multiprocessing routines
        setattr(self.problem, 'calcQs', self.measQ)
        setattr(self.problem, 'oversampling', self.oversampling)
        setattr(self.problem, 'resolution', self.instrument.resolution)

        # initialize mappers for Dream fit and for Q profile calculations
        mapper = MPMapper.start_mapper(self.problem, None, cpus=0)
        mappercalc = lambda points: MPMapper.pool.map(_MP_calc_qprofile, ((MPMapper.problem_id, p) for p in points))

        # set output stream
        if outfid is not None:
            monitor = StepMonitor(self.problem, outfid)
        else:
            monitor = ConsoleMonitor(self.problem)
        
        # Condition and run fit
        fitter = ar.DreamFitPlus(self.problem)
        options=_fill_defaults(self.fit_options, fitter.settings)
        result = fitter.solve(mapper=mapper, monitors=[monitor], initial_population=self.restart_pop, **options)

        # Save head state for initializing the next fit step
        _, chains, _ = fitter.state.chains()
        self.restart_pop = chains[-1, : ,:]

        # Analyze the fit state and save values
        fitter.state.keep_best()
        fitter.state.mark_outliers()

        step = self.steps[-1]
        step.chain_pop = chains[-1, :, :]
        step.draw = fitter.state.draw(thin=int(self.fit_options['steps']*0.2))
        step.best_logp = fitter.state.best()[1]
        self.problem.setp(fitter.state.best()[0])
        step.final_chisq = self.problem.chisq_str()
        step.H = ar.calc_entropy(step.draw.points)
        step.dH = self.init_entropy - step.H
        step.H_marg = ar.calc_entropy(step.draw.points, select_pars=self.sel)
        step.dH_marg = self.init_entropy_marg - step.H_marg

        # Calculate the Q profiles associated with posterior distribution
        print('Calculating %i Q profiles:' % (step.draw.points.shape[0]))
        init_time = time.time()
        step.qprofs = self.calc_qprofiles(step.draw.points, mappercalc)
        print('Calculation time: %f' % (time.time() - init_time))

        # Terminate the multiprocessing pool (required to avoid memory issues
        # if run is stopped after current fit step)
        MPMapper.stop_mapper(mapper)
        MPMapper.pool = None

    def take_step(self):
        """Analyze the last fitted step and add the next one
        
        Procedure:
            1. Calculate the figures of merit
            2. Apply penalties to the figures of merit
            TODO: apply penalties as a separate "cost function"
            3. Identify and produce the next self.npoints data points
                to simulate/measure
            4. Add a new step for fitting.
        """

        # Focus on the last step
        step = self.steps[-1]
        
        # Calculate figures of merit and proposed measurement times
        step.foms, step.meastimes = self.calc_foms(step)

        # Apply the minimum measurement time
        # TODO: Consider whether the minimum measurement time should be tied to the movement time?
        min_meas_times = [np.maximum(np.full_like(meastime, self.min_meas_time), meastime) for meastime in step.meastimes]

        points = []
        
        # Apply penalties to the figure of merit and find the optimal next point
        for i in range(self.npoints):
            # calculate movement time penalty (and time penalty to switch models if applicable)
            switch_time_penalty = [0.0 if j == self.curmodel else self.switch_time_penalty for j in range(self.nmodels)]
            movepenalty = [meastime / (meastime + self.instrument.movetime(x) + pen) for x, meastime, pen in zip(self.x, min_meas_times, switch_time_penalty)]

            # all models incur switch penalty except the current one
            spenalty = [1.0 if j == self.curmodel else self.switch_penalty for j in range(self.nmodels)]
            step.scaled_foms = [fom * movepen / pen for fom, pen, movepen in zip(step.foms, spenalty, movepenalty)]

            if False:
                for fom, scaled_fom, x in zip(step.foms, step.scaled_foms, self.x):
                    p = plt.semilogy(x, fom, '--')
                    plt.semilogy(x, scaled_fom, '-', color=p[0].get_color())
                
                plt.show()

            newpoint = self.select_new_point(step, start=i)

            # newpoint can be None if not enough maxima in the fom are found. In this case
            # stop looking for new points
            if newpoint is not None:
                newpoint.movet = self.instrument.movetime(newpoint.x)[0]
                points.append(newpoint)
                print('New data point:\t' + repr(newpoint))

                # Once a new point is added, update the current model so model switching
                # penalties can be reapplied correctly
                self.curmodel = newpoint.model

                # "move" instrument to new location for calculating the next movement penalty
                self.instrument.x = newpoint.x
            else:
                break
        
        self.add_step(points)

    def add_step(self, points, use=True):
        # Adds a set of DataPoint objects as a new ExperimentStep
        self.steps.append(ExperimentStep(points, use=use))

    def _marginalization_efficiency(self, Qth, qprof, points):
        """ Calculate the marginalization efficiency: the fraction of uncertainty in R(Q) that
            arises from the marginal parameters"""

        # define parameter numbers to select
        marg_points = points[:,self.sel]

        # Calculate the Jacobian matrix from a linear regression of the q-profiles against
        # the parameters. Do this for all parameters and selected parameters.
        # TODO: Calculate this once and then select the appropriate parameters
        reg = LinearRegression(fit_intercept=True)
        reg.fit(points/np.std(points, axis=0), qprof/np.std(qprof, axis=0))
        reg_marg = LinearRegression(fit_intercept=True)
        reg_marg.fit(marg_points[:,:]/np.std(marg_points[:,:], axis=0), qprof/np.std(qprof, axis=0))
        J = reg.coef_.T
        J_marg = reg_marg.coef_.T

        # Calculate the covariance matrices for all and selected parameters
        # TODO: Calculate this once and then select the appropriate parameters
        covX = np.cov((points/np.std(points, axis=0)).T)
        covX = np.array(covX, ndmin=2)
        covX_marg = np.cov((marg_points/np.std(marg_points, axis=0)).T)
        covX_marg = np.array(covX_marg, ndmin=2)

        # Calculate the fraction of the total uncertainty that can be accounted for
        # by the selected parameters
        df2s = np.zeros_like(Qth)
        df2s_marg = np.zeros_like(Qth)    
        for j in range(len(Qth)):
            Jj = J[:,j][:,None]
            df2s[j] = np.squeeze(Jj.T @ covX @ Jj)

            Jj = J_marg[:,j][:,None]
            df2s_marg[j] = np.squeeze(Jj.T @ covX_marg @ Jj)

        return df2s, df2s_marg, df2s_marg / df2s

    def calc_foms_cov(self, step):
        """Calculate figures of merit for each model, using a Jacobian/covariance matrix approach

        Inputs:
        step -- the step to analyze. Assumes that the step has been fit so
                step.draw and step.qprofs exist

        Returns:
        foms -- list of figures of merit (ndarrays), one for each model in self.problem
        meastimes -- list of suggested measurement times at each Q value, one for each model
        """

        foms = list()
        meas_times = list()
        # Cycle through models, with model-specific x, Q, calculated q profiles, and measurement background level
        for mnum, (m, xs, Qth, qprof, qbkg) in enumerate(zip(self.models, self.x, self.measQ, step.qprofs, self.meas_bkg)):

            # get the incident intensity for all x values
            incident_neutrons = self.instrument.intensity(xs)

            # define signal to background. For now, this is just a scaling factor on the effective rate
            # reference: Hoogerheide et al. J Appl. Cryst. 2022
            sbr = qprof / qbkg
            refl = np.mean(qprof/(1+2/sbr), axis=0)
            refl = np.maximum(refl, np.zeros_like(refl))

            # q-dependent noise. Use the minimum of the actual spread in Q and the expected spread from the nearest points. 
            # This can get stuck if the spread changes too rapidly, so dR is smoothed by dQ.
            smoothed_dR = list()
            Q, dR = m.fitness.probe.Q, m.fitness.probe.dR
            for Qi, dQi in zip(Q, m.fitness.probe.dQ):
                kernel = 1./(2*np.pi*dQi**2) * np.exp(-(Q - Qi) ** 2 / (2 * dQi **2))
                smoothed_dR.append(np.sum(dR * kernel / dR ** 2) / np.sum(kernel / dR ** 2))

            # TODO: Is this really the right thing to do? Should probably just be the actual spread; the problem is that if
            # the spread doesn't constrain the variables very much, then we just keep measuring at the same point over and over.
            minstd = np.min(np.vstack((np.std(qprof, axis=0), np.interp(Qth, m.fitness.probe.Q, smoothed_dR))), axis=0)
            normrefl = refl * (minstd/np.mean(qprof, axis=0))**4

            # Calculate marginalization efficiency
            _, _, marg_eff = self._marginalization_efficiency(Qth, qprof, step.draw.points)

            qfom_norm = marg_eff*normrefl

            # Calculate figures of merit and proposed measurement times
            fom = list()
            meas_time = list()
            old_meas_time = list()
            for x, intens in zip(xs, incident_neutrons):
                q = self.instrument.x2q(x)
                #xrefl = intens * np.interp(q, Qth, refl * (minstd/np.mean(qprof, axis=0))**2)
                # TODO: check this. Should it be the average of xrefl, or the sum?
                #old_meas_time.append(np.mean((1-self.eta) / (self.eta**2 * xrefl)))

                # calculate the figure of merit
                fom.append(np.sum(intens * np.interp(q, Qth, qfom_norm)))

            fom = np.array(fom)

            # calculate the effective number of detectors. If only a few are lighting up, this will be close to 1,
            # otherwise, if all the intensities are about the same, this will be close to the number of detectors
            # TODO: Calculate this correctly. For CANDOR in monochromatic mode, this might break because the effective
            # number of detectors is not 54, but 2 or 3. So just blindly taking all detectors is probably not correct.
            # An appropriately weighted sum would probably be better.
            effective_detectors = float(incident_neutrons.shape[1])
            #print(f'effective detectors: {effective_detectors}')

            for x, intens, ifom in zip(xs, incident_neutrons, fom):
                q = self.instrument.x2q(x)
                xrefl = intens * np.interp(q, Qth, refl * (minstd/np.mean(qprof, axis=0))**2)

                # original calculation
                old_meas_time.append(np.mean((1-self.eta) / (self.eta**2 * xrefl)))

                # automatic eta determination
                # sqrt is because the fom is proportional to (sigma ** 2) ** 2.
                # Use self.eta as an upper limit to avoid negative 1 - eta
                # Division by effective number of detectors accounts for simultaneous detection in
                # multiple detectors
                eta = min((np.mean(fom) / ifom) ** 0.5, self.eta)
                eta = 1 - (1 - eta) / effective_detectors
                meas_time.append(np.mean((1 - eta) / (eta ** 2 * xrefl)))

            foms.append(fom)
            meas_times.append(np.array(meas_time))
            #print(np.vstack((xs, old_meas_time, meas_time)).T)

        return foms, meas_times

    def _dHdt(self, pts, qprofs, incident_neutrons, n_steps=1):
        """ Calculate rate of change of entropy (dH/dt) for measuring at a given Q point
        
        Inputs:
        Qth -- the Q values represented by each Q profile (length nQ array)
        pts -- the parameter samples underlying each Q profile (nprof x npar array)
        qprofs -- Q profiles (nprof x nQ array)
        incident_neutrons -- intensity ()
        n_steps -- (ignored for now) optional parameter defaults to 1: number of forward steps to look
        
        Returns:
        ????
        """

        eta = 0.68  # measure to 1 standard deviation
        #for _ in range(n_steps):
        dH = list()
        dHdt = list()
        ts = list()
        goodidxs = list()
        H0 = ar.calc_entropy(pts, select_pars=self.sel)
        # cycle through all q values)
        for idx, intens in enumerate(incident_neutrons):
            
            # extract distribution of q profiles
            iqs = qprofs[:,idx]
            
            # calculate median and (eta x 100) CI
            med, ci = credible_interval(iqs, (0, eta))

            # estimate neutron flux (intens x med) and existing uncertainty
            xrefl = (intens * med * (np.diff(ci) / med) ** 2)[0]

            # estimate measurement time to equal standard deviation of a gaussian with this CI
            meastime = (1-eta) / (eta**2 * xrefl)

            # select only curves within the confidence interval
            # TODO: Use change in llf to resample curves, weighting by llf?
            crit = ((iqs > ci[0]) & (iqs < ci[1]))

            # find indices of selected curves
            goodidxs.append(np.arange(len(crit))[crit])

            # select corresponding parameter values
            newpts = pts[crit]

            # calculate marginalized entropy of selected points
            iH = ar.calc_entropy(newpts, select_pars=self.sel)

            # calculate differential entropy from initial state
            dH.append(H0 - iH)
            dHdt.append((H0 - iH) / meastime)
            ts.append(meastime)
        
        # selection (vestigial if n_steps == 1, used if doing multiple forecasting)
        maxidx = np.where(dHdt==np.max(dHdt))[0][0]
        goodidx = goodidxs[maxidx]
        pts = pts[goodidx,:]
        qprofs = qprofs[goodidx,:]

        return np.array(dHdt), np.array(ts)

    def calc_foms(self, step):
        """Calculate figures of merit for each model, using sampled R(Q) to predict multiple steps ahead

        Inputs:
        step -- the step to analyze. Assumes that the step has been fit so
                step.draw and step.qprofs exist

        Returns:
        foms -- list of figures of merit (ndarrays), one for each model in self.problem
        meastimes -- list of suggested measurement times at each Q value, one for each model
        """

        foms = list()
        meas_times = list()
        pts = step.draw.points
        # Cycle through models, with model-specific x, Q, calculated q profiles, and measurement background level
        for mnum, (m, xs, Qth, qprof, qbkg) in enumerate(zip(self.models, self.x, self.measQ, step.qprofs, self.meas_bkg)):

            # get the incident intensity for all x values
            incident_neutrons = self.instrument.intensity(xs)

            # define signal to background. For now, this is just a scaling factor on the effective rate
            # reference: Hoogerheide et al. J Appl. Cryst. 2022
            sbr = qprof / qbkg
            refl = qprof/(1+2/sbr)
            refl = np.clip(refl, a_min=0, a_max=None)

            interp_refl = interp1d(Qth, refl, axis=1)

            # Calculate figures of merit and proposed measurement times
            fom = list()
            meas_time = list()
            for x, intens in zip(xs, incident_neutrons):
                q = self.instrument.x2q(x)
                #xrefl = intens * np.interp(q, Qth, refl * (minstd/np.mean(qprof, axis=0))**2)
                # TODO: check this. Should it be the average of xrefl, or the sum?
                #old_meas_time.append(np.mean((1-self.eta) / (self.eta**2 * xrefl)))

                # calculate the figure of merit
                xqprof = np.array(interp_refl(q), ndmin=2).T

                idHdt, its = self._dHdt(pts, xqprof, intens)
                fom.append(np.sum(idHdt))
                meas_time.append(np.mean(its))

            foms.append(fom)
            meas_times.append(np.array(meas_time))

        return foms, meas_times

    def select_new_point(self, step, start=0):
        """Find a single new point to measure from the figure of merit
        
        Inputs:
        step -- the step to analyze. Assumes that step has foms (figures of merit) and
                measurement times precalculated
        start -- index of figure of merit maxima to begin searching. Allows multiple points
                to be identified by calling select_new_point sequentially, incrementing "start"

        Returns:
        a DataPoint object containing the new point with simulated data
        """

        # TODO: Implement a more random algorithm. One idea is to define a partition function
        #       Z = np.exp(fom / np.mean(fom)) - 1. The fom is then related to ln(Z(x)). Points are chosen
        #       using np.random.choice(x, size=self.npoints, p=Z/np.sum(Z)).
        #       I think that penalties will have to be applied differently, potentially directly to Z.

        # finds a single point to measure
        maxQs = []
        maxidxs = []
        maxfoms = []

        # find maximum figures of merit in each model

        for fom, Qth in zip(step.scaled_foms, self.measQ):
            
            # a. calculate whether gradient is > 0
            dfom = np.sign(np.diff(np.append(np.insert(fom, 0, 0),0))) < 0
            # b. find zero crossings
            xings = np.diff(dfom.astype(float))
            maxidx = np.where(xings>0)[0]
            maxfoms.append(fom[maxidx])
            maxQs.append(Qth[maxidx])
            maxidxs.append(maxidx)

        # condition the maximum indices
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
            calcR = ar.calc_expected_R(self.calcmodels[mnum].fitness, T, dT, L, dL, oversampling=self.oversampling, resolution='normal')
            #print('expected R:', calcR)
            incident_neutrons = self.instrument.intensity(newx) * new_meastime
            N, Nbkg, Ninc = ar.sim_data_N(calcR, incident_neutrons, resid_bkg=self.resid_bkg[mnum], meas_bkg=self.meas_bkg[mnum])
            
            t = max(self.min_meas_time, new_meastime)

            return DataPoint(newx, t, mnum, (T, dT, L, dL, N[0], Nbkg[0], Ninc[0]), merit=maxfom)
        
        else:

            return None

    def save(self, fn):
        """Save a pickled version of the experiment"""

        for step in self.steps[:-2]:
            step.draw.state = None

        with open(fn, 'wb') as f:
            dill.dump(self, f, recurse=True)

    @classmethod
    def load(cls, fn):
        """ Load a pickled version of the experiment
        
        Usage: <variable> = SimReflExperiment.load(<filename>)
        """

        with open(fn, 'rb') as f:
            return dill.load(f)

class SimReflExperimentControl(SimReflExperiment):
    r"""Control experiment with even or scaled distribution of count times
    
    Subclasses SimReflExperiment.

    New input:
    model_weights -- a vector of weights, with length equal to number of problems
                    in self.problem. Scaled by the sum.
    NOTE: a fixed $Q^2$ weighting is additionally applied to each Q point
    """

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
        r"""Overrides SimReflExperiment.take_step
        
        Generates a simulated reflectivity curve based on $Q^2$-scaled
        measurement times.
        """

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
                calcR = ar.calc_expected_R(self.calcmodels[mnum].fitness, T, dT, L, dL, oversampling=self.oversampling, resolution='normal')
            #print('expected R:', calcR)
                N, Nbkg, Ninc = ar.sim_data_N(calcR, intens.T * t, resid_bkg=resid_bkg, meas_bkg=meas_bkg)
                points.append(DataPoint(x, t, mnum, (T, dT, L, dL, N, Nbkg, Ninc), movet=self.instrument.movetime(x)[0]))
                self.instrument.x = x
        
        self.add_step(points)

def magik_intensity(Q, modelnum=None):
    """Give counts/second intensity as a function of Q
        for MAGIK.
        
    Obsolete function for back-compatibility with old code. Superceded by
    instrument.py
    """

    ps1 = np.array([ 1.35295366e+01, -9.99016840e-04])
    p_intens = np.array([ 5.56637543e+02,  7.27944632e+04,  2.13479802e+02, -4.37052050e+01])
    news1 = np.polyval(ps1, Q)
    incident_neutrons = np.polyval(p_intens, news1)

    return incident_neutrons

def _MP_calc_qprofile(problem_point_pair):
    """ Calculate q profiles based on a sample draw, for use with
        multiprocessing

        Adapted from refl1d.mapper
    """

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
    """Calculation function of q profiles using _MP_calc_qprofiles
    
    Inputs:
    calcproblem -- a bumps.BaseFitProblem or bumps.MultiFitProblem, prepopulated
                    with attributes:
                        calcQs (same as SimReflExperiment.measQ);
                        oversampling
                        resolution (either 'normal' or 'uniform', instrument-dependent)
    point -- parameter vector
    """
    
    mlist = [calcproblem] if hasattr(calcproblem, 'fitness') else list(calcproblem.models)
    newvars = [ar.gen_new_variables(Q) for Q in calcproblem.calcQs]
    qprof = list()
    for m, newvar in zip(mlist, newvars):
        calcproblem.setp(point)
        calcproblem.chisq_str()
        Rth = ar.calc_expected_R(m.fitness, *newvar, oversampling=calcproblem.oversampling, resolution=calcproblem.resolution)
        qprof.append(Rth)

    return qprof

def get_steps_time(steps, control=False):

    if not control:
        allt = np.cumsum([step.meastime() + step.movetime() for step in steps])
    else:
        # assume all movement was done only once
        allt = np.cumsum([step.meastime() for step in steps]) + np.array([step.movetime() for step in steps])

    return allt

def load_entropy(steps, control=False):

    allt = get_steps_time(steps, control)
    allH = [step.dH for step in steps]
    allH_marg = [step.dH_marg for step in steps]

    return allt, allH, allH_marg

def get_parameter_variance(steps, control=False):

    allt = get_steps_time(steps, control)
    allvars = np.array([np.var(step.draw.points, axis=0) for step in steps]).T

    return allt, allvars

def parameter_error_plot(exp, ctrl=None, fig=None, tscale='log', yscale='log', color=None):

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


def snapshot(exp, stepnumber, fig=None, power=4, tscale='log'):

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
        ar.plot_qprofiles(copy.copy(measQ), qprof, step.draw.logp, data=idata, ax=axtop, power=power)
        axtop.set_title(f'meas t = {steptimes[i]:0.0f} s\nmove t = {movetimes[i]:0.0f} s', fontsize='larger')
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

    fig.suptitle(f'measurement time = {sum(steptimes):0.0f} s\nmovement time = {sum(movetimes):0.0f} s', fontsize='larger', fontweight='bold')

    return fig, (axtops, axbots, axtopright, axbotright)

def makemovie(exp, outfilename, expctrl=None, fps=1, fmt='gif', power=4, tscale='log'):
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
