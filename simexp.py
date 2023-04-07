import numpy as np
import copy
import time
import dill
from typing import Tuple, Union, List

from bumps.fitters import ConsoleMonitor, _fill_defaults, StepMonitor
from bumps.initpop import generate
from bumps.mapper import MPMapper
from bumps.dream.state import MCMCDraw
from refl1d.names import FitProblem, Experiment
from refl1d.resolution import TL2Q
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from entropy import calc_entropy, calc_init_entropy, default_entropy_options
import autorefl as ar
import instrument

fit_options = {'pop': 10, 'burn': 1000, 'steps': 500, 'init': 'lhs', 'alpha': 0.001}

data_tuple = Tuple[Union[np.ndarray, list], Union[np.ndarray, list],
                                   Union[np.ndarray, list], Union[np.ndarray, list],
                                   Union[np.ndarray, list], Union[np.ndarray, list],
                                   Union[np.ndarray, list]]

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

    def __init__(self, x: float, meastime: float, modelnum: int,
                       data: data_tuple,
                       merit: Union[bool, None] = None,
                       movet: float = 0.0):
        self.model = modelnum
        self.t = meastime
        self.movet = movet
        self.merit = merit
        self.x = x
        self._data: data_tuple = None
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
    def data(self, newdata) -> None:
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

    def __init__(self, points: List[DataPoint], use=True) -> None:
        self.points = points
        self.H: Union[float, None] = None
        self.dH: Union[float, None] = None
        self.H_marg: Union[float, None] = None
        self.dH_marg: Union[float, None] = None
        self.foms: Union[List[np.ndarray], None] = None
        self.scaled_foms: Union[List[np.ndarray], None] = None
        self.meastimes: Union[List[np.ndarray], None] = None
        self.qprofs: Union[List[np.ndarray], None] = None
        self.qbkgs: Union[List[np.ndarray], None] = None
        self.best_logp: Union[float, None] = None
        self.final_chisq: Union[str, None] = None
        self.draw: Union[MCMCDraw, None] = None
        self.chain_pop: Union[np.ndarray, None] = None
        self.use: bool = use

    def getdata(self, attr: str, modelnum: int) -> list:
        # returns all data of type "attr" for a specific model
        if self.use:
            return [getattr(pt, attr) for pt in self.points if pt.model == modelnum]
        else:
            return []

    def meastime(self, modelnum: Union[int, None] = None) -> float:
        if modelnum is None:
            return sum([pt.t for pt in self.points])
        else:
            return sum([pt.t for pt in self.points if pt.model == modelnum])

    def movetime(self, modelnum: Union[int, None] = None) -> float:
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

    Inputs:
    problem -- a Refl1d FitProblem describing the reflectometry experiment. Multiple models (M)
                are supported. Required.
    Q -- numpy array or nested list of Q bins for reducing data. Can be either a single Q vector
        (applied to each model), or an M-element list of Q bins, one for each model. Required.
    instrument -- instrument definition based on the instrument.ReflectometerBase class; default
                   MAGIK
    eta -- confidence interval for measurement time determination; default 0.68
    npoints -- integer number of points to measure in each step; if > 1, forecasting is used to 
                determine subsequent points; default 1
    switch_penalty -- scaling factor on the figure of merit to switch models, i.e. applied only
                      to models that are not the current one; default 1.0 (no penalty)
    switch_time_penalty -- time required to switch models, i.e. applied only
                      to models that are not the current one; default 0.0 (no penalty)
    bestpars -- numpy array or list or None: best fit (ground truth) parameters (length P
                parameters). Used for simulating new data
    fit_options -- dictionary of Bumps fitter fit options; default {'pop': 10, 'burn': 1000,
                    'steps': 500, 'init': 'lhs', 'alpha': 0.001}
    entropy_options -- dictionary of entropy options; default entropy.default_entropy_options
    oversampling -- integer oversampling value for calculating Refl1D models. Default 11; should be
                    > 1 for accurate simulations.
    meas_bkg -- single float of list of M floats representing the measurement background level
                (unsubtracted) for each model.
    startmodel -- integer index of starting model; default 0.
    min_meas_time -- minimum measurement time (float); default 10.0 seconds
    select_pars -- selected parameters for entropy determination. None uses all parameters, otherwise
                    list of parameter indices
    """

    def __init__(self, problem: FitProblem,
                       Q: Union[np.ndarray, List[np.ndarray]],
                       instrument: instrument.ReflectometerBase = instrument.MAGIK(),
                       eta: float = 0.68,
                       npoints: int = 1,
                       switch_penalty: float = 1.0,
                       switch_time_penalty: float = 0.0,
                       bestpars: Union[np.ndarray, list, None] = None,
                       fit_options: dict = fit_options,
                       entropy_options: dict = default_entropy_options,
                       oversampling: int = 11,
                       meas_bkg: Union[float, List[float]] = 1e-6,
                       startmodel: int = 0,
                       min_meas_time: float = 10.0,
                       select_pars: Union[list, None] = None) -> None:
        
        self.attr_list = ['T', 'dT', 'L', 'dL', 'N', 'Nbkg', 'Ninc']

        # Load instrument
        self.instrument = instrument

        # Analysis options
        self.eta = eta
        self.npoints = int(npoints)
        self.switch_penalty = switch_penalty
        self.switch_time_penalty = switch_time_penalty
        self.min_meas_time = min_meas_time

        # Initialize the fit problem
        self.problem = problem
        models: List[Union[Experiment, FitProblem]] = [problem] if hasattr(problem, 'fitness') else list(problem.models)
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
        self.x: List[np.ndarray] = list()
        for Q in self.measQ:
            minx, maxx = self.instrument.qrange2xrange([min(Q), max(Q)])
            self.x.append(np.linspace(minx, maxx, len(Q), endpoint=True))

        # Create a copy of the problem for calculating the "true" reflectivity profiles
        self.npars = len(problem.getp())
        self.orgQ = [list(m.fitness.probe.Q) for m in models]
        calcmodel = copy.deepcopy(problem)
        self.calcmodels: List[Union[Experiment, FitProblem]] = [calcmodel] if hasattr(calcmodel, 'fitness') else list(calcmodel.models)
        if bestpars is not None:
            calcmodel.setp(bestpars)

        # deal with inherent measurement background
        if not isinstance(meas_bkg, (list, np.ndarray)):
            self.meas_bkg: np.ndarray = np.full(self.nmodels, meas_bkg)
        else:
            self.meas_bkg: np.ndarray = np.array(meas_bkg)

        # add residual background
        self.resid_bkg: np.ndarray = np.array([c.fitness.probe.background.value for c in self.calcmodels])

        # these are not used
        self.newmodels = [m.fitness for m in models]
        self.par_scale: np.ndarray = np.diff(problem.bounds(), axis=0)

        # set and condition selected parameters for marginalization; use all parameters
        # if none are specified
        if select_pars is None:
            self.sel: np.ndarray = np.arange(self.npars)
        else:
            self.sel: np.ndarray = np.array(select_pars, ndmin=1)

        # initialize objects required for fitting
        self.fit_options = fit_options
        self.steps: List[ExperimentStep] = []
        self.restart_pop: Union[np.ndarray, None] = None

# calculate initial MVN entropy in the problem
        self.entropy_options = {**default_entropy_options, **entropy_options}
        self.thinning = int(self.fit_options['steps']*0.05)
        self.init_entropy, _, _ = calc_init_entropy(problem, pop=fit_options['pop'] * fit_options['steps'] / self.thinning, options=entropy_options)
        self.init_entropy_marg, _, _ = calc_init_entropy(problem, select_pars=select_pars, pop=fit_options['pop'] * fit_options['steps'] / self.thinning, options=entropy_options)

    def get_all_points(self, modelnum: Union[int, None]) -> List[DataPoint]:
        # returns all data points associated with model with index modelnum
        return [pt for step in self.steps for pt in step.points if pt.model == modelnum]

    def getdata(self, attr: str, modelnum: Union[int, None]) -> list:
        # returns all data of type "attr" for a specific model
        return [getattr(pt, attr) for pt in self.get_all_points(modelnum)]

    def compile_datapoints(self, Qbasis, points) -> Tuple:
        # bins all of the data from a list "points" onto a Q-space "Qbasis"
        idata = [[val for pt in points for val in getattr(pt, attr)] for attr in self.attr_list]

        return ar.compile_data_N(Qbasis, *idata)

    def add_initial_step(self, dRoR=10.0) -> None:
        """ Generate initial data set. This is only necessary because of the requirement that
            dof > 0 in Refl1D (not strictly required for DREAM fit)
            
            Inputs:
            dRoR -- target uncertainty relative to the average of the reflectivity, default 10.0;
                    determines the "measurement time" for the initial data set. This should be
                    > 3 so as not to constrain the parameters before collecting any real data.
        """

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

    def update_models(self) -> None:
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

    def calc_qprofiles(self, drawpoints: np.ndarray, mappercalc) -> np.ndarray:
        # q-profile calculator using multiprocessing for speed
        # this version is limited to calculating profiles with measQ, cannot be used with initial calculation
        res = mappercalc(drawpoints)

        # condition output of mappercalc to a list of q-profiles for each model
        qprofs = list()
        for i in range(self.nmodels):
            qprofs.append(np.array([r[i] for r in res]))

        return qprofs

    def fit_step(self, outfid=None) -> None:
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
        step.draw = fitter.state.draw(thin=self.thinning)
        step.best_logp = fitter.state.best()[1]
        self.problem.setp(fitter.state.best()[0])
        step.final_chisq = self.problem.chisq_str()
        step.H, _, _ = calc_entropy(step.draw.points, select_pars=None, options=self.entropy_options)
        step.dH = self.init_entropy - step.H
        step.H_marg, _, _ = calc_entropy(step.draw.points, select_pars=self.sel, options=self.entropy_options)
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

    def take_step(self, allow_repeat=True) -> None:
        """Analyze the last fitted step and add the next one
        
        Procedure:
            1. Calculate the figures of merit
            2. Identify the next self.npoints data points
                to simulate/measure
            (1 and 2 are currently done in _fom_from_draw)
            3. Simulate the new data points
            4. Add a new step for fitting.

        Inputs:
            allow_repeat -- toggles whether to allow measurement at the same point over and over;
                            default True. (Can cause issues if MCMC fit isn't converged in
                            high-gradient areas, e.g. around the critical edge)
        """

        # Focus on the last step
        step = self.steps[-1]
        
        # Calculate figures of merit and proposed measurement times with forecasting
        print('Calculating figures of merit:')
        init_time = time.time()
        pts = step.draw.points[:, self.sel]

        # can scale parameters for entropy calculations. Helps with GMM.
        # TODO: check that this works. Does it need to be implemented in entropy.calc_entropy?
        if self.entropy_options['scale']:
            pts = copy.copy(pts) / self.par_scale[:, self.sel]

        foms, meastimes, _, newpoints = self._fom_from_draw(pts, step.qprofs, select_ci_level=0.68, meas_ci_level=self.eta, n_forecast=self.npoints, allow_repeat=allow_repeat)
        print('Total figure of merit calculation time: %f' % (time.time() - init_time))

        # populate step foms
        # TODO: current analysis code can't handle multiple foms, could pass all of them in here
        step.foms, step.meastimes = foms[0], meastimes[0]

        # Determine next measurement point(s).
        # Number of points to be used is determined from n_forecast (self.npoints)
        # NOTE: At some point this could be turned into an asynchronous "point queue"; in this case the following loop will have to be
        #       over self.npoints
        points = []
        for pt, fom in zip(newpoints, foms):
            mnum, idx, newx, new_meastime = pt
            newpoint = self._generate_new_point(mnum, newx, new_meastime, fom[mnum][idx])
            newpoint.movet = self.instrument.movetime(newpoint.x)[0]
            points.append(newpoint)
            print('New data point:\t' + repr(newpoint))

            # Once a new point is added, update the current model so model switching
            # penalties can be reapplied correctly
            self.curmodel = newpoint.model

            # "move" instrument to new location for calculating the next movement penalty
            self.instrument.x = newpoint.x
        
        self.add_step(points)

    def add_step(self, points, use=True) -> None:
        """Adds a set of DataPoint objects as a new ExperimentStep

        Inputs:
            points: list of DataPoints for the step\
            use: boolean toggle for whether to use this step for plotting. Default true.
        """
        self.steps.append(ExperimentStep(points, use=use))

    def _apply_fom_penalties(self, foms, curmodel=None) -> List[np.ndarray]:
        """
        Applies any penalties that scale the figures of merit directly

        Inputs:
        foms -- list of of figure of merits, one for each model
        curmodel -- integer index of current model

        Returns:
        scaled_foms -- scaled list of figures of merit (list of numpy arrays)
        """

        if curmodel is None:
            curmodel = self.curmodel

        # Calculate switching penalty
        spenalty = [1.0 if j == curmodel else self.switch_penalty for j in range(self.nmodels)]

        # Perform scaling
        scaled_foms = [fom  / pen for fom, pen in zip(foms, spenalty)]

        return scaled_foms

    def _apply_time_penalties(self, foms, meastimes, curmodel=None) -> List[np.ndarray]:
        """
        Applies any penalties that act to increase the measurement time, e.g. movement penalties or model switch time penalities
        NOTE: uses current state of the instrument (self.instrument.x).

        Inputs:
        foms -- list of of figure of merits, one for each model
        meastimes -- list of proposed measurement time vectors, one for each model
        curmodel -- integer index of current model

        Returns:
        scaled_foms -- scaled list of figures of merit (list of numpy arrays)
        """

        if curmodel is None:
            curmodel = self.curmodel

        # Apply minimum to proposed measurement times
        min_meas_times = [np.maximum(np.full_like(meastime, self.min_meas_time), meastime) for meastime in meastimes]

        # Calculate time penalty to switch models
        switch_time_penalty = [0.0 if j == curmodel else self.switch_time_penalty for j in range(self.nmodels)]

        # Add all movement time penalties together.
        movepenalty = [meastime / (meastime + self.instrument.movetime(x) + pen) for x, meastime, pen in zip(self.x, min_meas_times, switch_time_penalty)]

        # Perform scaling
        scaled_foms = [fom * movepen for fom,movepen in zip(foms, movepenalty)]

        return scaled_foms

    def _fom_from_draw(self, pts: np.ndarray,
                        qprofs: List[np.ndarray],
                        select_ci_level: float = 0.68,
                        meas_ci_level: float = 0.68,
                        n_forecast: int = 1,
                        allow_repeat: bool = True) -> Tuple[List[List[np.ndarray]],
                                                            List[List[np.ndarray]],
                                                            List[float],
                                                            List[Tuple[int, int, float, float]]]:
        """ Calculate figure of merit from a set of draw points and associated q profiles
        
            Inputs:
            pts -- draw points. Should be already selected for marginalized paramters
            qprofs -- list of q profiles, one for each model of size <number of samples in pts> x <number of measQ values>
            select_ci_level -- confidence interval level to use for selection (default 0.68)
            meas_ci_level -- confidence interval level to target for measurement (default 0.68, typically use self.eta)
            n_forecast -- number of forecast steps to take (default 1)
            allow_repeat -- whether or not the same point can be measured twice in a row. Turn off to improve stability.

            Returns:
            all_foms -- list (one for each forecast step) of lists of figures of merit (one for each model)
            all_meastimes -- list (one for each forecast step) of lists of proposed measurement times (one for each model)
            all_H0 -- list (one for each forecast step) of maximum entropy (not entropy change) before that step
            all_new -- list of forecasted optimal data points (one for each forecast step). Each element in the list is a list
                        of properties of the new point with format: [<model number>, <x index>, <x value>, <measurement time>])
        """

        """shape definitions:
            X -- number of x values in xs
            D -- number of detectors
            N -- number of samples
            P -- number of marginalized parameters"""

        # Cycle through models, with model-specific x, Q, calculated q profiles, and measurement background level
        # Populate q vectors, interpolated q profiles (slow), and intensities
        intensities = list()
        intens_shapes = list()
        qs = list()
        xqprofs = list()
        init_time = time.time()
        for xs, Qth, qprof, qbkg in zip(self.x, self.measQ, qprofs, self.meas_bkg):

            # get the incident intensity and q values for all x values (should have same shape X x D).
            # flattened dimension is XD
            incident_neutrons = self.instrument.intensity(xs)
            init_shape = incident_neutrons.shape
            incident_neutrons = incident_neutrons.flatten()
            q = self.instrument.x2q(xs).flatten()

            # define signal to background. For now, this is just a scaling factor on the effective rate
            # reference: Hoogerheide et al. J Appl. Cryst. 2022
            sbr = qprof / qbkg
            refl = qprof/(1+2/sbr)
            refl = np.clip(refl, a_min=0, a_max=None)

            # perform interpolation. xqprof should have shape N x XD. This is a slow step (and should only be done once)
            interp_refl = interp1d(Qth, refl, axis=1, fill_value=(refl[:,0], refl[:,-1]), bounds_error=False)
            xqprof = np.array(interp_refl(q))

            intensities.append(incident_neutrons)
            intens_shapes.append(init_shape)
            qs.append(q)
            xqprofs.append(xqprof)

        print(f'Forecast setup time: {time.time() - init_time}')

        all_foms = list()
        all_meas_times = list()
        all_H0 = list()
        all_new = list()
        org_curmodel = self.curmodel
        org_x = self.instrument.x

        """For each stage of the forecast, go through:
            1. Calculate the foms
            2. Select the new points
            3. Repeat
        """
        for i in range(n_forecast):
            init_time = time.time()
            Hlist = list()
            foms = list()
            meas_times = list()
            #newidxs_select = list()
            newidxs_meas = list()
            newxqprofs = list()
            N, P = pts.shape
            minci_sel, maxci_sel =  int(np.floor(N * (1 - select_ci_level) / 2)), int(np.ceil(N * (1 + select_ci_level) / 2))
            minci_meas, maxci_meas =  int(np.floor(N * (1 - meas_ci_level) / 2)), int(np.ceil(N * (1 + meas_ci_level) / 2))
            H0, _, predictor = calc_entropy(pts, select_pars=None, options=self.entropy_options, predictor=None)   # already marginalized!!
            if predictor is not None:
                predictor.warm_start = True

            all_H0.append(H0)
            # cycle though models
            for incident_neutrons, init_shape, q, xqprof in zip(intensities, intens_shapes, qs, xqprofs):

                #init_time2a = time.time()
                # TODO: Shouldn't these already be sorted by the second step?
                idxs = np.argsort(xqprof, axis=0)
                #print(f'Sort time: {time.time() - init_time2a}')
                #print(idxs.shape)

                # Select new points and indices in CI. Now has dimension M x XD X P
                A = np.take_along_axis(pts[:, None, :], idxs[:, :, None], axis=0)[minci_sel:maxci_sel]
                
                #init_time2a = time.time()
                # calculate new index arrays and xqprof values
                # this also works: meas_sigma = 0.5*np.diff(np.take_along_axis(xqprof, idxs[[minci, maxci],:], axis=0), axis=0)
                newidx = idxs[minci_meas:maxci_meas]
                meas_xqprof = np.take_along_axis(xqprof, newidx, axis=0)#[minci:maxci]
                meas_sigma = 0.5 * (np.max(meas_xqprof, axis=0) - np.min(meas_xqprof, axis=0))
                sel_xqprof = np.take_along_axis(xqprof, idxs[minci_sel:maxci_sel], axis=0)#[minci:maxci]
                sel_sigma = 0.5 * (np.max(sel_xqprof, axis=0) - np.min(sel_xqprof, axis=0))

                #print(f'Sel calc time: {time.time() - init_time2a}')
                
                #sel_sigma = 0.5 * np.diff(np.take_along_axis(xqprof, idxs[[minci_sel, maxci_sel],:], axis=0), axis=0)
                #meas_sigma = 0.5 * np.diff(np.take_along_axis(xqprof, idxs[[minci_meas, maxci_meas],:], axis=0), axis=0)

                init_time2 = time.time()

                # Condition shape (now has dimension M X P X XD)
                A = np.moveaxis(A, -1, 1)
                Hs, _, predictor = calc_entropy(A, None, options=self.entropy_options, predictor=predictor)

                # Calculate measurement times (shape XD)
                med = np.median(xqprof, axis=0)
                xrefl_sel = (incident_neutrons * med * (sel_sigma / med) ** 2)
                xrefl_meas = (incident_neutrons * med * (meas_sigma / med) ** 2)
                meastime_sel = 1.0 / xrefl_sel
                meastime_meas = 1.0 / xrefl_meas

                # apply min measurement time (turn this off initially to test operation)
                #meastime = np.maximum(np.full_like(meastime, self.min_meas_time), meastime)

                # figure of merit is dHdt (reshaped to X x D)
                dHdt = (H0 - Hs) / meastime_sel
                dHdt = np.reshape(dHdt, init_shape)

                # calculate fom and average time (shape X)
                fom = np.sum(dHdt, axis=1)
                meas_time = 1./ np.sum(1./np.reshape(meastime_meas, init_shape), axis=1)

                Hlist.append(Hs)
                foms.append(fom)
                meas_times.append(meas_time)
                newxqprofs.append(meas_xqprof)
                newidxs_meas.append(newidx)
                
            # populate higher-level lists
            all_foms.append(foms)
            all_meas_times.append(meas_times)

            # apply penalties
            scaled_foms = self._apply_fom_penalties(foms, curmodel=self.curmodel)
            scaled_foms = self._apply_time_penalties(scaled_foms, meas_times, curmodel=self.curmodel)

            # remove current point from contention if allow_repeat is False
            if (not allow_repeat) & (self.instrument.x is not None):
                curidx = np.where(self.x[self.curmodel]==self.instrument.x)[0][0]
                scaled_foms[self.curmodel][curidx] = 0.0

            # perform point selection
            top_n = self._find_fom_maxima(scaled_foms, start=0)
            #print(top_n)
            if top_n is not None:
                _, mnum, idx = top_n
                newx = self.x[mnum][idx]
                new_meastime = max(meas_times[mnum][idx], self.min_meas_time)
                
                all_new.append([mnum, idx, newx, new_meastime])
            else:
                break

            # apply point selection
            self.instrument.x = newx
            self.curmodel = mnum

            # choose new points. This is not straightforward if there is more than one detector, because
            # each point in XD may choose a different detector. We will choose without replacement by frequency.
            # idx_array has shape M x D
            idx_array = newidxs_meas[mnum].reshape(-1, *intens_shapes[mnum])[:, idx, :]
            #print(idx_array.shape)
            if idx_array.shape[1] == 1:
                # straightforward case, with 1 detector
                chosen = np.squeeze(idx_array)
            else:
                # select those that appear most frequently
                #print(idx_array.shape)
                freq = np.bincount(idx_array.flatten(), minlength=len(pts))
                freqsort = np.argsort(freq)
                chosen = freqsort[-idx_array.shape[0]:]
                
            newpts = pts[chosen]
            newxqprofs = [xqprof[chosen] for xqprof in xqprofs]

            # set up next iteration
            xqprofs = newxqprofs
            pts = newpts

            print(f'Forecast step {i}:\tNumber of samples: {N}\tCalculation time: {time.time() - init_time}')

        # reset instrument state
        self.instrument.x = org_x
        self.curmodel = org_curmodel

        return all_foms, all_meas_times, all_H0, all_new

    def _find_fom_maxima(self, scaled_foms: List[np.ndarray],
                         start: int = 0) -> List[Tuple[float, int, int]]:
        """Finds all maxima in the figure of merit, including the end points
        
            Inputs:
            scaled_foms -- figures of merit. They don't have to be scaled, but it should be the "final"
                            FOM with any penalties already applied
            start -- index of the first peak to select. Defaults to zero (start with the highest).

            Returns:
            top_n -- sorted list 

        """

        # TODO: Implement a more random algorithm (probably best appplied in a different function 
        #       to the maxima themselves). One idea is to define a partition function
        #       Z = np.exp(fom / np.mean(fom)) - 1. The fom is then related to ln(Z(x)). Points are chosen
        #       using np.random.choice(x, size=self.npoints, p=Z/np.sum(Z)).
        #       I think that penalties will have to be applied differently, potentially directly to Z.

        # finds a single point to measure
        maxQs = []
        maxidxs = []
        maxfoms = []

        # find maximum figures of merit in each model

        for fom, Qth in zip(scaled_foms, self.measQ):
            
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

        # returns sorted list of lists, each with entries [max fom value, model number, measQ index]
        return top_n

    def _generate_new_point(self, mnum: int,
                                  newx: float,
                                  new_meastime: float,
                                  maxfom: Union[float, None] = None) -> DataPoint:
        """ Generates a new data point with simulated data from the specified x
            position, model number, and measurement time
            
            Inputs:
            mnum -- the model number of the new point
            newx -- the x position of the new point
            new_meastime -- the measurement time
            maxfom -- the maximum of the figure of merit. Only used for record-keeping

            Returns a single DataPoint object
        """
        
        T = self.instrument.T(newx)[0]
        dT = self.instrument.dT(newx)[0]
        L = self.instrument.L(newx)[0]
        dL = self.instrument.dL(newx)[0]

        # for simulating data, need to subtract theta_offset from calculation models
        # not all probes have theta_offset, however
        # for now this is turned off. 
        if False:
            try:
                to_calc = self.calcmodels[mnum].fitness.probe.theta_offset.value
            except AttributeError:
                to_calc = 0.0

        calcR = ar.calc_expected_R(self.calcmodels[mnum].fitness, T, dT, L, dL, oversampling=self.oversampling, resolution='normal')
        #print('expected R:', calcR)
        incident_neutrons = self.instrument.intensity(newx) * new_meastime
        N, Nbkg, Ninc = ar.sim_data_N(calcR, incident_neutrons, resid_bkg=self.resid_bkg[mnum], meas_bkg=self.meas_bkg[mnum])
        
        return DataPoint(newx, new_meastime, mnum, (T, dT, L, dL, N[0], Nbkg[0], Ninc[0]), merit=maxfom)

    def select_new_point(self, step: ExperimentStep, start: int = 0) -> Union[DataPoint, None]:
        """ (Deprecated) Find a single new point to measure from the figure of merit
        
        Inputs:
        step -- the step to analyze. Assumes that step has foms (figures of merit) and
                measurement times precalculated
        start -- index of figure of merit maxima to begin searching. Allows multiple points
                to be identified by calling select_new_point sequentially, incrementing "start"

        Returns:
        a DataPoint object containing the new point with simulated data
        """

        top_n = self._find_fom_maxima(step.scaled_foms, start=start)

        if len(top_n):
            # generate a DataPoint object with the maximum point
            _, mnum, idx = top_n
            maxfom = step.foms[mnum][idx]       # use unscaled version for plotting
            newx = self.x[mnum][idx]
            new_meastime = max(step.meastimes[mnum][idx], self.min_meas_time)

            return self._generate_new_point(mnum, newx, new_meastime, maxfom=maxfom)

        else:

            return None

    def save(self, fn) -> None:
        """Save a pickled version of the experiment"""

        for step in self.steps[:-2]:
            step.draw.state = None

        with open(fn, 'wb') as f:
            dill.dump(self, f, recurse=True)

    @classmethod
    def load(cls, fn) -> None:
        """ Load a pickled version of the experiment
        
        Usage: <variable> = SimReflExperiment.load(<filename>)
        """

        with open(fn, 'rb') as f:
            exp = dill.load(f)
        
        # for back compatibility
        if not hasattr(exp, 'entropy_options'):
            exp.entropy_options = default_entropy_options
        
        return exp

class SimReflExperimentControl(SimReflExperiment):
    r"""Control experiment with even or scaled distribution of count times
    
    Subclasses SimReflExperiment.

    Additional input:
    model_weights -- a vector of weights, with length equal to number of problems
                    in self.problem. Scaled by the sum.
    NOTE: an instrument-defined default weighting is additionally applied to each Q point
    """

    def __init__(self, problem: FitProblem,
                       Q: Union[np.ndarray, List[np.ndarray]],
                       model_weights: Union[List[float], None] = None,
                       instrument: instrument.ReflectometerBase = instrument.MAGIK(),
                       eta: float = 0.68,
                       npoints: int = 1,
                       switch_penalty: float = 1.0,
                       switch_time_penalty: float = 0.0,
                       bestpars: Union[np.ndarray, list, None] = None,
                       fit_options: dict = fit_options,
                       entropy_options: dict = default_entropy_options,
                       oversampling: int = 11,
                       meas_bkg: Union[float, List[float]] = 1e-6,
                       startmodel: int = 0,
                       min_meas_time: float = 10.0,
                       select_pars: Union[list, None] = None) -> None:
        super().__init__(problem, Q, instrument=instrument, eta=eta, npoints=npoints, switch_penalty=switch_penalty, switch_time_penalty=switch_time_penalty, bestpars=bestpars, fit_options=fit_options, entropy_options=entropy_options, oversampling=oversampling, meas_bkg=meas_bkg, startmodel=startmodel, min_meas_time=min_meas_time, select_pars=select_pars)

        if model_weights is None:
            model_weights = np.ones(self.nmodels)
        else:
            assert (len(model_weights) == self.nmodels), "weights must have same length as number of models"
        
        model_weights = np.array(model_weights) / np.sum(model_weights)

        self.meastimeweights = list()
        for x, weight in zip(self.x, model_weights):
            f = self.instrument.meastime(x, weight)
            self.meastimeweights.append(f)

    def take_step(self, total_time: float) -> None:
        r"""Overrides SimReflExperiment.take_step
        
        Generates a simulated reflectivity curve based on weighted / scaled
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
