"""Tools for data simulation, binning, and analysis"""

from typing import Tuple
import numpy as np
from bumps.fitters import DreamFit, _fill_defaults
from scipy.stats import poisson
from reflred.candor import edges, _rebin_bank
from reflred.refldata import ReflData, Sample, Detector, Monochromator
import dataflow.lib.err1d as err1d

def sim_data_N(R, incident_neutrons, resid_bkg=0, meas_bkg=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate specular, background, and intensity counts from a Poisson distribution

    Args:
        R (list or ndarray): array of reflectivity values
        incident_neutrons (list or ndarray): array of incident counts, same shape as R
        resid_bkg (float, optional): Residual background (not accounted for by a model). Defaults to 0.
        meas_bkg (int, optional): Actual measurement background. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: size R vectors of specular, background, and intensity
    """
    R = np.array(R, ndmin=1)
    _bR = np.ones_like(R)*(meas_bkg - resid_bkg)*incident_neutrons
    _R = (R + meas_bkg - resid_bkg)*incident_neutrons
    N = poisson.rvs(_R, size=_R.shape)
    bN = poisson.rvs(_bR, size=_bR.shape)

    return N, bN, incident_neutrons

def calc_expected_R(fitness, T, dT, L, dL, oversampling=None, resolution='normal'):
    """Calculates the reflectivity from a model for a given set of T, dT, L, dL

    Args:
        fitness (Union[refl1d.names.Experiment, refl1d.names.FitProblem]): Refl1D model
        T (np.ndarray): incident angle (Theta) vector
        dT (np.ndarray): angular resolution (dTheta) vector, same size as T
        L (np.ndarray): wavelength (lambda) vector, same size as T
        dL (np.ndarray): wavelength resolution (dlambda) vector, same size as T
        oversampling (Union[int, None], optional): Integer number of oversampling points
            to use for the calculation. Defaults to None (no oversampling).
        resolution (str, optional): Resolution calculation, either 'uniform' or 'normal'. 'Uniform'
            is used for instruments like CANDOR with many Q values contributing to each bin.
             Defaults to 'normal'.

    Returns:
        np.ndarray: reflectivity calculation (same size as T)
    """
    # currently requires sorted values (by Q) because it returns sorted values.
    # this will need to be modified for CANDOR.
    fitness.probe._set_TLR(T, dT, L, dL, R=None, dR=None, dQ=None)
    fitness.probe.resolution = resolution
    if oversampling is not None:
        fitness.probe.oversample(oversampling)
    fitness.update()
    return fitness.reflectivity()[1]

def compile_data_N(Qbasis, T, dT, L, dL, Ntot, Nbkg, Ninc):
    """Uses Reductus modules to reduce reflectivity data

    Args:
        Qbasis (np.ndarray): Q bin centers
        T (np.ndarray): incident angle (Theta) vector
        dT (np.ndarray): angular resolution (dTheta) vector, same size as T
        L (np.ndarray): wavelength (lambda) vector, same size as T
        dL (np.ndarray): wavelength resolution (dlambda) vector, same size as T
        Ntot (np.ndarray): specular counts, same size as T
        Nbkg (np.ndarray): background counts, same size as T
        Ninc (np.ndarray): incident intensity counts, same size as T

    Returns:
        Tuple[
            np.ndarray: _Ti, incident angle (Theta) vector of Qz bin centers
            np.ndarray: _dT, angular resolution (dTheta) vector of Qz bin centers
            np.ndarray: _L, wavelength (lambda) vector of Qz bin vectors
            np.ndarray: _dL, wavelength resolution (dlambda) vector of Qz bin vectors
            np.ndarray: reduced and binned reflectivity, (specular - background) / intensity
            np.ndarray: error in reflectivity (standard deviation)
            np.ndarray: qz, Qz values of reflectivity
            np.ndarray: dq, Qz resolution
        ]
    """
    crit = np.round(Ninc)>0 # prevents zero-intensity data points
    T, dT, L, dL, Ntot, Nbkg, Ninc = [np.array(a)[crit] for a in (T, dT, L, dL, Ntot, Nbkg, Ninc)]
    if len(T):
        Ninc = np.round(Ninc)
        #print(T, dT, L, dL, Ntot, Nbkg, Ninc)
        q_edges = edges(Qbasis, extended=True)

        v = Ntot
        dv = np.sqrt(Ntot)
        vbkg = Nbkg
        dvbkg = np.sqrt(Nbkg)
        vinc = Ninc
        dvinc = np.sqrt(Ninc)
        normbase = 'time'
        wavelength_resolution = dL[:,None,None]  / L[:,None,None]
        spec = ReflData(monochromator=Monochromator(wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        sample=Sample(angle_x=T[:,None,None]),
                        angular_resolution=dT[:,None,None],
                        detector=Detector(angle_x=2*T[:,None,None], wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        _v=v[:,None,None], _dv=dv[:,None,None], Qz_basis='actual', normbase=normbase)
        bkg = ReflData(monochromator=Monochromator(wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        sample=Sample(angle_x=T[:,None,None]),
                        angular_resolution=dT[:,None,None],
                        detector=Detector(angle_x=2*T[:,None,None], wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        _v=vbkg[:, None, None], _dv=dvbkg[:,None, None], Qz_basis='actual', normbase=normbase)
        inc = ReflData(monochromator=Monochromator(wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        sample=Sample(angle_x=T[:,None,None]),
                        angular_resolution=dT[:,None,None],
                        detector=Detector(angle_x=2*T[:,None,None], wavelength=L[:,None,None], wavelength_resolution=wavelength_resolution),
                        _v=vinc[:, None, None], _dv=dvinc[:,None, None], Qz_basis='actual', normbase=normbase)

        # Bin values
        qz, dq, vspec, dvspec, _Ti, _dT, _L, _dL = _rebin_bank(spec, 0, q_edges, 'poisson')
        _, _, vbkg, dvbkg, _, _, _, _ = _rebin_bank(bkg, 0, q_edges, 'poisson')
        _, _, vinc, dvinc, _, _, _, _ = _rebin_bank(inc, 0, q_edges, 'poisson')

        # subtract background
        vsub, dvsub2 = err1d.sub(vspec, (dvspec**2 + (dvspec==0)), vbkg, (dvbkg**2 + (dvbkg==0)))

        # divide intensity
        vdiv, dvdiv2 = err1d.div(vsub, (dvsub2 + (dvsub2==0)), vinc, dvinc**2)

        return _Ti, _dT, _L, _dL, vdiv, np.sqrt(dvdiv2), qz, dq

    else:

        return tuple([np.array([]) for _ in range(8)])

class DreamFitPlus(DreamFit):
    """Modified DreamFit object that allows specification of the initial population in the solve method
    """
    def __init__(self, problem):
        super().__init__(problem)

    def solve(self, monitors=None, abort_test=None, mapper=None, initial_population=None, **options):
        """Same as DreamFit.solve with one additional parameter:

        Additional arg:
            initial_population (Union[np.ndarray, None], optional): initial population to use. Same size
                as results of bumps.fitters.initpop.generate Defaults to None.
        """
        from bumps.dream import Dream
        from bumps.fitters import MonitorRunner, initpop
        if abort_test is None:
            abort_test = lambda: False
        options = _fill_defaults(options, self.settings)
        #print(options, flush=True)

        if mapper:
            self.dream_model.mapper = mapper
        self._update = MonitorRunner(problem=self.dream_model.problem,
                                     monitors=monitors)

        population = initpop.generate(self.dream_model.problem, **options) if initial_population is None else initial_population
        pop_size = population.shape[0]
        draws, steps = int(options['samples']), options['steps']
        if steps == 0:
            steps = (draws + pop_size-1) // pop_size
        # TODO: need a better way to announce number of steps
        # maybe somehow print iteration # of # iters in the monitor?
        print("# steps: %d, # draws: %d"%(steps, pop_size*steps))
        population = population[None, :, :]
        sampler = Dream(model=self.dream_model, population=population,
                        draws=pop_size * steps,
                        burn=pop_size * options['burn'],
                        thinning=options['thin'],
                        monitor=self._monitor, alpha=options['alpha'],
                        outlier_test=options['outliers'],
                        DE_noise=1e-6)

        self.state = sampler.sample(state=self.state, abort_test=abort_test)

        self._trimmed = self.state.trim_portion() if options['trim'] else 1.0
        #print("trimming", options['trim'], self._trimmed)
        self.state.mark_outliers(portion=self._trimmed)
        self.state.keep_best()
        self.state.title = self.dream_model.problem.name

        # TODO: Temporary hack to apply a post-mcmc action to the state vector
        # The problem is that if we manipulate the state vector before saving
        # it then we will not be able to use the --resume feature.  We can
        # get around this by just not writing state for the derived variables,
        # at which point we can remove this notice.
        # TODO: Add derived/visible variable support to other optimizers
        fn, labels = getattr(self.problem, 'derive_vars', (None, None))
        if fn is not None:
            self.state.derive_vars(fn, labels=labels)
        visible_vars = getattr(self.problem, 'visible_vars', None)
        if visible_vars is not None:
            self.state.set_visible_vars(visible_vars)
        integer_vars = getattr(self.problem, 'integer_vars', None)
        if integer_vars is not None:
            self.state.set_integer_vars(integer_vars)

        x, fx = self.state.best()

        # Check that the last point is the best point
        #points, logp = self.state.sample()
        #assert logp[-1] == fx
        #print(points[-1], x)
        #assert all(points[-1, i] == xi for i, xi in enumerate(x))
        return x, -fx


