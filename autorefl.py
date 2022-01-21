import datetime
import numpy as np
import copy
from bumps.cli import load_model, load_best
from bumps.fitters import DreamFit, ConsoleMonitor, _fill_defaults, StepMonitor
from bumps.initpop import generate
from bumps.dream.state import load_state
from bumps.mapper import nice
from refl1d.names import FitProblem, Experiment
from refl1d.resolution import TL2Q, dTdL2dQ
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from bumps.mapper import MPMapper, can_pickle, SerialMapper
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson

d_intens = np.loadtxt('calibration/magik_intensity_hw106.refl')

spec = np.loadtxt('calibration/magik_specular_hw106.dat', usecols=[31, 26, 5], skiprows=9, unpack=False)

qs, s1s, ares = np.delete(spec, 35, 0).T

qs, s1s, ares = spec.T
wv = 5.0
dwv = 0.01648374 * wv
p_intens = np.polyfit(d_intens[:,0], d_intens[:,1], 3, w=1/d_intens[:,2])
pres = np.polyfit(qs, ares, 1)
ps1 = np.polyfit(qs, s1s, 1)

def q2a(q, L):
    return np.degrees(np.arcsin(np.array(q)*L/(4*np.pi)))

def a2q(T, L):
    return 4*np.pi/L * np.sin(np.radians(T))


def sim_data(R, incident_neutrons, addnoise=True, background=0):
    bR = np.ones_like(R)*background*incident_neutrons
    dbR = np.sqrt(bR)
    R = (R+background)*incident_neutrons
    dR = np.sqrt(R)
    if addnoise:
        R += np.random.randn(len(R))*dR
        R[R<0] = 1.0
        #dR = np.sqrt(R)

    return (R-bR)/incident_neutrons, np.sqrt(dR**2 + dbR**2)/incident_neutrons

def sim_data_N(R, incident_neutrons, addnoise=True, resid_bkg=0, meas_bkg=0):
    R = np.array(R, ndmin=1)
    _bR = np.ones_like(R)*(meas_bkg - resid_bkg)*incident_neutrons
    _R = (R + meas_bkg)*incident_neutrons
    N = poisson.rvs(_R, size=_R.shape)
    bN = poisson.rvs(_bR, size=_bR.shape)

    return N, bN, incident_neutrons

def gen_new_variables(newQ):
    Q = np.array(newQ, ndmin=1)
    T = q2a(Q, wv)
    dT = np.polyval(pres, Q)
    L = wv
    dL = dwv
    
    return T, dT, L, dL

def calc_expected_R(fitness, T, dT, L, dL, oversampling=None):
    # currently requires sorted values (by Q) because it returns sorted values.
    # this will need to be modified for CANDOR.
    fitness.probe._set_TLR(T, dT, L, dL, R=None, dR=None, dQ=None)
    if oversampling is not None:
        fitness.probe.oversample(oversampling)
    fitness.update()
    return fitness.reflectivity()[1]

def append_data(newQ, Rth, meas_time, bkgd, T, dT, L, dL, R, dR):
    news1 = np.polyval(ps1, newQ)
    incident_neutrons = np.polyval(p_intens, news1) * meas_time
    newR, newdR = sim_data(Rth, incident_neutrons, background=bkgd)
    T = np.append(T, q2a(newQ, wv))
    dT = np.append(dT, np.polyval(pres, newQ))
    L = np.append(L, np.ones_like(newQ)*wv)
    dL = np.append(dL, np.ones_like(newQ)*dwv)
    R = np.append(R, newR)
    dR = np.append(dR, newdR)
    
    return T, dT, L, dL, R, dR

def append_data_N(newQ, Rth, meas_time, bkgd, T=[], dT=[], L=[], dL=[], N=[], Nbkg=[], Ninc=[]):
    news1 = np.polyval(ps1, newQ)
    incident_neutrons = np.polyval(p_intens, news1) * meas_time
    newN, newNbkg, newNinc = sim_data_N(Rth, incident_neutrons.astype(int), meas_bkg=bkgd)
    T = np.append(T, q2a(newQ, wv))
    dT = np.append(dT, np.polyval(pres, newQ))
    L = np.append(L, np.ones_like(newQ)*wv)
    dL = np.append(dL, np.ones_like(newQ)*dwv)
    N = np.append(N, newN)
    Nbkg = np.append(Nbkg, newNbkg)
    Ninc = np.append(Ninc, newNinc)
    
    return T, dT, L, dL, N, Nbkg, Ninc

def create_init_data_N(newQs, qprofs, dRoR):
    init_data = list()
    for newQ, qprof in zip(newQs, qprofs):
        newR, newdR = np.mean(qprof, axis=0), dRoR * np.std(qprof, axis=0)
        targetN = (newR / newdR) ** 2
        target_incident_neutrons = targetN / newR
        N, Nbkg, Ninc = sim_data_N(newR, target_incident_neutrons.astype(int), meas_bkg=0)
        #print(newR, target_incident_neutrons, N, Nbkg, Ninc)
        T = q2a(newQ, wv)
        dT = np.polyval(pres, newQ)
        L = np.ones_like(newQ)*wv
        dL = np.ones_like(newQ)*dwv
    
        init_data.append((T, dT, L, dL, N, Nbkg, Ninc))

    return init_data

def create_init_data(newQ, qprof, dRoR):
    newR, newdR = np.mean(qprof, axis=0), dRoR * np.std(qprof, axis=0)
    T = q2a(newQ, wv)
    dT = np.polyval(pres, newQ)
    L = np.ones_like(newQ)*wv
    dL = np.ones_like(newQ)*dwv
    R = newR
    dR = newdR
    
    return T, dT, L, dL, R, dR

def compile_data(Qbasis, T, dT, L, dL, R, dR):
    _Q = TL2Q(T=T, L=L)
        # make sure end bins contain the first and last Q values (always should)
    Qbasis[0] = min(min(Qbasis), min(_Q))
    Qbasis[-1] = max(max(Qbasis), max(_Q))
    _R, _bins = np.histogram(_Q, Qbasis, weights=R/dR**2)
    nz = _R.nonzero()
    _normR = np.histogram(_Q, Qbasis, weights=1./dR**2)[0][nz]
    _R = _R[nz]/_normR
    _T = np.histogram(_Q, Qbasis, weights=T/dR**2)[0][nz]/_normR
    _dR = np.sqrt(1./_normR)
    _L = np.ones_like(_T) * wv
    _dL = np.ones_like(_T) * dwv
    _Q = TL2Q(_T, _L)
    _dT = np.polyval(pres, _Q)
    _dQ = dTdL2dQ(_T, _dT, _L, _dL)

    return _T, _dT, _L, _dL, _R, _dR, _Q, _dQ

def compile_data_N(Qbasis, T, dT, L, dL, Ntot, Nbkg, Ninc):
    _Qbasis = copy.copy(Qbasis)
    _Q = TL2Q(T=T, L=L)
    #print('compile_data_N: ', len(_Q), _Q, _Qbasis)
    if len(_Q):
    # make sure end bins contain the first and last Q values (always should)
        _Qbasis[0] = min(min(_Qbasis), min(_Q))
        _Qbasis[-1] = max(max(_Qbasis), max(_Q))
        _N, _bins = np.histogram(_Q, _Qbasis, weights=Ntot)
        _Nbkg = np.histogram(_Q, _Qbasis, weights=Nbkg)[0]
        _norm = np.histogram(_Q, _Qbasis, weights=Ninc)[0]
        nz = _norm.nonzero()
        _R = (_N[nz]-_Nbkg[nz])/_norm[nz]
        _Nmin = np.max(np.vstack(((_N + _Nbkg), np.ones_like(_N))), axis=0)
        _dR = np.sqrt(_Nmin)[nz] / _norm[nz]
        #print(_Q.shape, _dR.shape)
        _normR = np.histogram(_Q, _Qbasis, weights=1./np.array(dT)**2)[0][nz]
        _normRL = np.histogram(_Q, _Qbasis, weights=1./np.array(dL)**2)[0][nz]
        _T = np.histogram(_Q, _Qbasis, weights=np.array(T)/np.array(dT)**2)[0][nz]/_normR
        _L = np.histogram(_Q, _Qbasis, weights=np.array(L)/np.array(dL)**2)[0][nz]/_normRL
        _dL = np.ones_like(_L) * np.mean(dL)
        _Q = TL2Q(_T, _L)
        _dT = np.ones_like(_T) * np.mean(dT)
        _dQ = dTdL2dQ(_T, _dT, _L, _dL)    

        return _T, _dT, _L, _dL, _R, _dR, _Q, _dQ

    else:

        return tuple([np.array([]) for _ in range(8)])

def append_data_overlap(newQ, Rth, meas_time, bkgd, T, dT, L, dL, R, dR, overlap_index=None):
    news1 = np.polyval(ps1, newQ)
    incident_neutrons = np.polyval(p_intens, news1) * meas_time
    newR, newdR = sim_data(Rth, incident_neutrons, background=bkgd)
    if overlap_index is None:
        T = np.append(T, q2a(newQ, wv))
        dT = np.append(dT, np.polyval(pres, newQ))
        L = np.append(L, np.ones_like(newQ)*wv)
        dL = np.append(dL, np.ones_like(newQ)*dwv)
        R = np.append(R, newR)
        dR = np.append(dR, newdR)
    else:
        #print(R[overlap_index], dR[overlap_index], newR, newdR)
        totalR = (R[overlap_index]/dR[overlap_index]**2 + newR/newdR**2) / (1./dR[overlap_index]**2 + 1./newdR**2)
        totaldR = np.sqrt(1./(1./dR[overlap_index]**2 + 1./newdR**2))
        R[overlap_index], dR[overlap_index] = totalR, totaldR
        #print(totalR, totaldR)
    
    return T, dT, L, dL, R, dR

def find_data_overlap(newQ, Qbasis, T, L):
    # checks to see if two data points should be combined. Only works for adding
    # one data point at a time.
    Qs = a2q(T, L)
    #print(Qs[0], Qbasis[0])
    h = np.histogram(np.append(Qs, newQ), bins=Qbasis)[0]
    #print(h, h[h.nonzero()])
    idx = np.where(h[h.nonzero()]>1)[0]
    if len(idx):
        return idx[0]
    else:
        return None

def calc_entropy(pts, select_pars=None):
    # pts is, e.g. MCMCDraw.points
    npar = len(pts[0,:])

    if select_pars is None:
        sel = np.arange(npar)
    else:
        sel = np.array(select_pars)

    pts = pts[:,sel]

    npar = len(sel)

    covX = np.cov(pts.T)
    H = 0.5 * npar * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(covX)[1]
    return H

def calc_init_entropy(problem, select_pars=None):
    #par_scale = np.diff(problem.bounds(), axis=0)
    return calc_entropy(generate(problem, init='random', pop=9, use_point=False), select_pars=select_pars)

def calc_qprofiles(problem, drawpoints, Qth, oversampling=None):
    # given a problem and a sample draw and a Q-vector, calculate the profiles associated with each sample
    calcproblem = copy.deepcopy(problem)
    mlist = [calcproblem] if hasattr(calcproblem, 'fitness') else list(calcproblem.models)
    newvars = [gen_new_variables(Qs) for Qs in Qth]
    qprof = list()
    Qbkg = list()
    for m, newvar in zip(mlist, newvars):
        qprof_item = list()
        Qbkg_item = list()
        for p in drawpoints:
            calcproblem.setp(p)
            calcproblem.chisq_str()
            Rth = calc_expected_R(m.fitness, *newvar, oversampling=oversampling)
            qprof_item.append(Rth)
            Qbkg_item.append(m.fitness.probe.background.value)
        qprof.append(np.array(qprof_item))
        Qbkg.append(np.array(Qbkg_item))

    return qprof, Qbkg

def plot_qprofiles(Qth, qprofs, logps, data=None, ax=None, exclude_from=0, power=4):
    #Qs, Rs, dRs = problem.fitness.probe.Q[exclude_from:], problem.fitness.probe.R[exclude_from:], problem.fitness.probe.dR[exclude_from:]
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    else:
        fig = ax.figure.canvas

    if data is not None:
        _, _, _, _, Rs, dRs, Qs, _ = compile_data_N(Qth, *data)
        print('plot_qprofiles: ', len(Qs), Qs)
        if len(Qs) > 0:
            ax.errorbar(Qs[exclude_from:], (Rs*Qs**power)[exclude_from:], (dRs*Qs**power)[exclude_from:], fmt='o', color='k', markersize=10, alpha=0.7, capsize=8, linewidth=3, zorder=100)

    cmin, cmax = np.median(logps) + 2 * np.std(logps) * np.array([-1,1])
    colornorm = colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = cm.ScalarMappable(norm=colornorm, cmap=cm.jet)

    for qp, logp in zip(qprofs, logps):
        ax.plot(Qth, qp*Qth**power, '-', alpha=0.3, color=cmap.to_rgba(logp))

    ax.set_yscale('log')
    if 0:
        ax2 = ax.inset_axes([1.1, 0, 0.08, 1], transform=ax.transAxes)
        plt.colorbar(cmap, ax=ax, cax=ax2)
        ax2.tick_params(axis='y', right=True, left=True, labelright=False, labelleft=True)
        ax3 = ax2.inset_axes([1,0,2,1], transform=ax2.transAxes)
        h = np.histogram(logps, bins=int(round(len(logps)/10)), range=[cmin, cmax])
        ax3.plot(h[0], 0.5 * (h[1][1:] + h[1][:-1]), linewidth=3)
        ax3.fill_betweenx(0.5 * (h[1][1:] + h[1][:-1]), h[0], alpha=0.4)
        xlims = ax3.get_xlim()
        ax3.set_xlim([0, max(xlims)])
        ylims = ax2.get_ylim()
        ax3.set_ylim(ylims)
        ax3.tick_params(axis='y', labelleft=False)
        ax3.set_ylabel('log likelihood')
        ax3.set_xlabel('N')
        ax.set_xlabel(r'$Q_z$ (' + u'\u212b' + r'$^{-1}$)')
        ax.set_ylabel(r'$R \times Q_z^%i$ (' % power + u'\u212b' + r'$^{-4}$)')
    else:
        ax2 = None
        ax3 = None

    return fig, (ax, ax2, ax3)


def calc_foms(problem, drawpoints, Qth, qprofs, qbkgs, eta=0.5, select_pars=None, axs=None):
    
    models = [problem] if hasattr(problem, 'fitness') else list(problem.models)
    nmodels = len(models)

    # define parameter numbers to select
    if select_pars is None:
        sel = np.arange(len(drawpoints[0,:]))
    else:
        sel = np.array(select_pars)

    if axs is None:
        axs = [None] * nmodels
    else:
        assert len(axs) == len(nmodels), 'Axis list must have same length as number of models'

    pts = drawpoints[:,sel]

    # calculate incident intensity everywhere
    news1 = np.polyval(ps1, Qth)
    incident_neutrons = np.polyval(p_intens, news1)    # incident rate per second

    # alternative approach
    models = [problem] if hasattr(problem, 'fitness') else list(problem.models)
    foms = list()
    meas_times = list()
    for m, qprof, qbkg, ax in zip(models, qprofs, qbkgs, axs):
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
        reg.fit(drawpoints/np.std(drawpoints, axis=0), qprof/np.std(qprof, axis=0))
        reg_marg = LinearRegression(fit_intercept=True)
        reg_marg.fit(pts[:,:]/np.std(pts[:,:], axis=0), qprof/np.std(qprof, axis=0))
        J = reg.coef_.T
        J_marg = reg_marg.coef_.T
        covX = np.cov((drawpoints/np.std(drawpoints, axis=0)).T)
        covX_marg = np.cov((pts/np.std(pts, axis=0)).T)
        df2s = np.zeros_like(Qth)
        df2s_marg = np.zeros_like(Qth)    
        for j in range(len(Qth)):
            Jj = J[:,j][:,None]
            df2s[j] = np.squeeze(Jj.T @ covX @ Jj)
            #print('all',np.linalg.det(Jj.T @ Jj))

            Jj = J_marg[:,j][:,None]
            df2s_marg[j] = np.squeeze(Jj.T @ covX_marg @ Jj)
            #print(np.linalg.det(Jj @ Jj.T))

    #    fom2 = 0.5 * np.sum(np.abs(reg_marg.coef_)**2*totalrate[:, None]/varX, axis=1)

        fom = df2s_marg/df2s*totalrate
#        plt.plot(fom)
        foms.append(fom)

        meas_time = (1-eta) / (eta**2 * refl_rate * (minstd/np.mean(qprof, axis=0))**2)
        meas_times.append(meas_time)

        if ax is not None:
            ax.semilogy(Qth, fom)

    return foms, meas_times

def select_new_points(Qth, foms, meas_times, npoints=1):

    nmodels = len(foms)

    maxQs = []
    maxidxs = []
    maxfoms = []

    for fom in foms:
        # find maximum positions
        # a. calculate whether gradient is > 0
        dfom = np.sign(np.diff(np.append(np.insert(fom, 0, 0),0))) < 0
        # b. find zero crossings
        xings = np.diff(dfom.astype(float))
        maxidx = np.where(xings>0)[0]
        maxfoms.append(fom[maxidx])
        maxQs.append(Qth[maxidx])
        maxidxs.append(maxidx)

    #print(maxidxs, maxfoms, maxQs)

    maxidxs_m = [[fom, m, idx] for m, (idxs, mfoms) in enumerate(zip(maxidxs, maxfoms)) for idx, fom in zip(idxs, mfoms)]
    #print(maxidxs_m)
    # select top npoints (if there are enough of them)
    top_n = sorted(maxidxs_m, reverse=True)[:min(npoints, len(maxidxs_m))]
    #print(top_n)

    # generate lists of new Q values, new measurement times, and the figure of merit at each point    
    # must be sorted in ascending Q order for reflectometry calculations
    newQs = []
    new_meastimes = []
    new_foms = []
    for i in range(nmodels):
        # extract values for each model
        newQ = [Qth[idx] for _, j, idx in top_n if i==j]
        new_meastime = [meas_times[i][idx] for _, j, idx in top_n if i==j]
        new_fom = [fom for fom, j, _ in top_n if i==j]

        # sort in ascending Q order
        new_meastimes.append([x for y, x in sorted(zip(newQ, new_meastime))])
        new_foms.append([x for y, x in sorted(zip(newQ, new_fom))])
        newQs.append(sorted(newQ))
    

    return newQs, new_meastimes, new_foms

class DreamFitPlus(DreamFit):
    def __init__(self, problem):
        super().__init__(problem)

    def solve(self, monitors=None, abort_test=None, mapper=None, initial_population=None, **options):
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


