import numpy as np
from bumps.initpop import generate

default_entropy_options = {'method': 'mvn_fast'}

def gmm_entropy(points, n_est=None, n_components=None, covariance_type='full', predictor=None):
    r"""
    Use sklearn.mixture.BayesianGaussianMixture to estimate entropy.

    *points* are the data points in the sample.

    *n_est* are the number of points to use in the estimation; default is
    10,000 points, or 0 for all the points.

    *n_components* are the number of Gaussians in the mixture. Default is
    $5 \sqrt{d}$ where $d$ is the number of dimensions.
    
    *covariance_type* is the GMM parameter of the same name

    *predictor* is the GMM predictor; if None, a new GMM predictor is created;
    using a previously created GMM predictor allows for more efficient calculations

    Returns estimated entropy and uncertainty in the estimate.

    This method uses BayesianGaussianMixture from scikit-learn to build a
    model of the point distribution, then uses Monte Carlo sampling to
    determine the entropy of that distribution. The entropy uncertainty is
    computed from the variance in the MC sample scaled by the number of
    samples. This does not incorporate any uncertainty in the sampling that
    generated the point distribution or the uncertainty in the GMM used to
    model that distribution.
    """
    #from sklearn.mixture import GaussianMixture as GMM
    from sklearn.mixture import BayesianGaussianMixture as GMM
    n, d = points.shape

    # Default to the full set
    if n_est is None:
        n_est = 10000
    elif n_est == 0:
        n_est = n

    # reduce size of draw to n_est
    if n_est >= n:
        x = points
        n_est = n
    else:
        x = points[np.permutation(n)[:n_est]]
        n = n_est

    if n_components is None:
        n_components = int(5*np.sqrt(d))

    ## Standardization doesn't seem to help
    ## Note: sigma may be zero
    #x, mu, sigma = standardize(x)   # if standardized
    predictor = GMM(n_components=n_components, covariance_type=covariance_type,
                    #verbose=True,
                    max_iter=1000) if predictor is None else predictor
    predictor.fit(x)
    #eval_x, _ = predictor.sample(n_est)
    #eval_x, _ = predictor.sample(max(n_est, 10000))
    weight_x = predictor.score_samples(x)
    H = -np.mean(weight_x)
    #with np.errstate(divide='ignore'): H = H + np.sum(np.log(sigma))   # if standardized
    dH = np.std(weight_x, ddof=1) / np.sqrt(n)
    ## cross-check against own calcs
    #alt = GaussianMixture(predictor.weights_, mu=predictor.means_, sigma=predictor.covariances_)
    #print("alt", H, alt.entropy())
    #print(np.vstack((weight_x[:10], alt.logpdf(eval_x[:10]))).T)
    return H / np.log(2), dH / np.log(2), predictor

def calc_entropy(pts, select_pars=None, options=default_entropy_options, predictor=None):
    """Function for calculating entropy

        Inputs:
        pts -- N (number of samples) x P (number of parameters) array, e.g. from MCMCDraw.points
            OR N x P x D, where D is a number of entropy values to be calculated in parallel
        select_pars -- if None, all parameters are used; otherwise a list or array of parameter indices to use
                        for marginalization
        method -- 'mvn' (multivariate normal) or 'gmm' (gaussian mixture model)
        predictor -- predictor (used only for gmm)
        warm_start -- used only for gmm

        Returns:
        H -- MVN entropy (marginalized if select_pars is not None) of pts

    """

    # define number of parameters (always the size of the second dimension)
    npar = int(pts.shape[1])

    # select parameters of interest (all if select_pars is None)
    if select_pars is None:
        sel = np.arange(npar)
    else:
        sel = np.array(select_pars)

    # reset number of parameters of interest
    npar = len(sel)

    pts = pts[:,sel]

    # ensure 3 dimensions
    if pts.ndim < 3:
        pts = np.expand_dims(pts, -1)

    # now has shape D x N x P
    A = np.moveaxis(pts, -1, 0)

    # calculate entropy with selected method
    if options['method'] == 'mvn_fast':

        # remove averages over all samples
        A = A - np.mean(A, axis=-2, keepdims=True)

        # calculate transpose (shape D x P x N)
        A_T = np.swapaxes(A, -1, -2)

        # Calculate covariance matrix (shape D X P X P)
        covs = np.einsum('ikl,ilm->ikm', A_T, A, optimize='greedy') / (A_T.shape[-1] - 1)

        # Alternate approach (slower for small arrays, faster for very large arrays)
        #covs = list()
        #for a in A_T:
        #    covs.append(np.cov(a))

        # Calculate determinant (shape XD)
        _, dets = np.linalg.slogdet(covs)
        Hs = 0.5 * npar * (np.log(2 * np.pi) + 1) + dets
        Hs /= np.log(2)
        dHs, predictor = None, None

    elif options['method'] == 'mvn':
        Hs = list()
        for a in A:
            covX = np.cov(a.T)

            # protects against single selected parameters that give zero-dimension covX
            covX = np.array(covX, ndmin=2)

            H = 0.5 * npar * (np.log(2 * np.pi) + 1) + np.linalg.slogdet(covX)[1]
            H /= np.log(2)
            Hs.append(H)

        Hs = np.array(Hs)
        dHs, predictor = None, None

    elif options['method'] == 'gmm':
        Hs = list()
        dHs = list()
        n_components = None if 'n_components' not in options.keys() else options['n_components']
        for a in A:
            H, dH, predictor = gmm_entropy(a, n_components=n_components, covariance_type='full', predictor=predictor)
            Hs.append(H)
            dHs.append(dH)
            
        Hs = np.array(Hs)
        dHs = np.array(dHs)

    else:
        Hs, dHs, predictor = None, None, None
    
    return Hs, dHs, predictor

def calc_init_entropy(problem, pop, select_pars=None, options=default_entropy_options):
    #pop = fit_params['pop'] * fit_params['steps'] / thinning
    return calc_entropy(generate(problem, init='random', pop=pop, use_point=False), select_pars=select_pars, options=options)
