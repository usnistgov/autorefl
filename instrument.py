import numpy as np
import json
import warnings
from autorefl import q2a, a2q

class ReflectometerBase(object):
    def __init__(self) -> None:
        self._L = None
        self._dL = None
        self.xlabel = ''
        self.name = None

    def x2q(self, x):
        pass

    def x2a(self, x):
        pass

    def qrange2xrange(self, qmin, qmax):
        pass

    def intensity(self, x):
        pass

    def T(self, x):
        
        return q2a(x, self.L(x))

    def dT(self, x):
        pass

    def L(self, x):
        
        return np.array(np.ones_like(x) * self._L, ndmin=1)

    def dT(self, x):
        
        return np.array(np.ones_like(x) * self._dL, ndmin=1)


class MAGIK(ReflectometerBase):
    """ MAGIK Reflectometer
    x = Q """
    def __init__(self) -> None:
        super().__init__()
        self._L = np.array([5.0])
        self._dL = 0.01648374 * self._L
        self.xlabel = r'$Q_z$ (' + u'\u212b' + r'$^{-1}$)'
        self.name = 'MAGIK'

        # load calibration files
        # TODO: tie resolution function to instrument geometry or use Reductus data
        try:
            d_intens = np.loadtxt('calibration/magik_intensity_hw106.refl')

            spec = np.loadtxt('calibration/magik_specular_hw106.dat', usecols=[31, 26, 5], skiprows=9, unpack=False)
            #qs, s1s, ares = np.delete(spec, 35, 0).T
            qs, s1s, ares = spec.T
            self.p_intens = np.polyfit(d_intens[:,0], d_intens[:,1], 3, w=1/d_intens[:,2])
            self.pres = np.polyfit(qs, ares, 1)
            self.ps1 = np.polyfit(qs, s1s, 1)
        except OSError:
            warnings.warn('MAGIK calibration files not found, using defaults')
            self.p_intens = np.array([ 5.56637543e+02,  7.27944632e+04,  2.13479802e+02, -4.37052050e+01])
            self.ps1 = np.array([ 1.35295366e+01, -9.99016840e-04])
            self.pres = np.array([ 2.30358547e-01, -1.18046955e-05])

    def x2q(self, x):
        return x

    def x2a(self, x):
        return q2a(x, self._L)

    def qrange2xrange(self, bounds):
        return min(bounds), max(bounds)

    def intensity(self, x):
        news1 = np.polyval(self.ps1, x)
        incident_neutrons = np.polyval(self.p_intens, news1)
    
        return np.array(incident_neutrons, ndmin=2).T

    def T(self, x):
        # TODO: use instrument geometry parameters to calculate this
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self.x2a(x), (len(self._L), len(x))).T

    def dT(self, x):
        # TODO: use instrument geometry parameters to calculate this
        x = np.array(x, ndmin=1)
        dTs = np.polyval(self.pres, x)
        return np.broadcast_to(dTs, (len(self._L), len(x))).T

    def L(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._L, (len(x), len(self._L)))

    def dL(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._dL, (len(x), len(self._L)))

class CANDOR(ReflectometerBase):
    """ CANDOR Reflectometer with a single bank
    x = T """
    def __init__(self, bank=0) -> None:
        super().__init__()
        
        self.name = 'CANDOR'
        self.xlabel = r'$\Theta$ $(\degree)$'

        L12 = 4000.
        L2S = 356.
        LS3 = 356.
        L34 = 3000.

        # load wavelength calibration
        wvcal = np.flipud(np.loadtxt(f'calibration/DetectorWavelengths_PG_integrate_sumeff_bank{bank}.csv', delimiter=',', usecols=[1, 2]))
        self._L = wvcal[:,0]
        self._dL = wvcal[:,1]

        # load intensity calibration
        with open('calibration/flowcell_d2o_r12_2_5_maxbeam_60_qoverlap0_751388_unpolarized_intensity.json', 'r') as f:
            d = json.load(f)
        
        self.intens_calib = np.squeeze(np.array(d['outputs'][0]['v']))
        self.s1_intens_calib = np.squeeze(d['outputs'][0]['x'])
        #ps1 = np.polynomial.polynomial.polyfit(s1, intens, 1)

        # load resolution calibration
        with open('calibration/flowcell_d2o_r12_2_5_maxbeam_60_qoverlap0_751394_unpolarized_specular.json', 'r') as f:
            spec = json.load(f)
        
        #s1idx = spec['outputs'][0]['scan_label'].index('slitAperture1.softPosition')
        #s1spec = spec['outputs'][0]['scan_value'][s1idx]
        #Tspec = spec['outputs'][0]['scan_value'][spec['outputs'][0]['scan_label'].index('sampleAngleMotor.softPosition')]
        self.s1_spec_calib = np.squeeze(spec['outputs'][0]['scan_value'][spec['outputs'][0]['scan_label'].index('slitAperture1.softPosition')])
        self.angular_resolution_calib = np.squeeze(spec['outputs'][0]['angular_resolution'])
        self.T_calib = np.squeeze(spec['outputs'][0]['scan_value'][spec['outputs'][0]['scan_label'].index('sampleAngleMotor.softPosition')])

    def x2q(self, x):
        return a2q(x, self.L(x))

    def x2a(self, x):
        return x

    def qrange2xrange(self, qbounds):
        qbounds = np.array(qbounds)
        minx = q2a(min(qbounds), max(self._L))
        maxx = q2a(max(qbounds), min(self._L))
        return minx, maxx

    def intensity(self, x):

        news1 = np.interp(x, self.T_calib, self.s1_spec_calib, left=np.nan, right=np.nan)
        incident_neutrons = [np.interp(news1, self.s1_intens_calib, intens) for intens in self.intens_calib.T]
    
        return np.array(incident_neutrons, ndmin=2).T

    def T(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(x, (len(self._L), len(x))).T

    def dT(self, x):
        x = np.array(x, ndmin=1).T
        dTs = np.interp(x, self.T_calib, self.angular_resolution_calib, left=np.nan, right=np.nan)
        return np.broadcast_to(dTs, (len(self._L), len(x))).T

    def L(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._L, (len(x), len(self._L)))

    def dL(self, x):
        x = np.array(x, ndmin=1)
        return np.broadcast_to(self._dL, (len(x), len(self._L)))
    

# TODO: Replace these (currently not used) with Reductus functions for angular resolution using instrument geometry
def angular_width(s1, s2, L12):

    return (s1 + s2)/L12

def beamwidthatsample(s1, s2, L12, L2S):

    bws = s2 + angular_width(s1, s2, L12)*(L2S)

    return bws

def beamwidthatdetector(s1, s2, L12, L2S, LS3, L34):

    bw4 = s2 + angular_width(s1, s2, L12)*(L2S + LS3 + L34)

    return bw4