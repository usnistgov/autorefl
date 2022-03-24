import numpy
from scipy.special import erf
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline, interpn
from scipy.spatial.transform import Rotation
from scipy.ndimage.filters import gaussian_filter

from periodictable.fasta import xray_sld
from components import Component

class nSLDObj:

    def __init__(self, name=None):
        self.bWrapping = True
        self.bConvolution = False
        self.bProtonExchange = False
        self.dSigmaConvolution = 1
        self.iNumberOfConvPoints = 7
        self.absorb = 0
        
        self.z = 0
        self.l = 0
        self.vol = 0
        self.nSL = 0
        self.nf = 0
        self.sigma = 0

        if name is not None:
            self.name = name
        
    def fnGetAbsorb(self, z):
        return self.absorb
        
    # returns a n-point gaussian interpolation of the area within 4 sigma
    # all area calculations are routed through this function, whether they use convolution or not
    # convolution works only for objects with fixed nSLD. Broadening an nSLD profile is not as direct as
    # broadening a nSL profile. For reason, however, objects report a nSLD(z) and not a nSL(z)
    # if it becomes necessary to broaden profiles with variable nSLD, structural changes to the code
    # have to be implemented.
    
    def fnGetArea(self, z):
        raise NotImplementedError()

    def fnGetProfiles(self, z):
        raise NotImplementedError()

    def fnGetConvolutedArea(self, dz):
        # TODO: use scipy image filters or numpy convolution to do this.
        if self.bConvolution:
            dnormsum = 0
            dsum = 0
            for i in range(self.iNumberOfConvPoints):
                dd = 8 / float(self.iNumberOfConvPoints*i) - 4
                dgauss = numpy.exp((-0.5)*dd*dd)       # (sigma_convolution)^2/(sigma_convolution)^2 cancels
                dnormsum += dgauss
                dsum += self.fnGetArea(dz + dd * self.dSigmaConvolution) * dgauss
            if dnormsum != 0:
                return dsum/dnormsum
            else:
                return 0
        else:
            return self.fnGetArea(dz)
            
    def fnGetLowerLimit(self):
        raise NotImplementedError()
        
    def fnGetnSLD(self, z):
        raise NotImplementedError()
        
    def fnGetUpperLimit(self):
        raise NotImplementedError()

    def fnGetZ(self):
        return self.z
        
    def fnSetConvolution(self, sigma_convolution, iNumberOfConvPoints):
        self.bConvolution = True
        self.dSigmaConvolution = sigma_convolution
        self.iNumberOfConvPoints = iNumberOfConvPoints
        
    def fnWriteData2File(self, f, cName, z):

        header = "z"+cName+" a"+cName+" nsl"+cName
        area, nsl, _ = self.fnGetProfiles(z)
        A = numpy.vstack((z, area, nsl)).T
        numpy.savetxt(f, A, fmt='%0.6e', delimiter=' ', comments='', header=header)
        f.write('\n')
        # TODO: implement wrapping

    def fnWriteData2Dict(self, rdict, z):
        area, nsl, nsld = self.fnGetProfiles(z)
        rdict['zaxis'] = z
        rdict['area'] = area
        rdict['nsl'] = nsl
        rdict['nsld'] = nsld
        return rdict

    def fnWritePar2File(self, fp, cName, z):
        raise NotImplementedError()

    def fnWritePar2Dict(self, rdict, cName, z):
        # Same as fnWritePar2File, but output is saved in a dictionary
        raise NotImplementedError()

    @staticmethod
    def fnWriteConstant(fp, name, darea, dSLD, z):
        header = "Constant " + name + " area " + str(darea) + " \n" + \
                 "z_" + name + " a_" + name + " nsl_" + name
        area = darea * numpy.ones_like(z)
        nsl = dSLD * darea * numpy.gradient(z)
        A = numpy.vstack((z, area, nsl)).T
        numpy.savetxt(fp, A, fmt='%0.6e', delimiter=' ', comments='', header=header)
        fp.write('\n')

    def fnWriteConstant2Dict(self, rdict, name, darea, dSLD, z):
        rdict[name] = {}
        rdict[name]['header'] = "Constant " + name + " area " + str(darea)
        rdict[name]['area'] = darea
        rdict[name]['sld'] = dSLD
        constarea = numpy.ones_like(z) * darea
        constsld = numpy.ones_like(z) * dSLD
        rdict[name]['zaxis'] = z
        rdict[name]['area'] = constarea
        rdict[name]['nsl'] = constsld * numpy.gradient(z) *constarea
        rdict[name]['nsld'] = constsld
        return rdict

    # Philosophy for this first method: You simply add more and more volume and nSLD to the
    # volume and nSLD array. After all objects have filled up those arrays the maximal area is
    # determined which is the area per molecule and unfilled volume is filled with bulk solvent.
    # Hopefully the fit algorithm finds a physically meaningful solution. There has to be a global
    # hydration paramter for the bilayer.
    # Returns maximum area

    def fnWriteProfile(self, z, aArea=None, anSL=None):
    
        # do we want a 0.5 * stepsize shift? I believe refl1d FunctionalLayer uses 
        # z = numpy.linspace(0, dimension * stepsize, dimension, endpoint=False)
        # z, aArea, anSL must be numpy arrays with the same shape
       
        # TODO implement wrapping
        # TODO implement absorption

        aArea = numpy.zeros_like(z) if aArea is None else aArea
        anSL = numpy.zeros_like(z) if anSL is None else anSL

        assert(aArea.shape == z.shape)
        assert(anSL.shape == z.shape)

        area, nsl, _ = self.fnGetProfiles(z)
        dMaxArea = area.max()
        aArea += area
        anSL += nsl

        return dMaxArea, aArea, anSL

    def fnOverlayProfile(self, z, aArea, anSL, dMaxArea):
        # z, aArea, anSL must be numpy arrays with the same shape
        assert (aArea.shape == z.shape)
        assert (anSL.shape == z.shape)
        
        # TODO implement wrapping
        # TODO implement absorption

        area, nsl, _ = self.fnGetProfiles(z)
        temparea = aArea + area

        # find any area for which the final area will be greater than dMaxArea
        overmax = temparea > dMaxArea
        # note: unphysical overfill will trigger the following assertion error
        # TODO: implement a correction instead of throwing an error
        assert (not numpy.any((temparea - dMaxArea) > aArea))
        anSL[overmax] = anSL[overmax] * (1 - ((temparea[overmax] - dMaxArea)/aArea[overmax])) + nsl[overmax]
        aArea[overmax] = dMaxArea

        # deal with area for which the final area is not greater than dMaxArea
        aArea[~overmax] += area[~overmax]
        anSL[~overmax] += nsl[~overmax]

        return aArea, anSL


class CompositenSLDObj(nSLDObj):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fnFindSubgroups(self):
        # this should be run at the end of init. Could also store keys if memory is an issue
        # and then use self.__dict__[key]
        # Could use a "subgroup adder" function that adds to subgroups
        self.subgroups = [getattr(self, attr) for attr in dir(self) if isinstance(getattr(self, attr), nSLDObj)]

    def fnGetProfiles(self, z):
        area = numpy.zeros_like(z)
        nsl = numpy.zeros_like(z)
        nsld = numpy.zeros_like(z)
        
        # make sure FindSubGroups was called. TODO: speed tests whether this is too expensive to do every time
        if not hasattr(self, 'subgroups'):
            self.fnFindSubgroups()

        for g in self.subgroups:
            newarea, newnsl, _ = g.fnGetProfiles(z)
            area += newarea
            nsl += newnsl
        
        nsld = numpy.zeros_like(z)
        pos = (area > 0)
        nsld[pos] = nsl[pos] / (area[pos] * numpy.gradient(z)[pos])

        return area * self.nf, nsl * self.nf, nsld

    def fnWritePar2File(self, f, cName, z):
        if not hasattr(self, 'subgroups'):
            self.fnFindSubgroups()

        for g in self.subgroups:
            # this allows objects with the same names from multiple bilayers
            g.fnWritePar2File(f, cName+'.'+g.name, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        if not hasattr(self, 'subgroups'):
            self.fnFindSubgroups()

        for g in self.subgroups:
            # this allows objects with the same names from multiple bilayers
            rdict = g.fnWritePar2Dict(rdict, cName+'.'+g.name, z)

        return rdict


class Box2Err(nSLDObj):

    def __init__(self, dz=20, dsigma1=2, dsigma2=2, dlength=10, dvolume=10, dnSL=0, dnumberfraction=1, name=None):
        super().__init__(name=name)
        self.z = dz
        self.sigma1 = dsigma1
        self.sigma2 = dsigma2
        self.l = dlength
        self.init_l = dlength
        self.vol = dvolume
        self.nSL = dnSL
        self.nf = dnumberfraction
        self.nsldbulk_store = 0.
        self.nSL2 = None

    def fnGetProfiles(self, z, bulknsld=0):

        if bulknsld != 0:
            self.nsldbulk_store = bulknsld

        # calculate area
        # Gaussian function definition, integral is volume, return value is area at positions z
        if (self.l != 0) and (self.sigma1 != 0) and (self.sigma2 != 0):
            area = erf((z - self.z + 0.5 * self.l) / (numpy.sqrt(2) * self.sigma1))
            # result = math.erf((dz - self.z + 0.5 * self.l) / math.sqrt(2) / self.sigma1)
            # result -= math.erf((dz - self.z - 0.5 * self.l) / math.sqrt(2) / self.sigma2)
            area -= erf((z - self.z - 0.5 * self.l) / (numpy.sqrt(2) * self.sigma2))            
            area *= (self.vol / self.l) * 0.5
            area *= self.nf
        else:
            area = numpy.zeros_like(z)
        
        # calculate nSLD
        nsld = self.fnGetnSL(bulknsld) / self.vol * numpy.ones_like(z) if self.vol != 0 else numpy.zeros_like(z)

        # calculate nSL.
        nsl = area * nsld * numpy.gradient(z)
        
        return area, nsl, nsld

    def fnGetnSL(self, bulknsld):
        if self.bProtonExchange:
            if self.vol != 0:
                return ((bulknsld + 0.56e-6) * self.nSL2 + (6.36e-6 - bulknsld) * self.nSL) / (6.36e-6 + 0.56e-6)
            else:
                return 0.
        else:
            return self.nSL

    # Gaussians are cut off below and above 3 sigma double
    def fnGetLowerLimit(self):
        return self.z - 0.5 * self.l - 3 * self.sigma1

    def fnGetUpperLimit(self):
        return self.z + 0.5 * self.l + 3 * self.sigma2

    # 7/6/2021 new feature: only use proton exchange if nSL2 is explicitly set!
    def fnSetnSL(self, _nSL, _nSL2=None):
        self.nSL = _nSL
        if _nSL2 is not None:
            self.nSL2 = _nSL2
            self.bProtonExchange = True
        else:
            self.bProtonExchange = False

    def fnSetSigma(self, sigma1, sigma2=None):
        self.sigma1 = sigma1
        self.sigma2 = sigma1 if sigma2 is None else sigma2

    def fnSetZ(self, dz):
        self.z = dz

    def fnWritePar2File(self, fp, cName, z):
        fp.write("Box2Err "+cName+" z "+str(self.z)+" sigma1 "+str(self.sigma1)+" sigma2 "+str(self.sigma2)+" l "
                 + str(self.l)+" vol "+str(self.vol)+" nSL "+str(self.nSL)+" nSL2 "+str(self.nSL2)+" nf "
                 + str(self.nf)+" \n")
        self.fnWriteData2File(fp, cName, z)

    def fnWritePar2Dict(self, rdict, cName, z):

        rdict[cName] = {}
        rdict[cName]['header'] = "Box2Err " + cName + " z " + str(self.z) + " sigma1 " + str(self.sigma1) + \
                                   " sigma2 " + str(self.sigma2) + " l " + str(self.l) + " vol " + str(self.vol) + \
                                   " nSL " + str(self.nSL) + " nSL2 " + str(self.nSL2) + " nf " + str(self.nf)
        rdict[cName]['z'] = self.z
        rdict[cName]['sigma1'] = self.sigma1
        rdict[cName]['sigma2'] = self.sigma2
        rdict[cName]['l'] = self.l
        rdict[cName]['vol'] = self.vol
        rdict[cName]['nSL'] = self.nSL
        rdict[cName]['nSL2'] = self.nSL2
        rdict[cName]['nf'] = self.nf
        rdict[cName] = self.fnWriteData2Dict(rdict[cName], z)

        return rdict

class ComponentBox(Box2Err):
    """ Box2Err from a components.Component object"""
    def __init__(self, molecule=None, xray_wavelength=None, **kwargs):
        assert molecule is not None, 'Molecule must be specified'
        assert isinstance(molecule, Component), 'Molecule must be a components.Component object'
        super().__init__(dvolume=molecule.cell_volume, dlength=molecule.length, **kwargs)
        self.fnSetnSL(*molecule.fnGetnSL(xray_wavelength))

# TODO: A headgroup class must contain an "innerleaflet" flag that determines whether the headgroup
# is in the inner or outer leaflet. The profiles so obtained should be flipped. This should perhaps
# be standardized, (1) by creating a CompositeHeadgroup class that has this flag already, and (2) by
# finding a reasonable way of flipping the profile after calculating it (removes lots of if statements)


class PC(CompositenSLDObj):
    def __init__(self, name='PC', innerleaflet=False, xray_wavelength=None, **kwargs):
        # innerleaflet flag locates it in the inner leaflet and flips the order of cg, phosphate,
        # and choline groups. If False, it's the outer leaflet
        super().__init__(name=name, **kwargs)

        from components import carbonyl_glycerol, phosphate, choline

        self.cg = ComponentBox(name='cg', molecule=carbonyl_glycerol, xray_wavelength=xray_wavelength)
        self.phosphate = ComponentBox(name='phosphate', molecule=phosphate, xray_wavelength=xray_wavelength)
        self.choline = ComponentBox(name='choline', molecule=choline, xray_wavelength=xray_wavelength)
        
        self.innerleaflet = innerleaflet

        self.groups = {"cg": self.cg, "phosphate": self.phosphate, "choline": self.choline}

        if innerleaflet:

            self.cg.sigma2, self.cg.sigma1 = 2.53, 2.29
            self.phosphate.sigma2, self.phosphate.sigma1 = 2.29, 2.02
            self.choline.sigma2, self.choline.sigma1 = 2.02, 2.26

        else:
            self.cg.sigma1, self.cg.sigma2 = 2.53, 2.29
            self.phosphate.sigma1, self.phosphate.sigma2 = 2.29, 2.02
            self.choline.sigma1, self.choline.sigma2 = 2.02, 2.26

        self.cg.nf=1 
        self.phosphate.nf=1 
        self.choline.nf=1

        self.l = 9.575
        self.init_l = self.l
        self.vol = self.cg.vol+self.phosphate.vol+self.choline.vol
        self.nSL = self.cg.nSL+self.phosphate.nSL+self.choline.nSL
        self.ph_relative_pos = .58
        self.nf = 1.
        self.fnFindSubgroups()        
        self.fnAdjustParameters()

    def fnAdjustParameters(self):
        
        # make sure no group is larger than the entire length of the headgroup
        for g in self.subgroups:
            g.l = min(self.init_l, g.l)

        if self.innerleaflet:
            self.cg.z = self.z + 0.5 * self.l - 0.5 * self.cg.l
            self.choline.z = self.z - 0.5 * self.l + 0.5 * self.choline.l
            z0 = self.z - 0.5 * self.l + 0.5 * self.phosphate.l
            z1 = self.z + 0.5 * self.l - 0.5 * self.phosphate.l
            self.phosphate.z = z0 + (z1 - z0) * (1 - self.ph_relative_pos)
        else:
            self.cg.z = self.z - 0.5 * self.l + 0.5*self.cg.l
            self.choline.z = self.z + 0.5 * self.l - 0.5*self.choline.l
            z0 = self.z - 0.5*self.l + 0.5 * self.phosphate.l
            z1 = self.z + 0.5 * self.l - 0.5*self.phosphate.l
            self.phosphate.z = z0 + (z1 - z0) * self.ph_relative_pos
    
    def fnSet(self, l=None, ph_relative_pos=None, cg_nSL=None, ch_nSL=None, ph_nSL=None):
        if cg_nSL is not None:
            self.cg.nSL = cg_nSL
        if ch_nSL is not None:
            self.choline.nSL = ch_nSL
        if ph_nSL is not None:
            self.phosphate.nSL = ph_nSL
        if l is not None:
            self.l = l
        if ph_relative_pos is not None:
            self.ph_relative_pos = ph_relative_pos
        self.fnAdjustParameters()
    
    def fnGetLowerLimit(self):
        return self.choline.fnGetLowerLimit() if self.innerleaflet else self.cg.fnGetLowerLimit()
    
    def fnGetUpperLimit(self):
        return self.cg.fnGetUpperLimit() if self.innerleaflet else self.choline.fnGetUpperLimit()
    
    def fnGetnSL(self, bulknsld=None):
        return self.cg.nSL + self.phosphate.nSL + self.choline.nSL
    
    def fnGetZ(self): 
        return self.z
    
    def fnSetSigma(self, sigma):
        self.cg.sigma1 = sigma
        self.cg.sigma2 = sigma
        self.phosphate.sigma1 = sigma
        self.phosphate.sigma2 = sigma
        self.choline.sigma1 = sigma
        self.choline.sigma2 = sigma
    
    def fnSetZ(self, dz):
        self.z = dz
        self.fnAdjustParameters()
    
    def fnWritePar2File(self, fp, cName, z):
        prefix = "PCm" if self.innerleaflet else "PC"
        fp.write(prefix + " "+cName+" z "+str(self.z)+" l "+str(self.l)+" vol " +
                 str(self.cg.vol + self.phosphate.vol + self.choline.vol)+" nf " + str(self.nf)+" \n")
        self.fnWriteData2File(fp, cName, z)

        super().fnWritePar2File(fp, cName, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        prefix = "PCm" if self.innerleaflet else "PC"
        rdict[cName] = {}
        rdict[cName]['header'] = prefix + " " + cName + " z " + str(self.z) + " l " + str(self.l) + " vol "
        rdict[cName]['header'] += str(self.cg.vol + self.phosphate.vol + self.choline.vol) + " nf " + str(self.nf)
        rdict[cName]['z'] = self.z
        rdict[cName]['l'] = self.l
        rdict[cName]['vol'] = self.cg.vol + self.phosphate.vol + self.choline.vol
        rdict[cName]['nf'] = self.nf
        rdict[cName] = self.fnWriteData2Dict(rdict[cName], z)

        rdict = super().fnWritePar2Dict(rdict, cName, z)

        return rdict


class PCm(PC):
    # deprecated. Use PC(innerleaflet=True)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cg.sigma2 = 2.53
        self.cg.sigma1 = 2.29
        self.phosphate.sigma2 = 2.29
        self.phosphate.sigma1 = 2.02
        self.choline.sigma2 = 2.02
        self.choline.sigma1 = 2.26
        self.fnAdjustParameters()
        
    def fnAdjustParameters(self):
        self.cg.z = self.z + 0.5 * self.l - 0.5 * self.cg.l
        self.choline.z = self.z - 0.5 * self.l + 0.5 * self.choline.l
        z0 = self.z - 0.5 * self.l + 0.5 * self.phosphate.l
        z1 = self.z + 0.5 * self.l - 0.5 * self.phosphate.l
        self.phosphate.z = z0 + (z1 - z0) * (1 - self.ph_relative_pos)

    def fnGetLowerLimit(self):
        return self.choline.fnGetLowerLimit()

    def fnGetUpperLimit(self): 
        return self.cg.fnGetUpperLimit()

    def fnWritePar2File(self, fp, cName, z):
        fp.write("PCm " + cName + " z " + str(self.z) + " l " + str(self.l) + " vol " +
                 str(self.cg.vol + self.phosphate.vol + self.choline.vol) + " nf " + str(self.nf) + " \n")
        self.fnWriteData2File(fp, cName, z)

        super().fnWritePar2File(fp, cName, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        rdict[cName] = {}
        rdict[cName]['header'] = "PCm " + " " + cName + " z " + str(self.z) + " l " + str(self.l) + " vol "
        rdict[cName]['header'] += str(self.cg.vol + self.phosphate.vol + self.choline.vol) + " nf " + str(self.nf)
        rdict[cName]['z'] = self.z
        rdict[cName]['l'] = self.l
        rdict[cName]['vol'] = self.cg.vol + self.phosphate.vol + self.choline.vol
        rdict[cName]['nf'] = self.nf
        rdict[cName] = self.fnWriteData2Dict(rdict[cName], z)

        rdict = super().fnWritePar2Dict(rdict, cName, z)

        return rdict

class BLM(CompositenSLDObj):
    def __init__(self, lipids, lipid_nf, xray_wavelength=None, **kwargs):
        """ Free bilayer object. Requires:
            o lipids: a list of components.Lipid objects
            o lipid_nf: a list of number fractions (not necessarily normalized) of 
                        equal length to 'lipids'
            
            To use an xray probe, set xray_wavelength to the appropriate value in Angstroms."""

        def _unpack_lipids():
            """ Helper function for BLM classes that unpacks a lipid list into headgroup objects
                and lists of acyl chain and methyl volumes and nSLs. Creates the following attributes:
                o headgroups1: a list of inner leaflet headgroup objects
                o headgroups2: a list of outer leaflet headgroup objects
                NB: each object in headgroups1 and headgroups2 is created as a standalone attribute in "self"
                    so that searching for all nSLDObj in self will return each headgroup individually as
                    headgroup1_1, headgroup1_2, ... headgroup1_n for n lipids. Similarly for headgroup2.
                o vol_acyl_lipids: a numpy array of acyl chain volumes
                o vol_methyl_lipids: a numpy array of methyl volumes
                o nsl_acyl_lipids: a numpy array of acyl chain nSL
                o nsl_methyl_lipids: a numpy array of methyl nSL
                """

            # create objects for each lipid headgroup and a composite tail object
            # TODO: create separate lipid objects so methyl sigmas can be different for different species
            for i, lipid in enumerate(lipids):
                ihg_name = 'headgroup1_%i' % (i + 1)
                ohg_name = 'headgroup2_%i' % (i + 1)

                if isinstance(lipid.headgroup, Component):
                    # populates nSL, nSL2, vol, and l
                    ihg_obj = ComponentBox(name=ihg_name, molecule=lipid.headgroup, xray_wavelength=xray_wavelength)
                    ohg_obj = ComponentBox(name=ohg_name, molecule=lipid.headgroup, xray_wavelength=xray_wavelength)

                elif issubclass(lipid.headgroup, CompositenSLDObj):
                    ihg_obj = lipid.headgroup(name=ihg_name, innerleaflet=True, xray_wavelength=xray_wavelength)
                    ohg_obj = lipid.headgroup(name=ohg_name, innerleaflet=False, xray_wavelength=xray_wavelength)

                else:
                    raise TypeError('Lipid.hg must be a Headgroup object or a subclass of CompositenSLDObj')

                # Note that because there's always one lipid, the objects self.headgroups1[0]
                # and self.headgroups2[0] always exist
                self.__setattr__(ihg_name, ihg_obj)
                self.headgroups1.append(self.__getattribute__(ihg_name))
                self.__setattr__(ohg_name, ohg_obj)
                self.headgroups2.append(self.__getattribute__(ohg_name))

                # find null headgroups to exclude from averaging over headgroup properties
                self.null_hg1 = numpy.array([hg.vol <= 0.0 for hg in self.headgroups1], dtype=bool)
                self.null_hg2 = numpy.array([hg.vol <= 0.0 for hg in self.headgroups2], dtype=bool)

                self.vol_acyl_lipids[i] = lipid.tails.cell_volume
                self.vol_methyl_lipids[i] = lipid.methyls.cell_volume
                if xray_wavelength is None:
                    self.nsl_acyl_lipids[i] = lipid.tails.sld * lipid.tails.cell_volume * 1e-6
                    self.nsl_methyl_lipids[i] = lipid.methyls.sld * lipid.methyls.cell_volume * 1e-6
                else:
                    self.nsl_acyl_lipids[i] = xray_sld(lipid.tails.formula, wavelength=xray_wavelength)[
                                                  0] * lipid.tails.cell_volume * 1e-6
                    self.nsl_methyl_lipids[i] = xray_sld(lipid.methyls.formula, wavelength=xray_wavelength)[
                                                    0] * lipid.methyls.cell_volume * 1e-6

            self.initial_hg1_lengths = numpy.array([hg1.l for hg1 in self.headgroups1])

        super().__init__(**kwargs)
        assert len(lipids) == len(lipid_nf), \
            'List of lipids and number fractions must be of equal length, not %i and %i' % (len(lipids), len(lipid_nf))
        assert len(lipids) > 0, 'Must specify at least one lipid'

        # normalize number fractions. This allows ratios of lipids to be given instead of number fractions
        self.lipid_nf = numpy.array(lipid_nf) / numpy.sum(lipid_nf)
        n_lipids = len(lipids)
        self.lipids = lipids  # store this information

        self.headgroups1 = []  # list of inner headgroups
        self.headgroups2 = []  # list of outer headgroups
        self.null_hg1 = numpy.zeros(n_lipids)
        self.null_hg2 = numpy.zeros(n_lipids)
        self.vol_acyl_lipids = numpy.zeros(n_lipids)  # list of acyl chain volumes
        self.nsl_acyl_lipids = numpy.zeros(n_lipids)  # list of acyl chain nsls
        self.vol_methyl_lipids = numpy.zeros(n_lipids)  # list of acyl chain methyl volumes
        self.nsl_methyl_lipids = numpy.zeros(n_lipids)  # list of acyl chain methyl nsls

        _unpack_lipids()

        self.lipid1 = Box2Err(name='lipid1')
        self.methyl1 = Box2Err(name='methyl1')
        self.methyl2 = Box2Err(name='methyl2')
        self.lipid2 = Box2Err(name='lipid2')

        self.defect_hydrocarbon = Box2Err(name='defect_hc')
        self.defect_headgroup = Box2Err(name='defect_hg')
        
        self.nf = 1.

        self.vf_bilayer = 1.0
        self.absorb = 0.
        self.l_lipid1 = 11.
        self.l_lipid2 = 11.
        self.bulknsld = -0.56e-6
        self.normarea = 60.
        self.startz = 50.
        self.sigma = 2.
        self.methyl_sigma = 2.
        self.radius_defect = 100.
        self.hc_substitution_1 = 0
        self.hc_substitution_2 = 0

        self._calc_av_hg()
        self.initial_hg1_l = self.av_hg1_l

        self.fnAdjustParameters()
        self.fnFindSubgroups()

    def _calc_av_hg(self):
        # calculate average headgroup lengths, ignore zero volume (e.g. cholesterol)
        self.av_hg1_l = numpy.sum(numpy.array([hg.l for hg, use in zip(self.headgroups1, ~self.null_hg1) if use]) *
                                  self.lipid_nf[~self.null_hg1]) / numpy.sum(self.lipid_nf[~self.null_hg1])
        self.av_hg2_l = numpy.sum(numpy.array([hg.l for hg, use in zip(self.headgroups2, ~self.null_hg2) if use]) *
                                  self.lipid_nf[~self.null_hg2]) / numpy.sum(self.lipid_nf[~self.null_hg2])

    def fnAdjustParameters(self):
        self._adjust_lipids()
        self._adjust_z(self.startz + self.av_hg1_l)
        self._adjust_defects()
        self.fnSetSigma(self.sigma)

    def _adjust_lipids(self):
        self.l_lipid1 = max(self.l_lipid1, 0.01)
        self.l_lipid2 = max(self.l_lipid2, 0.01)
        for nf in self.lipid_nf:
            nf = max(nf, 0)

        # this is changed normalization behavior, but it's a bit more intuitive
        self.lipid_nf /= numpy.sum(self.lipid_nf)

        self.vf_bilayer = max(self.vf_bilayer, 1E-5)
        self.vf_bilayer = min(self.vf_bilayer, 1)

        # calculate average headgroup lengths, ignore zero volume (e.g. cholesterol)
        self.av_hg1_l = numpy.sum(numpy.array([hg.l for hg, use in zip(self.headgroups1, ~self.null_hg1) if use]) *
                                  self.lipid_nf[~self.null_hg1]) / numpy.sum(self.lipid_nf[~self.null_hg1])
        self.av_hg2_l = numpy.sum(numpy.array([hg.l for hg, use in zip(self.headgroups2, ~self.null_hg2) if use]) *
                                  self.lipid_nf[~self.null_hg2]) / numpy.sum(self.lipid_nf[~self.null_hg2])
        
        # outer hydrocarbons
        l_ohc = self.l_lipid2
        nf_ohc_lipid = self.lipid_nf
        V_ohc = numpy.sum(nf_ohc_lipid * (self.vol_acyl_lipids - self.vol_methyl_lipids))
        nSL_ohc = numpy.sum(nf_ohc_lipid * (self.nsl_acyl_lipids - self.nsl_methyl_lipids))

        self.normarea = V_ohc / l_ohc
        c_s_ohc = self.vf_bilayer
        c_A_ohc = 1
        c_V_ohc = 1
        
        self.lipid2.l = l_ohc
        self.lipid2.vol = V_ohc
        self.lipid2.nSL = nSL_ohc
        self.lipid2.nf = c_s_ohc * c_A_ohc * c_V_ohc
        
        # outer methyl
        nf_om_lipid = nf_ohc_lipid
        # cholesterol has a methyl volume of zero and does not contribute
        V_om = numpy.sum(nf_om_lipid * self.vol_methyl_lipids)
        l_om = l_ohc * V_om / V_ohc
        nSL_om = numpy.sum(nf_om_lipid * self.nsl_methyl_lipids)
        
        c_s_om = c_s_ohc
        c_A_om = 1
        c_V_om = 1
        
        self.methyl2.l = l_om
        self.methyl2.vol = V_om
        self.methyl2.nSL = nSL_om
        self.methyl2.nf = c_s_om * c_A_om * c_V_om
        
        # inner hydrocarbons
        l_ihc = self.l_lipid1
        nf_ihc_lipid = nf_ohc_lipid
        V_ihc = numpy.sum(nf_ihc_lipid * (self.vol_acyl_lipids - self.vol_methyl_lipids))
        nSL_ihc = numpy.sum(nf_ihc_lipid * (self.nsl_acyl_lipids - self.nsl_methyl_lipids))
        
        c_s_ihc = self.vf_bilayer
        c_A_ihc = self.normarea * l_ihc / V_ihc
        c_V_ihc = 1
        
        self.lipid1.l = l_ihc
        self.lipid1.vol = V_ihc
        self.lipid1.nSL = nSL_ihc
        self.lipid1.nf = c_s_ihc * c_A_ihc * c_V_ihc
        
        # inner methyl
        nf_im_lipid = nf_ihc_lipid
        # cholesterol has a methyl volume of zero and does not contribute
        V_im = numpy.sum(nf_im_lipid * self.vol_methyl_lipids)
        l_im = l_ihc * V_im / V_ihc
        nSL_im = numpy.sum(nf_im_lipid * self.nsl_methyl_lipids)

        c_s_im = c_s_ihc
        c_A_im = c_A_ihc
        c_V_im = 1
        
        self.methyl1.l = l_im
        self.methyl1.vol = V_im
        self.methyl1.nSL = nSL_im
        self.methyl1.nf = c_s_im * c_A_im * c_V_im

        # headgroup size and position
        for hg1, nf_ihc, hg2, nf_ohc in zip(self.headgroups1, nf_ihc_lipid, self.headgroups2, nf_ohc_lipid):
            hg2.nf = c_s_ohc * c_A_ohc * nf_ohc * (1 - self.hc_substitution_2)
            hg1.nf = c_s_ihc * c_A_ihc * nf_ihc * (1 - self.hc_substitution_1)
            if hasattr(hg1, 'fnAdjustParameters'):
                hg1.fnAdjustParameters()
            if hasattr(hg2, 'fnAdjustParameters'):
                hg2.fnAdjustParameters()

    # defects
    def _adjust_defects(self):
        hclength = self.lipid1.l + self.methyl1.l + self.methyl2.l + self.lipid2.l
        hglength = self.av_hg1_l + self.av_hg2_l

        if self.radius_defect<(0.5*(hclength+hglength)):
            self.radius_defect = 0.5 * (hclength+hglength)
        
        volhalftorus = numpy.pi**2 * (self.radius_defect - (2. * hclength / 3. / numpy.pi)) * hclength * hclength / 4.
        volcylinder = numpy.pi * self.radius_defect * self.radius_defect * hclength
        defectarea = volhalftorus / volcylinder * (1 - self.vf_bilayer) * self.normarea
        
        self.defect_hydrocarbon.vol = defectarea * hclength
        self.defect_hydrocarbon.l = hclength
        self.defect_hydrocarbon.z = self.lipid1.z - 0.5 * self.lipid1.l + 0.5 * hclength
        self.defect_hydrocarbon.nSL = self.lipid2.nSL / self.lipid2.vol * self.defect_hydrocarbon.vol
        self.defect_hydrocarbon.fnSetSigma(self.sigma)
        self.defect_hydrocarbon.nf = 1
        
        defectratio = self.defect_hydrocarbon.vol / self.lipid2.vol
        self.defect_headgroup.vol = defectratio * numpy.sum([hg.nf * hg.vol for hg in self.headgroups2])
        self.defect_headgroup.l = hclength + hglength
        self.defect_headgroup.z = self.lipid1.z - 0.5 * self.lipid1.l - \
                                  0.5 * self.av_hg1_l + 0.5 * (hclength + hglength)
        self.defect_headgroup.nSL = defectratio * numpy.sum([hg.nf * hg.fnGetnSL(self.bulknsld)
                                                             for hg in self.headgroups2])
        self.defect_headgroup.fnSetSigma(self.sigma)
        self.defect_headgroup.nf = 1

    def _adjust_z(self, startz):
        # startz is the position of the hg1/lipid1 interface.
        # change here: use average headgroup length instead of a specific headgroup
        self.lipid1.z= startz + 0.5 * self.lipid1.l
        self.methyl1.z = self.lipid1.z + 0.5 * (self.lipid1.l + self.methyl1.l)
        self.methyl2.z = self.methyl1.z + 0.5 * (self.methyl1.l + self.methyl2.l)
        self.lipid2.z = self.methyl2.z + 0.5 * (self.methyl2.l + self.lipid2.l)

        for hg1, hg2 in zip(self.headgroups1, self.headgroups2):
            hg1.fnSetZ(self.lipid1.z - 0.5 * self.lipid1.l - 0.5 * hg1.l)
            hg2.fnSetZ(self.lipid2.z + 0.5 * self.lipid2.l + 0.5 * hg2.l)

    # return value of center of the membrane
    def fnGetCenter(self):
        return self.methyl1.z + 0.5 * self.methyl1.l

    # Use limits of molecular subgroups
    def fnGetLowerLimit(self):
        return min([hg.fnGetLowerLimit() for hg in self.headgroups1])
        
    def fnGetUpperLimit(self):
        return max([hg.fnGetUpperLimit() for hg in self.headgroups2])

    def fnSet(self, sigma, bulknsld, startz, l_lipid1, l_lipid2, vf_bilayer,
              nf_lipids=None, hc_substitution_1=0.,
              hc_substitution_2=0., radius_defect=100.):
        self.sigma = sigma
        self.bulknsld = bulknsld
        self.startz = startz
        self.l_lipid1 = l_lipid1
        self.l_lipid2 = l_lipid2
        self.vf_bilayer = vf_bilayer
        if nf_lipids is not None:   # pass None to keep lipid_nf unchanged
            assert len(nf_lipids) == len(self.lipids), \
                'nf_lipids must be same length as number of lipids in bilayer, not %i and %i' % (len(nf_lipids),
                                                                                                 len(self.lipids))
            self.lipid_nf = numpy.array(nf_lipids)
        self.hc_substitution_1 = hc_substitution_1
        self.hc_substitution_2 = hc_substitution_2
        self.radius_defect = radius_defect
        
        self.fnAdjustParameters()

    def fnSetSigma(self, sigma):
        for hg1, hg2 in zip(self.headgroups1, self.headgroups2):
            hg1.fnSetSigma(sigma)
            hg2.fnSetSigma(sigma)
        self.lipid1.fnSetSigma(sigma,numpy.sqrt(sigma**2+self.methyl_sigma**2))
        self.methyl1.fnSetSigma(numpy.sqrt(sigma**2+self.methyl_sigma**2), numpy.sqrt(sigma**2+self.methyl_sigma**2))
        self.methyl2.fnSetSigma(numpy.sqrt(sigma**2+self.methyl_sigma**2), numpy.sqrt(sigma**2+self.methyl_sigma**2))
        self.lipid2.fnSetSigma(numpy.sqrt(sigma**2+self.methyl_sigma**2),sigma)
        self.defect_hydrocarbon.fnSetSigma(sigma)
        self.defect_headgroup.fnSetSigma(sigma)

    def fnWritePar2File(self, fp, cName, z):
        super().fnWritePar2File(fp, cName, z)
        self.fnWriteConstant(fp, cName+"_normarea", self.normarea, 0, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        rdict = super().fnWritePar2Dict(rdict, cName, z)
        rdict = self.fnWriteConstant2Dict(rdict, cName + ".normarea", self.normarea, 0, z)
        return rdict


class ssBLM(BLM):
    """
    Solid supported lipid bilayer
    """
    def __init__(self, lipids, lipid_nf, xray_wavelength=None, **kwargs):
        """ Solid supported bilayer object. Requires:
            o lipids: a list of components.Lipid objects
            o lipid_nf: a list of number fractions (not necessarily normalized) of 
                        equal length to 'lipids'
            
            To use an xray probe, set xray_wavelength to the appropriate value in Angstroms.
        """
        # add ssBLM-specific subgroups
        self.substrate = Box2Err(name='substrate')
        self.siox = Box2Err(name='siox')

        self.substrate.l = 40
        self.substrate.z = 0
        self.substrate.nf = 1
        self.rho_substrate = 2.07e-6
        self.l_siox = 1
        self.rho_siox = 3.55e-6

        self.siox.l = 20
        self.siox.z = self.substrate.z + self.substrate.l / 2.0 + self.siox.l / 2.0
        self.siox.nf = 1

        self.l_submembrane = 10.
        self.global_rough = 2.0

        super().__init__(lipids, lipid_nf, xray_wavelength=xray_wavelength, **kwargs)

    def fnAdjustParameters(self):
        
        self._adjust_lipids()

        # substrate
        self.substrate.vol = self.normarea * self.substrate.l
        self.substrate.nSL = self.rho_substrate * self.substrate.vol
        self.siox.l = self.l_siox
        self.siox.vol = self.normarea * self.siox.l
        self.siox.nSL = self.rho_siox * self.siox.vol
        self.siox.z=self.substrate.z + self.substrate.l / 2 + 0.5 * self.siox.l

        self._adjust_z(self.substrate.z + self.substrate.l / 2 + self.siox.l + self.l_submembrane + self.av_hg1_l)
        self._adjust_defects()

        self.fnSetSigma(self.sigma)

    def fnSetSigma(self, sigma):
        super().fnSetSigma(sigma)
        self.substrate.fnSetSigma(self.global_rough)
        self.siox.fnSetSigma(self.global_rough)

    def fnGetLowerLimit(self):
        return self.substrate.fnGetLowerLimit()
        # does this make sense since this goes to negative z and isn't intended to be used?

    def fnSet(self, _sigma, _bulknsld, _global_rough, _rho_substrate, _rho_siox, _l_siox, _l_submembrane, _l_lipid1,
              _l_lipid2, _vf_bilayer, _nf_lipids=None, _hc_substitution_1=0.,
              _hc_substitution_2=0., _radius_defect=100.):
        self.sigma = _sigma
        self.bulknsld = _bulknsld
        self.global_rough = _global_rough
        self.rho_substrate = _rho_substrate
        self.rho_siox = _rho_siox
        self.l_siox = _l_siox 
        self.l_submembrane = _l_submembrane 
        self.l_lipid1 = _l_lipid1
        self.l_lipid2 = _l_lipid2
        self.vf_bilayer = _vf_bilayer
        if _nf_lipids is not None:   # pass None to keep lipid_nf unchanged
            assert len(_nf_lipids)==len(self.lipids), \
                'nf_lipids must be same length as number of lipids in bilayer, not %i and %i' % \
                (len(_nf_lipids), len(self.lipids))
            self.lipid_nf = numpy.array(_nf_lipids)
        self.hc_substitution_1 = _hc_substitution_1
        self.hc_substitution_2 = _hc_substitution_2
        self.radius_defect = _radius_defect

        self.fnAdjustParameters()


# ------------------------------------------------------------------------------------------------------
# Tethered Lipid bilayer
# ------------------------------------------------------------------------------------------------------
class tBLM(BLM):
    """
    Tethered lipid bilayer
    """
    def __init__(self, lipids, lipid_nf, tether, filler, xray_wavelength=None, **kwargs):
        """
        Tethered lipid bilayer. Requires:

        o tether: a components.Tether object        o lipids: a list of components.Lipid objects
        o lipid_nf: a list of number fractions (not necessarily normalized) of 
                    equal length to 'lipids' describing the tether.
        o filler: a components.Component object describing the filler molecule, including its length

        To use an xray probe, set xray_wavelength to the appropriate value in Angstroms.
        """

        # add ssBLM-specific subgroups
        self.substrate = Box2Err(name='substrate')
        self.nf_tether = 0.3
        self.l_tether=8.0
        self.mult_tether = 7.0/3.0
        self.xray_wavelength = xray_wavelength

        # change these to Headgroup2Box
        self.bME = ComponentBox(name='bME', molecule=filler, xray_wavelength=xray_wavelength)
        self.tether_bme = Box2Err(name='tether_bme')
        self.tether_free = Box2Err(name='tether_free')
        self.tether_hg = Box2Err(name='tether_hg')
        self.tether = Component(name='tether', formula=tether.tether.formula,
                                 cell_volume=tether.tether.cell_volume, length=self.l_tether)
        self.tether.nSLs = self.tether.fnGetnSL(xray_wavelength=xray_wavelength)
        self.tetherg = Component(name='tetherg', formula=tether.tetherg.formula,
                                 cell_volume=tether.tetherg.cell_volume, length=10.0)
        self.tetherg.nSLs = self.tetherg.fnGetnSL(xray_wavelength=xray_wavelength)

        self.substrate.l = 40
        self.substrate.z = 0
        self.substrate.nf = 1
        self.rho_substrate = 2.07e-6

        self.global_rough = 2.0

        self.vol_acyl_tether = tether.tails.cell_volume
        self.vol_methyl_tether = tether.methyls.cell_volume
        if xray_wavelength is None:
            self.nSL_acyl_tether = tether.tails.sld * tether.tails.cell_volume * 1e-6
            self.nSL_methyl_tether = tether.methyls.sld * tether.methyls.cell_volume * 1e-6
        else:
            self.nSL_acyl_tether = xray_sld(tether.tails.formula, wavelength=xray_wavelength)[0] * tether.tails.cell_volume * 1e-6
            self.nSL_methyl_tether = xray_sld(tether.methyls.formula, wavelength=xray_wavelength)[0] * tether.methyls.cell_volume * 1e-6

        self.initial_bME_l = self.bME.l

        super().__init__(lipids, lipid_nf, xray_wavelength=xray_wavelength, **kwargs)

    def fnAdjustParameters(self):
        
        self._adjust_lipids()

        # substrate
        self.substrate.vol = self.normarea * self.substrate.l
        self.substrate.nSL = self.rho_substrate * self.substrate.vol

        # set all lengths
        self.bME.z = 0.5 * self.bME.l + self.substrate.z + self.substrate.l * 0.5
        self.tether_bme.z = self.bME.z
        self.tether_free.z = self.tether_bme.z + 0.5 * self.tether_bme.l + 0.5 * self.tether_free.l
        self.tether_hg.z = self.tether_free.z + 0.5 * self.tether_free.l + 0.5 * self.tether_hg.l
        
        self._adjust_z(self.tether_hg.z + 0.5 * self.tether_hg.l)
        
        self._adjust_defects()

        self.fnSetSigma(self.sigma)

    def _adjust_lipids(self):

        # sufficiently different from parent class that code is repeated here with small adjustments
        # TODO: split into inner and outer leaflets so this can be made more efficient
        
        self.l_lipid1 = max(self.l_lipid1, 0.01)
        self.l_lipid2 = max(self.l_lipid2, 0.01)
        for nf in self.lipid_nf:
            nf = max(nf, 0)

        # this is changed normalization behavior, but it's a bit more intuitive
        self.lipid_nf /= numpy.sum(self.lipid_nf)

        self.vf_bilayer = max(self.vf_bilayer, 1E-5)
        self.vf_bilayer = min(self.vf_bilayer, 1)
    
        # outer hydrocarbons
        l_ohc = self.l_lipid2
        nf_ohc_lipid = self.lipid_nf
        V_ohc = numpy.sum(nf_ohc_lipid * (self.vol_acyl_lipids - self.vol_methyl_lipids))
        nSL_ohc = numpy.sum(nf_ohc_lipid * (self.nsl_acyl_lipids - self.nsl_methyl_lipids))

        self.normarea = V_ohc / l_ohc
        c_s_ohc = self.vf_bilayer
        c_A_ohc = 1
        c_V_ohc = 1
        
        self.lipid2.l = l_ohc
        self.lipid2.vol = V_ohc
        self.lipid2.nSL = nSL_ohc
        self.lipid2.nf = c_s_ohc * c_A_ohc * c_V_ohc
        
        # outer methyl
        nf_om_lipid = nf_ohc_lipid
        # cholesterol has a methyl volume of zero and does not contribute
        V_om = numpy.sum(nf_om_lipid * self.vol_methyl_lipids)
        l_om = l_ohc * V_om / V_ohc
        nSL_om = numpy.sum(nf_om_lipid * self.nsl_methyl_lipids)
        
        c_s_om = c_s_ohc
        c_A_om = 1
        c_V_om = 1
        
        self.methyl2.l = l_om
        self.methyl2.vol = V_om
        self.methyl2.nSL = nSL_om
        self.methyl2.nf = c_s_om * c_A_om * c_V_om
        
        # inner hydrocarbons
        l_ihc = self.l_lipid1
        
        nf_ihc_lipid = nf_ohc_lipid * (1 - self.nf_tether)
        nf_ihc_tether = self.nf_tether
        V_ihc = numpy.sum(nf_ihc_lipid * (self.vol_acyl_lipids - self.vol_methyl_lipids)) + nf_ihc_tether * (self.vol_acyl_tether - self.vol_methyl_tether)
        nSL_ihc = numpy.sum(nf_ihc_lipid * (self.nsl_acyl_lipids - self.nsl_methyl_lipids)) + nf_ihc_tether * (self.nSL_acyl_tether - self.nSL_methyl_tether)
        
        c_s_ihc = self.vf_bilayer
        c_A_ihc = self.normarea * l_ihc / V_ihc
        c_V_ihc = 1
        
        self.lipid1.l = l_ihc
        self.lipid1.vol = V_ihc
        self.lipid1.nSL = nSL_ihc
        self.lipid1.nf = c_s_ihc * c_A_ihc * c_V_ihc
        
        # inner methyl
        nf_im_lipid = nf_ihc_lipid
        nf_im_tether = nf_ihc_tether
        V_im = numpy.sum(nf_im_lipid * self.vol_methyl_lipids) + nf_im_tether * self.vol_methyl_tether
        l_im = l_ihc * V_im / V_ihc
        nSL_im = numpy.sum(nf_im_lipid * self.nsl_methyl_lipids) + nf_im_tether * self.nSL_methyl_tether
        
        c_s_im = c_s_ihc
        c_A_im = c_A_ihc
        c_V_im = 1
        
        self.methyl1.l = l_im
        self.methyl1.vol = V_im
        self.methyl1.nSL = nSL_im
        self.methyl1.nf = c_s_im * c_A_im * c_V_im

        # headgroup size and position
        for hg1, nf_ihc, hg2, nf_ohc in zip(self.headgroups1, nf_ihc_lipid, self.headgroups2, nf_ohc_lipid):
            hg2.nf = c_s_ohc * c_A_ohc * nf_ohc * (1 - self.hc_substitution_2)
            hg1.nf = c_s_ihc * c_A_ihc * nf_ihc * (1 - self.hc_substitution_1)
            if hasattr(hg1, 'fnAdjustParameters'):
                hg1.fnAdjustParameters()
            if hasattr(hg2, 'fnAdjustParameters'):
                hg2.fnAdjustParameters()

        self._calc_av_hg()

        # tether glycerol part
        c_s_tg = c_s_ihc
        c_A_tg = c_A_ihc
        c_V_tg = nf_ihc_tether

        # self.tetherg.l = self.tetherg.cell_volume / ((self.vol_acyl_tether -
        # self.vol_methyl_tether) / self.lipid1.l) / 0.9
        tetherg_nf = c_s_tg * c_A_tg * c_V_tg

        # tether EO part
        c_s_EO = c_s_ihc
        c_A_EO = c_A_ihc
        c_V_EO = nf_ihc_tether
        tether_nf = c_s_EO * c_A_EO * c_V_EO

        # Adjust the submembrane space.
        self._adjust_submembrane(tether_nf, tetherg_nf)
        # check to make sure there isn't too much volume in the submembrane space.
        # If there is, increase l_tether to accommodate it
        total_submembrane_V = self.tether.cell_volume * tether_nf + self.tetherg.cell_volume * tetherg_nf + \
                        self.mult_tether * tether_nf * self.bME.vol + numpy.sum([hg.nf * hg.vol for hg in self.headgroups1])
        if total_submembrane_V > self.l_tether * self.normarea * self.vf_bilayer:
            # print('Warning too much volume')
            # print('total_submembrane_V', 'allowed_submembrane_V', 'new l_tether: ', total_submembrane_V,
            # self.normarea * self.l_tether, total_submembrane_V / self.normarea)
            self.l_tether = total_submembrane_V / (self.normarea * self.vf_bilayer)
            # After changing submembrane space size, recalculate everything
            self._adjust_submembrane(tether_nf, tetherg_nf)

    def _adjust_submembrane(self, tether_nf, tetherg_nf):

        """new approach to submembrane space:
        1. divide into three regions, tether_bme, tether_free, and tether_hg
        2. attempt to evenly distribute density of everything in these three regions.
        3. Algorithm:
            Case 1: sparse (l_tether > BME.l + av_hg1_l)
                a. Calculate A_bme
                b. Calculate A_hg
                c. Calculate A_tether_bme, A_tether_free, A_tether_hg such that A_tether_bme + A_bme = A_tether_free = A_tether_hg + A_hg + A_tetherg
            Case 2: compact (l_tether < BME.l + av_hg1_l)
                a. Proportionately compress BME.l and av_hg1_l, so typically 1/3 BME compression, 2/3 hg compression, such that l_tether=BME.l + av_hg1_l
                b. Calculate A_bme
                c. Calculate A_hg
                d. Calculate A_tether_bme, A_tether_free, A_tether_hg such that A_tether_bme + A_bme = A_tether_free = A_tether_hg + A_hg
        4. Also changed: bME now includes some of the tether molecule as necessary. 
        """

        # Reset bME length and headgroup length
        # TODO: find a robust way to keep track of these initial values. If a user changes them between init and this point, the user value will be overwritten
        #       unless they specifically change initial_bME_l as well. This is already done for headgroups (fnSetHeadgroupLength)
        self.bME.l = self.initial_bME_l
        for hg1, initial_length in zip(self.headgroups1, self.initial_hg1_lengths):
            hg1.l = initial_length
        
        self._calc_av_hg()

        # If too much tether is present, adjust the value
        A_bme, min_A_tether_bme = self._adjust_mult_tether(tether_nf, tetherg_nf)

        # Find separation between bme and headgroups
        dsub = self.l_tether - (self.initial_bME_l + self.av_hg1_l)
        #print('dsub', dsub)

        # Total volume of tether molecules, including the glycerol
        V_tether = self.tether.cell_volume * tether_nf + self.tetherg.cell_volume * tetherg_nf

        # Set initial values
        V_tether_free = 0.0
        V_tether_bme = 0.0
        V_tether_hg = 0.0
        
        # Check whether there's enough submembrane space for a "free tether"
        if dsub > 0:

            # Calculate minimum area that has to reside in the headgroup region
            # NOTE: right now this is just the tetherg volume. It should probably be larger (at least 2 EO groups). This
            #       can be adjusted in the Tether molecule so tetherg has more volume.
            min_A_tether_hg = self.tetherg.cell_volume * tetherg_nf / self.av_hg1_l

            # Calculate base area for both headgroups and tether
            A_hg = numpy.sum([hg.nf * hg.vol / hg.l for hg in self.headgroups1]) + min_A_tether_hg
            V_tether_bme = min_A_tether_bme * self.bME.l
            V_tether_hg = min_A_tether_hg * self.av_hg1_l
            V_tether -= V_tether_bme
            V_tether -= V_tether_hg
            # this is now the remaining tether volume after minimum amounts have been apportioned to bme and hg regions

            # Which region gets tether first depends on the total existing area in each region
            if A_hg > A_bme:
                V_tether_free += A_bme * dsub # maximum area before start filling A_bme
                if V_tether_free >= V_tether: # all remaining tether volume goes into free region
                    V_tether_free += V_tether
                    V_tether = 0.0
                else:
                    V_tether -= V_tether_free # fill V_tether_free up to level of A_bme
                    # Fill up bme area with tether until V_tether is exhausted or A_bme reaches A_hg
                    if (A_hg - A_bme) * (self.bME.l + dsub) >= V_tether:    
                        V_tether_bme += V_tether * self.bME.l / (self.bME.l + dsub)
                        V_tether_free += V_tether * dsub / (self.bME.l + dsub)
                        V_tether = 0.0
                    else:
                        V_tether -= (A_hg - A_bme) * (self.bME.l + dsub)
                        V_tether_bme += (A_hg - A_bme) * self.bME.l / (self.bME.l + dsub)
                        V_tether_free += (A_hg - A_bme) * dsub / (self.bME.l + dsub)
            else: # A_bme < A_hg
                V_tether_free += A_hg * dsub # maximum area before start filling A_hg
                if V_tether_free >= V_tether: # all remaining tether volume goes into free region
                    V_tether_free += V_tether
                    V_tether = 0.0
                else:
                    V_tether -= V_tether_free # fill V_tether_free up to level of A_hg
                    if (A_bme - A_hg) * (self.av_hg1_l + dsub) >= V_tether:
                        V_tether_hg += V_tether * self.av_hg1_l / (self.av_hg1_l + dsub)
                        V_tether_free += V_tether * dsub / (self.av_hg1_l + dsub)
                        V_tether = 0.0
                    # Fill up bme area with tether until V_tether is exhausted or A_hg reaches A_bme
                    else:
                        V_tether -= (A_bme - A_hg) * (self.av_hg1_l + dsub)
                        V_tether_hg += (A_bme - A_hg) * self.av_hg1_l
                        V_tether_free += (A_bme - A_hg) * dsub
            # Apportion remaining tether proportionately among the three regions
            # (sometimes V_tether is zero and this does nothing)
            V_tether_bme += V_tether * self.bME.l / (self.bME.l + dsub + self.av_hg1_l)
            V_tether_free += V_tether * dsub / (self.bME.l + dsub + self.av_hg1_l)
            V_tether_hg += V_tether * self.av_hg1_l / (self.bME.l + dsub + self.av_hg1_l)

        else: # dsub <=0. In this case there's no tether_free region.
            # squish headgroups and bME proportionately to their existing size.
            d1s = self.l_tether - (self.initial_bME_l + self.initial_hg1_lengths)
            #print(d1s, dsub * self.initial_bME_l / (self.initial_bME_l + self.initial_hg1_l))
            self.bME.l += dsub * self.initial_bME_l / (self.initial_bME_l + self.initial_hg1_l)
            for hg1, d1, initial_length in zip(self.headgroups1, d1s, self.initial_hg1_lengths):
                d1 = self.l_tether - (self.bME.l + initial_length)
                #print(d1)
                hg1.l = initial_length + min(d1, 0.0)   # only squish headgroups if individual length is too big.

            # Recalculate average headgroup size
            self._calc_av_hg()

            # apportion volume. Logic is the same as previously, just without the tether_free region.
            A_bme, min_A_tether_bme = self._adjust_mult_tether(tether_nf, tetherg_nf) # do it again because bME thickness has changed

            # Calculate base area for both headgroups and tether
            min_A_tether_hg = self.tetherg.cell_volume * tetherg_nf / self.av_hg1_l
            A_hg = numpy.sum([hg.nf * hg.vol / hg.l for hg in self.headgroups1]) + min_A_tether_hg
            V_tether_bme = min_A_tether_bme * self.bME.l
            V_tether_hg = min_A_tether_hg * self.av_hg1_l
            V_tether -= V_tether_bme
            V_tether -= V_tether_hg
            #print('A_bme, A_hg, normarea, V_tether', A_bme, A_hg, self.normarea, V_tether)
            if A_hg > A_bme:
                #print('(A_hg - A_bme) * self.bME.l)', (A_hg - A_bme) * self.bME.l, 'V_tether', V_tether)
                if ((A_hg - A_bme) * self.bME.l) >= V_tether:
                    V_tether_bme += V_tether
                    V_tether = 0.0
                else:
                    V_tether -= (A_hg - A_bme) * self.bME.l
                    V_tether_bme += (A_hg - A_bme) * self.bME.l
            else:
                #print('(A_bme - A_hg) * self.av_hg1_l)', (A_bme - A_hg) * self.av_hg1_l, 'V_tether', V_tether)
                if (A_bme - A_hg) * self.av_hg1_l >= V_tether:
                    V_tether_hg += V_tether
                    V_tether = 0.0
                else:
                    V_tether -= (A_bme - A_hg) * self.av_hg1_l
                    V_tether_hg += (A_bme - A_hg) * self.av_hg1_l
            # Apportion remaining volume
            V_tether_bme += V_tether * self.bME.l / (self.bME.l + self.av_hg1_l)
            V_tether_hg += V_tether * self.av_hg1_l / (self.bME.l + self.av_hg1_l)

            # Shouldn't need this
            V_excess_bme = V_tether_bme + self.mult_tether * tether_nf * self.bME.vol - self.normarea * self.vf_bilayer * self.bME.l
            if V_excess_bme > 0:
                #print('V_excess_bme', V_excess_bme)
                V_tether_hg += V_excess_bme
                V_tether_bme -= V_excess_bme
            V_excess_hg = V_tether_hg + numpy.sum([hg.nf * hg.vol / hg.l for hg in self.headgroups1]) - self.normarea * self.vf_bilayer * self.av_hg1_l
            if V_excess_hg > 0:
                #print('V_excess_hg', V_excess_hg)
                V_tether_hg -= V_excess_hg
                V_tether_bme += V_excess_hg

        # print('A_bme', 'A_tether_bme', 'min_A_tether_bme', '*mult_tether', 'mult_tether', self.mult_tether * tether_nf
        # * self.bME.vol / self.bME.l, V_tether_bme / self.bME.l, min_A_tether_bme, min_A_tether_bme *
        # (1+self.mult_tether), self.mult_tether)
        # Set volumes of tether objects
        self.tether_bme.vol = V_tether_bme
        self.tether_free.vol = V_tether_free
        self.tether_hg.vol = V_tether_hg

        # Set lengths of tether objects
        self.tether_bme.l = self.bME.l
        self.tether_free.l = max(0.0, dsub)
        self.tether_hg.l = self.l_tether - self.tether_bme.l - self.tether_free.l
        
        # Set scattering lengths based on volume-averaged nSLDs of the components.
        self.tether_bme.fnSetnSL(*(self.tether.nSLs / self.tether.cell_volume * self.tether_bme.vol))
        self.tether_free.fnSetnSL(*(self.tether.nSLs / self.tether.cell_volume * self.tether_free.vol))
        frac_tether_g = self.tetherg.cell_volume / V_tether_hg
        self.tether_hg.fnSetnSL(*((self.tetherg.nSLs / self.tetherg.cell_volume * frac_tether_g + self.tetherg.nSLs / self.tetherg.cell_volume * (1 - frac_tether_g)) * self.tether_hg.vol))
        # print('tether_hg sld', (self.tetherg.nSL / self.tetherg.vol * frac_tether_g + self.tether.nSL /
        # self.tether.vol * (1 - frac_tether_g)))

        self.bME.nf = self.mult_tether * tether_nf

    def _adjust_mult_tether(self, tether_nf, tetherg_nf):
        """
        bME + tether in the bME region can't be bigger than normarea. If it is, reduce mult_tether until it isn't.
        """
        min_A_tether_bme = tether_nf * self.bME.vol / self.bME.l # minimum amount of volume in tether region.
        A_bme = self.mult_tether * tether_nf * self.bME.vol / self.bME.l + min_A_tether_bme # area in bme region
        if A_bme > self.normarea * self.vf_bilayer:
            self.mult_tether = max(0, (self.normarea * self.vf_bilayer - min_A_tether_bme) / (tether_nf * self.bME.vol / self.bME.l))
            # print('big A_bme', 'normarea', 'new mult_tether', A_bme, self.normarea, self.mult_tether)
            A_bme = self.mult_tether * tether_nf * self.bME.vol / self.bME.l + min_A_tether_bme # area in bme region

        return A_bme, min_A_tether_bme

    def fnSetHeadgroupLength(self, idx, value):
        """ Sets headgroup with index 'idx' (base 1 like the label) to length 'value'. Required
            because initial headgroup lengths are tracked. Ignored if headgroup doesn't exist.
            
            Does not recalculate values using fnAdjustParameters."""
        
        if idx < len(self.lipid_nf):
            self.headgroup1[idx].l = value
            self.headgroup2[idx].l = value
            self.initial_hg1_lengths[idx] = value

    def fnSetSigma(self, sigma):
        super().fnSetSigma(sigma)
        self.substrate.fnSetSigma(self.global_rough)
        if self.tether_free.vol > 0:
            self.bME.fnSetSigma(self.global_rough)
            self.tether_bme.fnSetSigma(self.global_rough)
            self.tether_free.fnSetSigma(self.global_rough, sigma)
            self.tether_hg.fnSetSigma(sigma)
        else:
            self.bME.fnSetSigma(self.global_rough, sigma)
            self.tether_bme.fnSetSigma(self.global_rough, sigma)
            self.tether_hg.fnSetSigma(sigma)

    def fnGetLowerLimit(self):
        return self.substrate.fnGetLowerLimit()
        # does this make sense since this goes to negative z and isn't intended to be used?

    def fnSet(self, _sigma, _bulknsld, _global_rough, _rho_substrate, _nf_tether, _mult_tether, _l_tether, _l_lipid1,
              _l_lipid2, _vf_bilayer, _nf_lipids=None, _hc_substitution_1=0.,
              _hc_substitution_2=0., _radius_defect=100.):
        self.sigma = _sigma
        self.bulknsld = _bulknsld
        self.global_rough = _global_rough
        self.rho_substrate = _rho_substrate
        self.nf_tether = _nf_tether
        self.mult_tether = _mult_tether
        self.l_tether = _l_tether
        self.l_lipid1 = _l_lipid1
        self.l_lipid2 = _l_lipid2
        self.vf_bilayer = _vf_bilayer
        if _nf_lipids is not None:   # pass None to keep lipid_nf unchanged
            assert len(_nf_lipids)==len(self.lipids), \
                'nf_lipids must be same length as number of lipids in bilayer, not %i and %i' % \
                (len(_nf_lipids), len(self.lipids))
            self.lipid_nf = numpy.array(_nf_lipids)
        self.hc_substitution_1 = _hc_substitution_1
        self.hc_substitution_2 = _hc_substitution_2
        self.radius_defect = _radius_defect

        self.fnAdjustParameters()

# ------------------------------------------------------------------------------------------------------
# Hermite Spline
# ------------------------------------------------------------------------------------------------------

"""
Notes on usage:
1. Instantiate the spline object, e.g. h = SLDHermite()
    NB: the only initialization variable that isn't overwritten by fnSetRelative is dnormarea, so the others have been removed
        and dnSLD has been moved to fnSetRelative instead of the initialization. This is to keep the fnSetRelative function calls
        similar between Hermite and SLDHermite (except in Hermite nSLD is a constant and in SLDHermite it's a list of control points)
2. Set all internal parameters, e.g. damping parameters, monotonic spline, etc.
3. Call fnSetRelative.
    NB: for speed, the spline interpolators are stored in the object. Only fnSetRelative will update them!
"""

def catmull_rom(xctrl, yctrl, **kwargs):
    """returns a catmull_rom spline using ctrl points xctrl, yctrl"""
    assert(len(xctrl)==len(yctrl)) # same shape

    x_xtnd = numpy.insert(xctrl, 0, xctrl[0] - (xctrl[1]-xctrl[0]))
    x_xtnd = numpy.append(x_xtnd, xctrl[-1]-xctrl[-2] + xctrl[-1])

    y_xtnd = numpy.insert(yctrl, 0, yctrl[0])
    y_xtnd = numpy.append(y_xtnd, yctrl[-1])
    dydx = 0.5 * (y_xtnd[2:] - y_xtnd[:-2]) / (x_xtnd[2:] - x_xtnd[:-2])

    return CubicHermiteSpline(xctrl, yctrl, dydx, **kwargs)


class Hermite(nSLDObj):
    def __init__(self, dnormarea, **kwargs):
        super().__init__(**kwargs)
        self.numberofcontrolpoints = 10
        self.nSLD=None
        self.normarea=dnormarea
        self.monotonic=True
        self.damping=True
        self.dampthreshold=0.001
        self.dampFWHM=0.0002
        self.damptrigger=0.04
        self.dstartposition = 0.0
        
        self.dp     = numpy.arange(self.numberofcontrolpoints)
        self.vf     = numpy.zeros(self.numberofcontrolpoints)
        self.damp   = numpy.zeros(self.numberofcontrolpoints)

        # TODO: get the interface with a previous layer correct by defining an erf function at the first control point or between the first two control points.

    def _apply_damping(self):
        """ only called from _set_area_spline, which is called from fnSetRelative """
        dampfactor = numpy.ones_like(self.vf)
        if self.damping:
            above_damptrigger = numpy.where(self.vf > self.damptrigger)[0] # find index of first point beyond damping trigger
            if len(above_damptrigger) > 0 : # if no points above trigger, damping doesn't apply
                dampfactor[above_damptrigger[0] + 1:] = 1./(1+numpy.exp(-2.1*(self.vf[above_damptrigger[0] + 1:]-self.dampthreshold)/self.dampFWHM))    # does nothing if peaked point is at the end
                dampfactor = numpy.cumprod(dampfactor)
        
        self.damp = self.vf * dampfactor

    def _set_area_spline(self):

        self._apply_damping()

        if self.monotonic:  # monotone interpolation
            self.area_spline = PchipInterpolator(self.dp, self.damp, extrapolate=False)

        else:   # catmull-rom

            self.area_spline = catmull_rom(self.dp, self.damp, extrapolate=False)
        
        self.area_spline_integral = self.area_spline.antiderivative()

    def fnGetProfiles(self, z):

        vf = self.area_spline(z)
        vf[numpy.isnan(vf)] = 0.0
        vf[vf < 0] = 0.0
        area = vf * self.normarea * self.nf

        sld = self.fnGetnSLDProfile(z)

        return area, area * numpy.gradient(z) * sld, sld

    def fnGetnSLDProfile(self, z):

        return self.nSLD * numpy.ones_like(z)

    def fnGetnSLD(self, dz): 
        return self.nSLD

    def fnGetLowerLimit(self): 
        return self.dp[0]

    def fnGetUpperLimit(self): 
        return self.dp[-1]

    def fnGetVolume(self, z1, z2):
        """ use stored antiderivatives to calculate volume """
        # make sure z1 and z2 are in the defined interval. If both are above or below, result will be zero
        z1 = max(self.fnGetLowerLimit(), z1)
        z1 = min(self.fnGetUpperLimit(), z1)
        z2 = max(self.fnGetLowerLimit(), z2)
        z2 = min(self.fnGetUpperLimit(), z2)

        return (self.area_spline_integral(z2) - self.area_spline_integral(z1)) * self.nf * self.normarea

    def fnSetNormarea(self, dnormarea): 
        self.normarea = dnormarea

    def fnSetnSLD(self, dnSLD): 
        self.nSLD = dnSLD

    def fnSetRelative(self, dSpacing, dStart, dDp, dVf, dnSLD, dnf):
        self.fnSetnSLD(dnSLD)
        self.vf = numpy.array(dVf)
        self.numberofcontrolpoints = len(self.vf)
        self.dp = dStart + dSpacing*numpy.arange(self.numberofcontrolpoints) + numpy.array(dDp)
        # make sure the control points have compatible shapes (previous command should fail if not)
        assert(self.vf.shape == self.dp.shape)
        self.dstartposition = dStart
        self.nf = dnf
        self._set_area_spline()

    def fnWritePar2File(self, fp, cName, z):
        fp.write("Hermite "+cName+" numberofcontrolpoints "+str(self.numberofcontrolpoints)+" normarea "
                 + str(self.normarea)+" nf "+str(self.nf)+" \n")
        self.fnWriteData2File(fp, cName, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        rdict[cName] = {}
        rdict[cName]['header'] = "Hermite " + cName + " numberofcontrolpoints " + str(self.numberofcontrolpoints)
        rdict[cName]['header'] += " normarea " + str(self.normarea) + " nf " + str(self.nf)
        rdict[cName]['numberofcontrolpoints'] = self.numberofcontrolpoints
        rdict[cName]['normarea'] = self.normarea
        rdict[cName]['nf'] = self.nf
        rdict[cName] = self.fnWriteData2Dict(rdict[cName], z)
        return rdict


class SLDHermite(Hermite):
    def __init__(self, dnormarea, **kwargs):
        super().__init__(dnormarea, **kwargs)
        self.sld   = numpy.zeros(self.numberofcontrolpoints)

        # TODO: get the interface with a previous layer correct by defining an erf function at the first control
        #  point or between the first two control points.

    def _set_sld_spline(self):

        if self.monotonic:  # monotone interpolation
            self.sld_spline = PchipInterpolator(self.dp, self.sld, extrapolate=False)

        else:   # catmull-rom
            self.sld_spline = catmull_rom(self.dp, self.sld, extrapolate=False)

    def fnGetnSLDProfile(self, z):

        sld = self.sld_spline(z)
        # deal with out of range values
        sld[z < self.dp[0]] = self.sld[0]
        sld[z > self.dp[-1]] = self.sld[-1]
        assert(not numpy.any(numpy.isnan(sld))) # shouldn't be any NaNs left

        return sld

    def fnSetnSLD(self, dnSLD): 
        self.nSLD = None

    def fnSetRelative(self, dSpacing, dStart, dDp, dVf, dnSLD, dnf):
        self.vf = numpy.array(dVf)
        self.numberofcontrolpoints = len(self.vf)
        self.dp = dStart + dSpacing*numpy.arange(self.numberofcontrolpoints) + numpy.array(dDp)
        # make sure the control points have compatible shapes (previous command should fail if not)
        assert(self.vf.shape == self.dp.shape)
        self.sld = numpy.array(dnSLD)
        assert(self.sld.shape == self.dp.shape)
        self.dstartposition = dStart
        self.nf = dnf
        self._set_area_spline()
        self._set_sld_spline()


class ContinuousEuler(nSLDObj):
    """ Uses scipy.spatial library to do Euler rotations in real time"""
    def __init__(self, fn8col, rotcenter=None, **kwargs):
        """ requires file name fn8col containing 8-column data, any header information commented with #:
            1. residue number
            2. x coordinate
            3. y coordinate
            4. z coordinate
            5. residue volume
            6. electon scattering length
            7. neutron scattering length (H)
            8. neutron scattering length (D)

            Use helper function pdbto8col to create this data file

            The simplest approach is to provide coordinates relative to the center of mass; then,
            "z" corresponds to the absolute z position of the center of mass of the object. Otherwise,
            if the rotation center "rotcenter" [x0, y0, z0] is specified, all coordinates are translated
            by "rotcenter" and "z" is the absolute z position of the rotation center.
        """
        super().__init__(**kwargs)
        self.fn = fn8col
        resdata = numpy.loadtxt(fn8col)
        self.resnums = resdata[:,0]
        self.rescoords = resdata[:,1:4]
        if rotcenter is not None:
            rotcenter = numpy.array(rotcenter)
            assert rotcenter.shape == self.rescoords[0,:].shape
            self.rescoords -= rotcenter
        self.rotcoords = numpy.zeros_like(self.rescoords)
        self.resscatter = resdata[:, 4:]
        self.gamma = 0.
        self.beta = 0.
        self.sigma = 2.
        self.z = 0.
        self.nf = 1.
        self.protexchratio = 1.
        self.R = Rotation.from_euler('zy', [self.gamma, self.beta], degrees=True)

        # TODO: Would it make sense to have a concept of "normarea"? Then there could be a "volume fraction" concept so that
        # max(area) = volume_fraction * normarea

    def _apply_transform(self):
        
        self.R = Rotation.from_euler('zy', [self.gamma, self.beta], degrees=True)
        self.rotcoords = self.R.apply(self.rescoords)
        self.rotcoords[:,2] += self.z

    def fnGetProfiles(self, z, bulknsld=None, xray=False):
        # xray=True for xray profile
        # xray=False (default) for a neutron probe; bulknsld determines fraction of exchangeable hydrogens used

        # get area profile
        dz = z[1] - z[0]                            # calculate step size (MUST be uniform)
        zbins = numpy.append(z, z[-1]+dz) - 0.5 * dz    # bin edges; z is treated as bin centers here
        # TODO: Smoothing with gaussian_filter requires constant step size. What happens if z is a single point?
        h = numpy.histogram(self.rotcoords[:,2], bins=zbins, weights=self.resscatter[:,0])
        area = gaussian_filter(h[0], self.sigma / dz, order=0, mode='constant', cval=0)
        area /= dz

        if xray:
            h = numpy.histogram(self.rotcoords[:,2], bins=zbins, weights=self.resscatter[:,1])
            nsl = gaussian_filter(h[0], self.sigma / dz, order=0, mode='constant', cval=0)
        else:
            # get nslH profile
            h = numpy.histogram(self.rotcoords[:,2], bins=zbins, weights=self.resscatter[:,2])
            nslH = gaussian_filter(h[0], self.sigma / dz, order=0, mode='constant', cval=0)

            if bulknsld is not None:
                # get nslD profile
                h = numpy.histogram(self.rotcoords[:,2], bins=zbins, weights=self.resscatter[:,3])
                nslD = gaussian_filter(h[0], self.sigma / dz, order=0, mode='constant', cval=0)
                fracD = self.protexchratio * (bulknsld + 0.56e-6) / (6.36e-6 + 0.56e-6)
                nsl = fracD * nslD + (1 - fracD) * nslH

            else:
                nsl = nslH

        nsld = numpy.zeros_like(z)
        pos = (area > 0)
        nsld[pos] = nsl[pos] / (area[pos] * numpy.gradient(z)[pos])

        return area * self.nf, nsl * self.nf, nsld

    def fnGetVolume(self, z1, z2):
        """ Calculates volume based on the number of residues of the rotated molecule located between
            z positions z1 and z2 (inclusive).
            
            Note: the result is (slightly) different from integrating the area, because the roughness has already
            been applied to the area. However, this remains more accurate than the roughened value because
            the integration limits  will typically correspond to the limits of a lipid Box2Err function
            which are themselves defined before the roughness is applied."""
        # use a single bin defined by bin edges z1 and z2.
        # First [0] selects the histogram array, second [0] gets the single value from the array

        volume = numpy.histogram(self.rotcoords[:,2], bins=[z1,z2], weights=self.resscatter[:,0])[0][0]

        return volume * self.nf

    def fnSet(self, gamma, beta, zpos, sigma, nf):

        self.gamma = gamma
        self.beta = beta
        self.z = zpos
        self.nf = nf
        self.sigma = sigma
        self._apply_transform()

    def fnWritePar2File(self, fp, cName, z):
        fp.write("ContinuousEuler "+cName+" StartPosition "+str(self.z)+" Gamma "
                 +str(self.gamma)+" Beta "+str(self.beta) +" nf "+str(self.nf)+" \n")
        self.fnWriteData2File(fp, cName, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        rdict[cName] = {}
        rdict[cName]['header'] = "ContinuousEuler " + cName + " StartPosition " + str(self.z) + " Gamma "
        rdict[cName]['header'] += str(self.gamma) + " Beta " + str(self.beta) + " nf " + str(self.nf)
        rdict[cName]['startposition'] = self.z
        rdict[cName]['gamma'] = self.gamma
        rdict[cName]['beta'] = self.beta
        rdict[cName]['nf'] = self.nf
        rdict[cName] = self.fnWriteData2Dict(rdict[cName], z)
        return rdict


aa3to1 = dict({'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
               'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
               'THR': 'T', 'TYR': 'Y', 'VAL': 'V', 'TRP': 'W'})


def pdbto8col(pdbfilename, datfilename, selection='all', center_of_mass=numpy.array([0,0,0]), deuterated_residues=None,
              xray_wavelength=1.5418):
    """ Creates an 8-column data file for use with ContinuousEuler from a pdb file\
        with optional selection. "center_of_mass" is the position in space at which to position the
        molecule's center of mass. "deuterated_residues" is a list of residue IDs for which to use deuterated values"""

    import MDAnalysis
    from MDAnalysis.lib.util import convert_aa_code
    from periodictable.fasta import Molecule, default_table, AMINO_ACID_CODES as aa

    elements = default_table()

    molec = MDAnalysis.Universe(pdbfilename)
    sel = molec.select_atoms(selection)
    Nres = sel.n_residues
    
    if not Nres:
        print('Warning: no atoms selected')

    sel.translate(-sel.center_of_mass() + center_of_mass)
   
    resnums = []
    rescoords = []
    resscatter = []
    resvol = numpy.zeros(Nres)
    resesl = numpy.zeros(Nres)
    resnslH = numpy.zeros(Nres)
    resnslD = numpy.zeros(Nres)
    deut_header = ''

    for i in range(Nres):
        resnum = molec.residues[i].resid
        resnums.append(resnum)
        rescoords.append(molec.residues[i].atoms.center_of_mass())
        key = convert_aa_code(molec.residues[i].resname)
        if resnum in deuterated_residues:
            resmol = Molecule(name='Dres', formula=aa[key].formula.replace(elements.H, elements.D), cell_volume=aa[key].cell_volume)
        else:
            resmol = aa[key]
        resvol[i] = resmol.cell_volume
        resesl[i] = xray_sld(resmol.formula, wavelength=xray_wavelength)
        resnslH[i] = resmol.sld
        resnslD[i] = resmol.Dsld
            
    resnums = numpy.array(resnums)
    rescoords = numpy.array(rescoords)

    # replace base value in nsl calculation with proper deuterated scattering length\
    resnsl = resscatter[:,2]
    if deuterated_residues is not None:
        deut_header = 'deuterated residues: ' + ', '.join(map(str, deuterated_residues)) + '\n'

    numpy.savetxt(datfilename, numpy.hstack((resnums[:,None], rescoords, resvol[:,None], resesl[:,None], resnslH[:, None], resnslD[:,None])), delimiter='\t',
            header=pdbfilename + '\n' + deut_header + 'resid\tx\ty\tz\tvol\tesl\tnslH\tnslD')

    return datfilename      # allows this to be fed into ContinuousEuler directly


class DiscreteEuler(nSLDObj):
    """ Uses precalculated Euler rotation densities"""
    def __init__(self, generic_filename, betas, gammas, betafmt='%i', gammafmt='%i', **kwargs):
        """ requires precalculated 4-column files for each angle beta and gamma. These should not have any smoothing applied.
            Each file must have same z vector and one header row. z can be negative.
            generic_filename contains the path and a generic file name with positions for angles beta and gamma are marked with <beta> and <gamma>.
            Format strings other than integer angles can be specified with betafmt and gammafmt keyword arguments.
            "betas" and "gammas" are lists or numpy arrays containing the beta and gamma points to be loaded.

        Example: DiscreteEuler('./dat/exou_beta<beta>_gamma<gamma>.txt', range(0, 190, 10), range(0, 370, 10))
        """
        super().__init__(**kwargs)
        betas = numpy.array(betas, dtype=float)
        gammas = numpy.array(gammas, dtype=float)
        areadata = None
        for i, beta in enumerate(betas):
            for j, gamma in enumerate(gammas):
                fn = generic_filename.replace('<beta>', betafmt % beta).replace('<gamma>', gammafmt % gamma)
                d = numpy.loadtxt(fn, skiprows=1)
                if areadata is None:
                    zdata = d[:, 0]
                    areadata = numpy.zeros((len(betas), len(gammas), len(zdata)))
                    nslHdata = numpy.zeros_like(areadata)
                    nslDdata = numpy.zeros_like(areadata)
                areadata[i, j, :] = d[:,1]
                nslHdata[i, j, :] = d[:,2]
                nslDdata[i, j, :] = d[:,3]

        self.betas = betas
        self.gammas = gammas
        self.areadata = areadata
        self.zdata = zdata
        self.nslHdata = nslHdata
        self.nslDdata = nslDdata
        self.beta = 0.         # current beta rotation
        self.gamma = 0.          # current gamma rotation
        self.sigma = 2.         # smoothing function
        self.z = 0.             # z offset value
        self.nf = 1.            # number fraction
        self.protexchratio = 1. # proton exchange ratio

        # TODO: Would it make sense to have a concept of "normarea"? Then there could be a "volume fraction"
        # concept so that max(area) = volume_fraction * normarea

    def fnGetProfiles(self, z, bulknsld=None):

        # perform interpolation
        self._set_interpolation_points(z)
        area = interpn((self.betas, self.gammas, self.zdata + self.z), self.areadata, self._interppoint, bounds_error=False, fill_value=0.0)
        nslH = interpn((self.betas, self.gammas, self.zdata + self.z), self.nslHdata, self._interppoint, bounds_error=False, fill_value=0.0)
        nslD = interpn((self.betas, self.gammas, self.zdata + self.z), self.nslDdata, self._interppoint, bounds_error=False, fill_value=0.0)

        # apply roughness (sigma)
        dz = z[1] - z[0]
        area = gaussian_filter(area, self.sigma / dz, order=0, mode='constant', cval=0)
        # area /= dz

        nslH = gaussian_filter(nslH, self.sigma / dz, order=0, mode='constant', cval=0)

        if bulknsld is not None:
            # get nslD profile
            nslD = gaussian_filter(nslD, self.sigma / dz, order=0, mode='constant', cval=0)
            fracD = self.protexchratio * (bulknsld + 0.56e-6) / (6.36e-6 + 0.56e-6)
            nsl = fracD * nslD + (1 - fracD) * nslH

        else:
            nsl = nslH

        nsld = numpy.zeros_like(z)
        pos = (area > 0)
        nsld[pos] = nsl[pos] / (area[pos] * numpy.gradient(z)[pos])

        return area * self.nf, nsl * self.nf, nsld

    def _set_interpolation_points(self, z):
        self._interppoint = numpy.zeros((len(z), 3))
        self._interppoint[:,:-1] = [self.beta, self.gamma]
        self._interppoint[:, -1] = z

    def fnGetVolume(self, z1, z2):
        """ Calculates volume based on the number of residues of the rotated molecule located between
            z positions z1 and z2 (inclusive).
            
            Note: the result is (slightly) different from integrating the area, because the roughness has already
            been applied to the area. However, this remains more accurate than the roughened value because
            the integration limits  will typically correspond to the limits of a lipid Box2Err function
            which are themselves defined before the roughness is applied."""

        # For volume calculation, use intrinsic z vector, and no smoothing.
        zvol = self.zdata + self.z
        self._set_interpolation_points(zvol)

        area = interpn((self.betas, self.gammas, zvol), self.areadata, self._interppoint, bounds_error=False, fill_value=0.0)

        crit = (zvol > min(z1, z2)) & (zvol < max(z1, z2))
        if numpy.any(crit):
            volume= numpy.trapz(area[crit], zvol[crit])
        else:
            volume = 0.0

        return volume * self.nf

    def fnSet(self, beta, gamma, zpos, sigma, nf):

        self.beta = beta
        self.gamma = gamma
        self.z = zpos
        self.nf = nf
        self.sigma = sigma

    def fnWritePar2File(self, fp, cName, z):
        fp.write("DiscreteEuler "+cName+" StartPosition "+str(self.z)+" beta "
                 +str(self.beta)+" gamma "+str(self.gamma) +" nf "+str(self.nf)+" \n")
        self.fnWriteData2File(fp, cName, z)

    def fnWritePar2Dict(self, rdict, cName, z):
        rdict[cName] = {}
        rdict[cName]['header'] = "DiscreteEuler " + cName + " StartPosition " + str(self.z) + " beta "
        rdict[cName]['header'] += str(self.beta) + " gamma " + str(self.gamma) + " nf " + str(self.nf)
        rdict[cName]['startposition'] = self.z
        rdict[cName]['beta'] = self.beta
        rdict[cName]['gamma'] = self.gamma
        rdict[cName]['nf'] = self.nf
        rdict[cName] = self.fnWriteData2Dict(rdict[cName], z)
        return rdict


