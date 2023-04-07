import numpy
from periodictable.fasta import Molecule, xray_sld
# in formulas, use H[1] for exchangeable hydrogens, then Molecule.sld, Molecule.Dsld
# to get limiting slds, or Molecule.D2Osld(D2O_fraction=??) to get arbitrary D2O fraction.

# ============ Class library ============

class Component(Molecule):
    # Subclasses Molecule to automatically store a component length for use later
    # and calculate total neutron scattering lengths
    def __init__(self, length=9.575, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.nSLs = self.fnGetnSL()

    def fnGetnSL(self, xray_wavelength=None):
        if xray_wavelength is None:
            return numpy.array([self.sld * self.cell_volume * 1e-6, self.Dsld * self.cell_volume * 1e-6])
        else:
            return numpy.array([xray_sld(self.formula, wavelength=xray_wavelength)[0] * self.cell_volume * 1e-6])

# null components and molecules for use with bilayer species that do not have headgroups or methyls (e.g. cholesterol)
null_molecule = Molecule(name=None, formula='', cell_volume=0.0)
null_component = Component(name=None, formula='', cell_volume=0.0, length=9.575)

class Lipid(object):
    def __init__(self, headgroup, tails, methyls, name=None):
        # hg = Molecule object with headgroup information or PC molgroups object
        # tails = List of molecule objects containing lipid tail information
        # methyls = List of methyl groups, one for each lipid tail; OR, a single group
        #           that is copied to each tail; OR, None, which uses a null group with
        #           no volume (use with cholesterol). Use methyl=Dmethyl for CD3.

        n_tails = len(tails)

        # tails section
        if isinstance(tails, list):
            tail_volume = sum([t.cell_volume for t in tails])
            tail_formula = ' '.join([str(t.formula) for t in tails])
            self.tails = Molecule(name='tails', formula=tail_formula, cell_volume=tail_volume)
        elif isinstance(tails, Molecule):
            self.tails = tails
        else:
            raise TypeError('Lipid.tails must be Molecule or list of Molecules')

        # headgroup
        self.headgroup = headgroup if headgroup is not None else null_component

        # Create methyl groups
        if isinstance(methyls, list):
            assert n_tails == len(methyls), 'Lipid tails and lipid methyl lists must have equal length, not %i and %i' \
                                            % (len(tails), len(methyls))
        elif isinstance(methyls, Molecule) | (methyls is None):
            methyls = n_tails * [methyls]
        else:
            raise TypeError('Lipid.methyls must be Molecule or list of Molecules')
        # Replace None with null molecule
        methyls = [m if m is not None else null_molecule for m in methyls]
        methylvolsum = sum([m.cell_volume for m in methyls])
        methylformula = ' '.join([str(m.formula) for m in methyls])
        self.methyls = Molecule(name='methyls', formula=methylformula, cell_volume=methylvolsum)

        if name is not None:
            self.name = name
        else:
            self.name = ' + '.join([c.name for c in (tails + [self.headgroup]) if c.name is not None])


class Tether(Lipid):
    """Subclass of Lipid for use with tether molecules.
        Uses hg field for tether glycerol.
        Tether includes both volume from the surface-attachment group (e.g. thiol)
            and the length of the separating polymer"""

    def __init__(self, tether, tetherg, tails, methyls, name=None):

        super().__init__(name=name, headgroup=tetherg, tails=tails, methyls=methyls)
        self.tether = tether
        self.tetherg = self.headgroup

        if name is not None:
            self.name = name
        else:
            self.name = ' + '.join([c.name for c in ([self.tether] + tails) if c.name is not None])

def AddMolecules(component_list, length=None):
    # Adds together components and molecules. Note that length information is lost
    # if length is not specified
    total_formula = ' '.join([str(c.formula) for c in component_list])
    total_cell_volume = sum([c.cell_volume for c in component_list])
    total_name = ' + '.join([c.name for c in component_list])
    if length is None:
        return Molecule(name=total_name, formula=total_formula, cell_volume=total_cell_volume)
    else:
        return Component(name=total_name, formula=total_formula, cell_volume=total_cell_volume, length=length)



# ========== Molecule library ============

# PC headgroup pieces
choline = Component(name='choline', formula='C5 H13 N', cell_volume=120., length=6.34)
phosphate = Component(name='phosphate', formula='PO4', cell_volume=54., length=3.86)
carbonyl_glycerol = Component(name='carbonyl + glycerol', formula='C5 O4 H5', cell_volume=147., length=4.21)

# standard headgroups
pc = Component(name='PC', formula='C10 H18 O8 N P', cell_volume=331.00, length=9.575)
pe = Component(name='PE', formula='C7 H9 H[1]3 O8 N P', cell_volume=262., length=7.7)
pg = Component(name='PG', formula='C8 H10 H[1]2 O10 P', cell_volume=240., length=7.8)
ps = Component(name='PS', formula='C8 H8 H[1]3 N O10 P', cell_volume=280., length=8.1)
pa = Component(name='PA', formula='C5 H5 H[1] O8 P', cell_volume=174., length=5.0)
pi = Component(name='PI', formula='C11 H7 H[1]5 O13 P', cell_volume=370.7, length=10.7)
pip2 = Component(name='PI(4,5)P2', formula='C11 H7 H[1]5 O19 P3', cell_volume=500., length=12.0)  # diff to molgroups.cc
cardiolipin = Component(name='cardiolipin', formula='C13 H15 H[1] O17 P2', cell_volume=684.4, length=9.56)

# standard acyl chains
oleoyl = Molecule(name='oleoyl', formula='C17 H33', cell_volume=972./2.0)
palmitoyl = Molecule(name='palmitoyl', formula='C15 H31', cell_volume=770./2.0)
myristoyl = Molecule(name='myristoyl', formula='C13 H27', cell_volume=770./2.0)
phytanoyl = Molecule(name='phytanoyl', formula='C19 H39', cell_volume=1095./2.0)
cholesterol = Molecule(name='cholesterol', formula='C27 H45 H[1] O', cell_volume=630.)

# methyl
methyl = Molecule(name='methyl', formula='CH3', cell_volume=98.8/2.0)
Dmethyl = Molecule(name='Dmethyl', formula='CD3', cell_volume=98.8/2.0)

# Tether components
SAc = Molecule(name='thiol acetate', formula='C2H3OS', cell_volume=117.0)
EO6 = Molecule(name='6x ethylene oxide', formula='(C2H4O)6', cell_volume=360.0)
tetherg_ether = Molecule(name='tether glycerol ether', formula='C5H9O2', cell_volume=125.40)
tetherg_ester = carbonyl_glycerol
ethanoyl = Molecule(name='ethanoyl', formula='C2H5O', cell_volume=(117 - 25.75))
thiol = Molecule(name='sulfur', formula='S', cell_volume=25.75)
SEO6 = AddMolecules([thiol, EO6])
SAcEO6 = AddMolecules([SAc, EO6])

# Filler molecules. Note that by including length= these return Components as required by molgroups.tBLM
bmeSAc = AddMolecules([SAc, ethanoyl], length=5.2)
bme = AddMolecules([thiol, ethanoyl], length=5.2)

# Combined lipids
DOPC = Lipid(name='DOPC', headgroup=pc, tails=2 * [oleoyl], methyls=methyl)
POPC = Lipid(name='POPC', headgroup=pc, tails=[palmitoyl, oleoyl], methyls=methyl)
DOPS = Lipid(name='DOPS', headgroup=ps, tails=2 * [oleoyl], methyls=methyl)
chol = Lipid(name='chol', headgroup=None, tails=[cholesterol], methyls=None)

# Combined tethers
WC14 = Tether(name='WC14', tether=SEO6, tetherg=tetherg_ether, tails=[myristoyl, myristoyl], methyls=methyl)
HC18 = Tether(name='HC18', tether=SEO6, tetherg=tetherg_ether, tails=[oleoyl, oleoyl], methyls=methyl)
HC18SAc = Tether(name='HC18SAc', tether=SAcEO6, tetherg=tetherg_ether, tails=[oleoyl, oleoyl], methyls=methyl)
