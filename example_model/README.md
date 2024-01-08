# Read Me for example models

## Installation
To use these models, first install the molgroups package:
```
pip install git+https://github.com/criosx/molgroups.git@440e542
```

## Running an autonomous experiment
Copy the contents of ```example_model``` into a different location. Navigate to that folder and then run the simulation via the command line interface, e.g.:

Autonomous experiment:

```
python -m autorefl.runauto_cli ssblm.py --pars ssblm_tosb0.par --name both_forecast_q025 --sel 10 11 12 13 14 --eta 0.50 --meas_bkg 3e-6 3e-5 --maxtime 43.2e3 --burn 1000 --steps 500 --pop 8 --instrument MAGIK --qmax 0.25 --qstep_max 0.0024
```

Control experiment:

```
python -m autorefl.runauto_cli ssblm.py --pars ssblm_tosb0.par --name both_q025 --sel 10 11 12 13 14 --meas_bkg 3e-6 3e-5 --maxtime 200e3 --burn 2000 --steps 500 --pop 8 --instrument MAGIK --control --nctrlpts 31 --qmax 0.25 --qstep_max 0.0024
```