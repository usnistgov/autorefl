# Autonomous Neutron Reflectometry

Framework for driving neutron autonomous reflectometry experiments. The overarching goal is to use autonomous experimentation techniques to measure a 1-D reflectometry pattern, R(Q), with maximum efficiency.

## Background

Neutron reflectometry measurements involve measuring the reflection of neutrons from structured interfaces. Measurements involve measuring a 1-D curve, R(Q). For a general reflectometer, the 1-D measurement coordinate is a general coordinate x with a transformation from x -> Q (sometimes multiple Q values per x).

## List of important files
simexp.py -- library containing SimReflExperiment and SimReflExperimentControl objects

instrument.py -- instrument definitions for monochromatic and polychromatic reflectometers

autorefl.py -- library of helper functions for simulating data and calculating entropies

runauto_cli.py -- command line interface for running simulations with an arbitrary model and simulation parameters

## Example usage of command line interface
Autonomous experiment:

python runauto_cli.py ssblm.py --pars ssblm_tosb0.par --name both_forecast_q025 --sel 10 11 12 13 14 --eta 0.50 --meas_bkg 3e-6 3e-5 --maxtime 43.2e3 --burn 1000 --steps 500 --pop 8 --instrument MAGIK --qmax 0.25 --qstep_max 0.0024

Control experiment:

python runauto_cli.py ssblm.py --pars ssblm_tosb0.par --name both_q025 --sel 10 11 12 13 14 --meas_bkg 3e-6 3e-5 --maxtime 200e3 --burn 2000 --steps 500 --pop 8 --instrument MAGIK --control --nctrlpts 31 --qmax 0.25 --qstep_max 0.0024
