import sys, os
sys.path.append('code')
from help_funcs import *

simname = 'base'

try:
    alpha_value = float(sys.argv[1])
    sst_longrange_value = float(sys.argv[2])
except IndexError:
    print('This script initializes directory and parameter file for simulation.')
    print('Pass alpha and sst long range parameters as inputs.')

sim_params = read_sim_params('parameter_sets/base.txt')

sim_params['alpha_ei'] = alpha_value
sim_params['p_far_s'] = sst_longrange_value

simdir = f'work_dir/{simname}_a{alpha_value}_pf{sst_longrange_value}'
try:
    os.mkdir(simdir)
except FileExistsError:
    print(simdir, 'exists.')

print_params(f'{simdir}/sim_parameters.txt', sim_params)

print(f'parameter file printed at {simdir}')
