import sys, os
sys.path.append('code')
from help_funcs import *

simname = 'base'

sim_params = read_sim_params('parameter_sets/base.txt')

for del_range in [0.0, 0.3, 1.0]:
    sim_params['del_range'] = del_range
    simdir = f'work_dir/dl_%s'%del_range
    try:
        os.mkdir(simdir)
    except FileExistsError:
        print(simdir, 'exists.')
    
    print_params(f'{simdir}/sim_parameters.txt', sim_params)
    
    print(f'parameter file printed at {simdir}')
