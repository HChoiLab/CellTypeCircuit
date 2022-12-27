from os.path import exists
from time import sleep, time
import os
import pickle

def run_sim(opt_name, simname, sim_parameters, print_results):

    simdir = '%s/%s'%(opt_name,simname)
    os.mkdir(simdir)
    with open(f'{simdir}/sim_parameters.pickle', 'wb') as f:
        pickle.dump(sim_parameters, f)

    sname = f'{opt_name}/qfiles/{simname}.sbatch'
    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -J%s\n'''%simdir)
    f.write('''#SBATCH --account=gts-hchoi387\n''')
    f.write('''#SBATCH -N1 --ntasks-per-node=1\n''')
    f.write('''#SBATCH --mem-per-cpu=4G\n''')
    f.write('''#SBATCH -t30\n''')
    f.write('''#SBATCH -qembers\n''')
    f.write(f'#SBATCH -o{opt_name}/reports/Report-{simname}-%j.out\n')
    f.write('''cd ~/scratch/circuit_gd/%s\n'''%simdir)
    f.write('''module load anaconda3\n''')
    f.write('''conda activate nest\n''')
    if print_results:
        f.write('''python ../../run_sim.py 1\n''')
    else:
        f.write('''python ../../run_sim.py 0\n''')
    f.close()

    os.system(f'sbatch {sname}')

def run_batch(opt_name, simnum, paramlist, submit_lim = 40, print_results = True):
    print('RUN BATCH:', len(paramlist))
    nsims = len(paramlist)
    start_num = 0
    stop_num = 0
    results = []

    while stop_num < nsims:
        batch_size = min(submit_lim, nsims-start_num) 
        stop_num = start_num + batch_size
        param_batch = paramlist[start_num:stop_num]
        
        simnum_list = [i for i in range(simnum+start_num,simnum+stop_num)]
        assert len(param_batch) == len(simnum_list)
        results += run_batch_sub(opt_name, simnum_list, param_batch, print_results)
        start_num = stop_num
    print('batch done.')

    return results

def run_batch_sub(opt_name, simnum_list, paramlist, print_results):
    nsims = len(paramlist)
    check_files = []
    for si, simnum in enumerate(simnum_list):
        sim_param = paramlist[si]
        run_sim(opt_name, simnum, sim_param, print_results)
        check_files.append('%s/%s/result.pickle'%(opt_name, simnum))

    found_list = [False for si in range(nsims)]
    results_list = [0 for si in range(nsims)]

    print('running %s sims:'%len(check_files),check_files)
    
    start_time = time()
    while False in found_list:
        add_found = False
        for si in range(nsims):
            if found_list[si]: continue
            if exists(check_files[si]):
                add_found = True
                print(check_files[si], 'found.')
                found_list[si] = True

                with open(check_files[si],'rb') as f:
                    results_list[si] = pickle.load(f)

        not_found_simnums = []
        not_found_paramlist = []
        if add_found:
            print(found_list.count(True), 'Found. Still waiting for:')
            for si in range(nsims):
                if found_list[si] == False:
                    print(check_files[si] , end=', ')
                    not_found_simnums.append(simnum_list[si])
                    not_found_paramlist.append(paramlist[si])
            print("%s left."%found_list.count(False))
        sleep(10)
        time_passed = time() - start_time
        if (not add_found) and (time_passed > 60*30):
            if len(not_found_simnums) > 0:
                print("%s seconds passed. RESUBMITTING %s JOBS"%(time_passed, len(not_found_simnums)))
                for sj, simnum in enumerate(not_found_simnums):
                    sim_param = not_found_paramlist[sj]
                    run_sim(opt_name, simnum, sim_param, print_results)
                    check_files.append('%s/%s/result.pickle'%(opt_name, simnum))
                start_time = time()
            
    return results_list

