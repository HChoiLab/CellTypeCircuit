from os.path import exists
from time import sleep, time
import os
import pickle
import subprocess
import sys
sys.path.append('/storage/home/hcoda1/5/skim3438/scratch/CellTypeCircuit/newCode/code')
from help_funcs import *

cmd = ['squeue', '-u', 'skim3438', '-t', 'running', '-h', '-o', '%i']

# Execute the squeue command and capture the output
output = subprocess.check_output(cmd).decode().strip()
num_jobs = len(output.splitlines())


def run_sim(simname, ri, ciseeds, sim_parameters, pace_queue, run_name):
    # simname is the name of my directory
    # sim_parameters is a dictionary of simulation parameters
    # ri is an index number
    # ciseeds is a list of lists of the form [ci, seed] where ci and seed are inputs to the simulation program

    simdir = '/storage/home/hcoda1/5/skim3438/scratch/CellTypeCircuit/newCode/data/%s'%simname
    try: 
        os.mkdir(simdir)
    except FileExistsError:
        pass

    print_params(f'{simdir}/sim_parameters.txt', sim_parameters)

    sname = f'qfiles/{simname}_{ri}.sbatch'
    f = open(sname,'w')
    f.write('''#!/bin/bash\n''')
    f.write('''#SBATCH -J%s_%s\n'''%(simname,ri))
    f.write('''#SBATCH --account=gts-hchoi387\n''')
    f.write('''#SBATCH -N1 --ntasks-per-node=4\n''')
    f.write('''#SBATCH --mem-per-cpu=16G\n''')
    f.write('''#SBATCH -t4:00:00\n''')
    f.write('''#SBATCH -q%s\n'''%pace_queue)
    f.write(f'#SBATCH -oreports/{simname}-{ri}-%j.out\n')
    f.write('''cd %s\n'''%simdir)
    f.write('''module load anaconda3\n''')
    f.write('''conda activate nest3\n''')
    for ci, seed in ciseeds:
        f.write('''python ../../code/%s.py %s %s\n'''%(run_name, seed, ci))
    f.close()

    os.system(f'sbatch {sname}')

contrast_values = [0.02, 0.05,0.1,0.18, 0.33]
conditions =  [['Spont',0]] +[  ['PV', c] for c in contrast_values] + [ ['SOM', c] for c in contrast_values] 

def sim_results_exist(simname, seed, ci):
    simdir = '/storage/home/hcoda1/5/skim3438/scratch/CellTypeCircuit/newCode/data/%s'%simname
    stim_type, contrast = conditions[ci]
    if stim_type == 'Spont':
        sim_name = 'Spont'
    elif stim_type in ['PV', 'SOM']:
        sim_name = f'{stim_type}_{contrast}'
    else:
        raise Exception()
    
    if os.path.isfile(simdir+'/results_%s/%s_spikes.pickle'%(seed,sim_name)):
        if os.path.isfile(simdir+'/results_%s/%s_positions.pickle'%(seed,sim_name)):
            return True
    return False

runname = 'run_single_v2'
parname = sys.argv[1]

sim_params = read_sim_params(parname+'.txt')

epsilonlist = [0.05,0.1,0.15,0.2]
seedlist = [ii for ii in range(1,11)]

par_ext_syn_1 = sim_params['par_ext_syn_1']
par_ext_syn_2 = sim_params['par_ext_syn_2']
par_ext_syn_3 = sim_params['par_ext_syn_3']
par_ext_syn_4 = sim_params['par_ext_syn_4']

ns_list = [1.0]
epsilonlist = [0.2]

maxjobs = 40

ri = 0
notdone = False
while True:

    output = subprocess.check_output(cmd).decode().strip()
    num_jobs = len(output.splitlines())
    print(f"Number of jobs running: {num_jobs}. Submit {maxjobs-num_jobs} jobs.")
    for ns in ns_list:
        sim_params['par_ext_syn_1'] = par_ext_syn_1*ns
        sim_params['par_ext_syn_2'] = par_ext_syn_2*ns
        sim_params['par_ext_syn_3'] = par_ext_syn_3*ns
        sim_params['par_ext_syn_4'] = par_ext_syn_4*ns
        for epsilon in epsilonlist:
            sim_params['epsilon'] = epsilon
            simname = 'l%sns%se%s'%(parname,ns, epsilon)
            ciseeds = []
            for ci in range(1,11):
                for seed in seedlist:
                    #check if sim is done
                    if sim_results_exist(simname, seed, ci):
                        print(simname, seed, ci, 'found.')
                        continue
                    else:
                        print('adding', simname, seed, ci)
                        ciseeds.append([ci, seed])
    
            chunk_size = 1
            chunks = [ciseeds[i:i+chunk_size] for i in range(0, len(ciseeds), chunk_size)]
            #chunks = [[ciseeds[ijk]] for ijk in range(len(ciseeds))]
    
            for cis in chunks:
                run_sim(simname, ri, cis, sim_params, 'embers', runname)
                ri += 1
                num_jobs +=1  
                if num_jobs >= maxjobs: 
                    notdone = True
                    break
            
            if num_jobs >= maxjobs:
                notdone = True
                break

        if num_jobs >= maxjobs:
            notdone = True
            break
    
    while notdone:
        print('sleeping...', end='')
        sleep(10)
        try:
            output = subprocess.check_output(cmd).decode().strip()
            num_jobs = len(output.splitlines())
            print(f"Number of jobs running: {num_jobs}.")
        except:
            print("squeue didn't work, trying again.")
            continue
        if num_jobs < 3:
            print('squeue empty, submit jobs.')
            break

    if notdone: continue
    else: break