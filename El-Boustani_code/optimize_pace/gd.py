from os.path import exists
import os, sys
from time import time
import ctypes
import itertools

from help_funcs import *
from run_funcs import *

use_power_law = False
result_dir = sys.argv[1]



def get_paramlist(sim_pars, simnum, trialnum):
    #somlist = list(itertools.product( [0.02,0.2,0.4,0.6,0.8,1.0], ['SOM'], sim_pars))
    #pvlist = list(itertools.product( [0.02,0.2,0.4,0.6,0.8,1.0], ['PV'], sim_pars))
    paramlist = []
    sim_pars_add = sim_pars.copy()
    sim_pars_add['stim_type'] = 'Spont'
    paramlist.append( sim_pars_add )
    simnum += 1
    for cont in contrast_values:
        sim_pars_add = sim_pars.copy()
        sim_pars_add['stim_type'] = 'PV'
        sim_pars_add['contrast'] = cont
        paramlist.append( sim_pars_add )
        simnum +=1

    for cont in contrast_values:
        sim_pars_add = sim_pars.copy()
        sim_pars_add['stim_type'] = 'SOM'
        sim_pars_add['contrast'] = cont
        paramlist.append( sim_pars_add )
        simnum +=1

    return simnum, paramlist

def get_paramlist_old(sim_pars, simnum, trialnum):
    #somlist = list(itertools.product( [0.02,0.2,0.4,0.6,0.8,1.0], ['SOM'], sim_pars))
    #pvlist = list(itertools.product( [0.02,0.2,0.4,0.6,0.8,1.0], ['PV'], sim_pars))
    paramlist = []
    paramlist.append( ('%s_%s'%(trialnum,simnum), 0., 'Spont', sim_pars) )
    simnum += 1
    for cont in contrast_values:
        paramlist.append( ('%s_%s'%(trialnum,simnum), cont, 'PV', sim_pars) )
        simnum +=1

    for cont in contrast_values:
        paramlist.append( ('%s_%s'%(trialnum,simnum), cont, 'SOM', sim_pars) )
        simnum +=1

    return simnum, paramlist

def calculate_rms(results):
    spont_rates = results[0]
    pv_rates = results[1:1+len(contrast_values)]
    som_rates = results[1+len(contrast_values):1+2*len(contrast_values)]

    tot = 0.0
    for pi in range(3):
        tot += 2*(spont_rates[pi] - target_spontaneous_rates[pi])**2

        for ci in range(len(contrast_values)):
            tot += (pv_rates[ci][pi][0] - target_exc_contrast[pi])**2
            tot += (som_rates[ci][pi][0] - target_exc_contrast[pi])**2

    return np.sqrt(tot)


def print_log(string):
    f = open('%s/log.txt'%(result_dir), 'a')
    f.write(string)
    f.close()

def print_simlog(sim_parameters,epoch):
    f = open('%s/simpars.txt'%(result_dir), 'a')
    f.write('epoch %s\n'%epoch)
    for par in fitting_parameters:
        f.write('%s    %s\n'%(par, sim_parameters[par]))
    f.write('\n')
    f.close()



def gradient_descent(lr):

    simnum = 0
    # Initiate Parameters
    #gext_baseline, g_exc, g_inh, par_ext_rate0, par_ext_rate1 =  18., 1.5e-3, 20.*1.5e-3, 200., 400.

    sim_parameters = read_sim_params('init_pars.txt')
    #sim_parameters = {}
    #sim_parameters['gext_baseline'] = 12.
    #sim_parameters['g_exc'] = 1.5e-3
    #sim_parameters['g_inh'] = 20.*1.5e-3
    #sim_parameters['par_gext_rate0'] = 200.
    #sim_parameters['par_gext_rate1'] = 200.
    #sim_parameters['par_ext_syn_1'] = 8.
    #sim_parameters['par_ext_syn_2'] = 8.
    #sim_parameters['par_ext_syn_3'] = 8.
    #sim_parameters['par_ext_syn_4'] = 4.
    #sim_parameters['chr2_str_som']  = 0.05
    #sim_parameters['chr2_str_pv']  = 0.1

    epoch = 0

    newsimnum, paramlist = get_paramlist(sim_parameters, simnum, epoch)
    results = run_batch(result_dir, simnum, paramlist)
    simnum = newsimnum
    current_rms = calculate_rms(results)
    plot_results(results,result_dir, epoch)

    print_log("EPOCH 0. RMS = %s\n"%(current_rms))
    print_simlog(sim_parameters,epoch)

    while (epoch < 100): 
        epoch +=1
        simnum, grad = compute_gradient(sim_parameters, current_rms, simnum, epoch) # array
        new_sim_parameters = update_parameters(sim_parameters, grad, lr)
        print('Gradient compute complete,', simnum)

        newsimnum, paramlist = get_paramlist(new_sim_parameters, simnum, epoch)
        results = run_batch(result_dir, simnum, paramlist)
        simnum = newsimnum

        current_rms = calculate_rms(results)
        plot_results(results,result_dir, epoch)
        print('Epoch %s Gradient step complete,'%epoch, simnum, 'RMS = ',current_rms)

        print_log('EPOCH %s. RMS = %s\n'%(epoch, current_rms))
        sim_parameters = new_sim_parameters
        print_simlog(sim_parameters, epoch)

def update_parameters(sim_parameters, grad, lr):
    new_parameters = sim_parameters.copy()
    for pi, parname in enumerate(fitting_parameters):
        incr = parameter_rules[parname][2]
        upper_bound = parameter_rules[parname][1]
        lower_bound = parameter_rules[parname][0]

        delta = - lr * incr * grad[pi]
        delta = min(delta, incr*3)
        delta = max(delta, -incr*3)

        new_pval = new_parameters[parname] + delta
        new_pval = min(upper_bound, new_pval)
        new_pval = max(lower_bound, new_pval)

        new_parameters[parname] = new_pval
    return new_parameters

       
global simnum
simnum = 0
def compute_gradient(sim_parameters, current_rms, simnum, epoch):
    grad_paramlist =[]
    newsimnum = simnum
    for pi, parname in enumerate(fitting_parameters):
        incr = parameter_rules[parname][2]
        delta_param = sim_parameters.copy()
        delta_param[parname] += incr

        newsimnum, paramlist = get_paramlist(delta_param, newsimnum, epoch)

        grad_paramlist += paramlist

    results = run_batch(result_dir, simnum, grad_paramlist, print_results = False) 
    assert len(results) == len(fitting_parameters)*datapoints

    grad = np.zeros(len(fitting_parameters))
    for pi in range(len(fitting_parameters)):
        new_rms = calculate_rms(results[datapoints*pi:datapoints*(pi+1)])
        grad[pi] =  (new_rms - current_rms)

    return newsimnum, grad

def greedy_algorithm():
    simnum = 0
    # Initiate Parameters
    #gext_baseline, g_exc, g_inh, par_ext_rate0, par_ext_rate1 =  18., 1.5e-3, 20.*1.5e-3, 200., 400.

    sim_parameters = {}
    sim_parameters['gext_baseline'] = 12.
    sim_parameters['g_exc'] = 1.5e-3
    sim_parameters['g_inh'] = 20.*1.5e-3
    sim_parameters['par_gext_rate0'] = 200.
    sim_parameters['par_gext_rate1'] = 200.
    sim_parameters['par_ext_syn_1'] = 8.
    sim_parameters['par_ext_syn_2'] = 8.
    sim_parameters['par_ext_syn_3'] = 8.
    sim_parameters['par_ext_syn_4'] = 4.
    sim_parameters['chr2_str_som']  = 0.05
    sim_parameters['chr2_str_pv']  = 0.1
    
    trialnum = 0
    simnum, paramlist = get_paramlist(sim_parameters, simnum, trialnum)

    npars = len(paramlist)
    pool = mp.Pool(npars) # Number of threads
    
    results = pool.map(pexec, paramlist)
    pool.close()
    current_rms = calculate_rms(results)
    plot_results(results,result_dir, trialnum)

    print_log("INIT RMS: %s\n"%(current_rms))
    #  plot_results(results, trialnum)

    while (trialnum < 100): 
        par_choice, par_sets  = random_update_parameters(sim_parameters)

        for pset in par_sets:

            trialnum +=1
            simnum, paramlist = get_paramlist(pset, simnum, trialnum)
            print_log("TRIAL %s (%s): testing %s from %s to %s\n"%(trialnum, simnum, par_choice,sim_parameters[par_choice], pset[par_choice]))

            npars = len(paramlist)
            pool = mp.Pool(npars) # Number of threads
            
            results = pool.map(pexec, paramlist)
            pool.close()
            new_rms = calculate_rms(results)
            plot_results(results,result_dir, trialnum)

            if new_rms < current_rms:
                #accept
                print_log('%s accepted. RMS = %s\n'%(trialnum, new_rms))
                current_rms = new_rms
                sim_parameters = pset
            else:
                print_log('%s rejected. RMS = %s\n'%(trialnum,new_rms))
            print_log('%s\n'%time())

#def plot_results(results, trialnum):
    ### plot the medians and contrast curves to result_dir against target values 
    #plt.savefig(f'{result_dir}/result_figs/{trialnum}.png')

fitting_parameters = ['gext_baseline', 'g_tha_e', 'g_tha_p',#'g_exc', 'g_inh',
        'par_gext_rate0', 'par_gext_rate1',
        'par_ext_syn_1', 'par_ext_syn_2', 'par_ext_syn_3', 'par_ext_syn_4',
        'g_syn_ee', 'g_syn_ep', 'g_syn_es',
        'g_syn_pe', 'g_syn_pe_far', 'g_syn_pp', 'g_syn_pp_far',
        'g_syn_se', 'g_syn_sp', 'g_syn_ss', 'g_syn_se_far', 'g_syn_sp_far']
        #'chr2_str_som', 'chr2_str_pv']

n_params = len(fitting_parameters)

parameter_rules = {} # List of lower boundary, upper boundary, and increments
parameter_rules['gext_baseline'] = [ 5., 30., 1.]
parameter_rules['g_tha_e'] = [ 0.2e-3, 5.0e-3, 0.1e-3]
parameter_rules['g_tha_p'] = [ 0.2e-3, 5.0e-3, 0.1e-3]
#parameter_rules['g_exc'] = [ 0.2e-3, 5.0e-3, 0.1e-3]
#parameter_rules['g_inh'] = [ 5.0e-3, 60.0e-3, 1e-3]
parameter_rules['par_gext_rate0'] = [10., 400., 10.]
parameter_rules['par_gext_rate1'] = [100., 1000., 10.]
parameter_rules['par_ext_syn_1'] = [1., 20., .5]
parameter_rules['par_ext_syn_2'] = [1., 20., .5]
parameter_rules['par_ext_syn_3'] = [1., 20., .5]
parameter_rules['par_ext_syn_4'] = [1., 20., .5]
parameter_rules['chr2_str_som']  = [0.001, 0.1, 0.01]
parameter_rules['chr2_str_pv']  = [0.001, 0.1, 0.01]
parameter_rules['g_syn_ee'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_ep'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_es'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_pe'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_pe_far'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_pp'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_pp_far'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_se'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_sp'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_ss'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_se_far'] = [0.0001, 0.1, 0.0002]
parameter_rules['g_syn_sp_far'] = [0.0001, 0.1, 0.0002]


import random
def random_update_parameters(sim_parameters):
    par_choice = random.choice(fitting_parameters)
    low_lim, hi_lim, incr = parameter_rules[par_choice]
    current_value = sim_parameters[par_choice]

    sets = []

    down_value = current_value - incr
    if down_value >= low_lim:
        down_parameters = sim_parameters.copy() 
        down_parameters[par_choice] = down_value
        sets.append(down_parameters)

    up_value = current_value + incr
    if up_value <= hi_lim:
        up_parameters = sim_parameters.copy() 
        up_parameters[par_choice] = up_value
        sets.append(up_parameters)

    return par_choice, sets

def single_run():

    simnum = 0
    # Initiate Parameters
    #gext_baseline, g_exc, g_inh, par_ext_rate0, par_ext_rate1 =  18., 1.5e-3, 20.*1.5e-3, 200., 400.

    sim_parameters = {}
    sim_parameters['gext_baseline'] = 12.
    sim_parameters['g_exc'] = 1.5e-3
    sim_parameters['g_inh'] = 20.*1.5e-3
    sim_parameters['par_gext_rate0'] = 200.
    sim_parameters['par_gext_rate1'] = 200.
    sim_parameters['par_ext_syn_1'] = 8.
    sim_parameters['par_ext_syn_2'] = 8.
    sim_parameters['par_ext_syn_3'] = 8.
    sim_parameters['par_ext_syn_4'] = 4.
    sim_parameters['chr2_str_som']  = 0.05
    sim_parameters['chr2_str_pv']  = 0.1

    epoch = 0
    new_simnum, paramlist = get_paramlist(sim_parameters, simnum, epoch)

    results = run_batch(result_dir, simnum, paramlist)
    simnum = new_simnum
    print(results)

if __name__ == '__main__':

    t1 = time()

    #mp.freeze_support()
    
    if not exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(result_dir+'/result_figs')
        os.mkdir(result_dir+'/qfiles')
        os.mkdir(result_dir+'/reports')

    #single_run()
    gradient_descent(0.01)

    #paramlist = paramlist[:0]
#    simnum = 0
#
#    # Initiate Parameters
#    #gext_baseline, g_exc, g_inh, par_ext_rate0, par_ext_rate1 =  18., 1.5e-3, 20.*1.5e-3, 200., 400.
#    sim_parameters = {}
#    sim_parameters['gext_baseline'] = 18.
#    sim_parameters['g_exc'] = 1.5e-3
#    sim_parameters['g_inh'] = 20.*1.5e-3
#    sim_parameters['par_gext_rate0'] = 200.
#    sim_parameters['par_gext_rate1'] = 400.
#    sim_parameters['par_ext_syn_1'] = 10.
#    sim_parameters['par_ext_syn_2'] = 3.
#    sim_parameters['par_ext_syn_3'] = 2.
#    sim_parameters['par_ext_syn_4'] = 8.
#    sim_parameters['chr2_str_som']  = 0.1
#    sim_parameters['chr2_str_pv']  = 0.05
#
#    #pool = mp.Pool(35, initializer=parallelinit, initargs=(num_P_glo, num_r_glo)) # Number of threads
#    if not exists('results'):
#        os.mkdir('results')
#
#    simnum, paramlist = get_paramlist(sim_parameters, simnum)
#    print(paramlist)
#
#    npars = len(paramlist)
#    pool = mp.Pool(npars) # Number of threads
#    
#    results = pool.map(pexec, paramlist)
#    pool.close()
#    loss = calculate_rms(results)
#    print(loss)
#
#    t2 = time()
#
#    print(t2-t1)
#
