import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import numpy as np
import os
from help_funcs import *
import sys


def run_simulation(sim_parameters, condition, rngseed = ''):
    print("SIMULATING:", condition)

    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 3})
    if rngseed != '':
        nest.rng_seed = rngseed
        rng = np.random.default_rng(rngseed)

    def reset_sim():
        nest.SetKernelStatus({'biological_time': 0.})
        for popu in [exc_neurons, pv_neurons, som_neurons]:
            dVms =  {"V_m": [Vr+(Vt-Vr)*rng.random() for x in range(len(popu))]}
            popu.set(dVms)
    
    ## Simulation Parameters
    gext_baseline = sim_parameters['gext_baseline']
    #g_exc         = sim_parameters['g_exc']
    #g_inh         = sim_parameters['g_inh']
    g_tha_e        = sim_parameters['g_tha_e'] * 1000.0
    g_tha_p        = sim_parameters['g_tha_p'] * 1000.0
    par_gext_rate0 = sim_parameters['par_gext_rate0']
    par_gext_rate1 = sim_parameters['par_gext_rate1']
    
    par_ext_syn_1 = sim_parameters['par_ext_syn_1'] * 1000.0
    par_ext_syn_2 = sim_parameters['par_ext_syn_2'] * 1000.0
    par_ext_syn_3 = sim_parameters['par_ext_syn_3'] * 1000.0
    par_ext_syn_4 = sim_parameters['par_ext_syn_4'] * 1000.0
    chr2_str_som  = sim_parameters['chr2_str_som'] * 1000.0
    chr2_str_pv   = sim_parameters['chr2_str_pv'] * 1000.0

    stim_range = sim_parameters['stim_range']
    
    g_syn_ee      = sim_parameters['g_syn_ee'] * 1000.0
    g_syn_ep      = sim_parameters['g_syn_ep'] * 1000.0
    g_syn_es      = sim_parameters['g_syn_es'] * 1000.0
    g_syn_pe      = sim_parameters['g_syn_pe'] * 1000.0
    g_syn_pp      = sim_parameters['g_syn_pp'] * 1000.0
    g_syn_se      = sim_parameters['g_syn_se'] * 1000.0
    g_syn_sp      = sim_parameters['g_syn_sp'] * 1000.0
    
    epsilon       = sim_parameters['epsilon']  # 0.02
    p_far_e       = sim_parameters['p_far_e']
    p_far_p       = sim_parameters['p_far_p']
    p_far_s       = sim_parameters['p_far_s']
    p_chr2_random = sim_parameters['p_chr2_random']
    
    ### Cell parameters ###
    tau_m     =  25.  # ms Membrane time constant
    c_m       =  0.25*1e3 # pF Capacitance
    tau_exc   =  6.   # ms Synaptic time constant (excitatory)
    tau_inh   =  20.  # ms Synaptic time constant (inhibitory)
    tau_ref   =  5.   # ms Refractory period
    El        = -70.  # mV Leak potential
    Vt        = -40.  # mV Spike Threhold
    Vr        = -65.  # mV Spike Reset
    e_rev_E   =   0.  # mV Reversal Potential for excitatory synapses
    e_rev_I   = -75.  # mV Reversal Potential for inhibitory synapses
    
    neuron_params = {
        'tau_syn_ex'  : tau_exc,  'tau_syn_in'  : tau_inh,
        'E_L'     : El,       'V_reset'    : Vr,       'V_th'   : Vt,
        'C_m'         : c_m,      't_ref' : tau_ref,  'E_ex'    : e_rev_E,
        'E_in'    : e_rev_I}
    
    
    ### Network Parameters ###
    n_cells       = 10000               # Total number of cells in the recurrent network
    n_thalamus    = 1000                # Total number of cells in the input layer
    n_exc         = 8000
    n_inh         = n_cells - n_exc
    n_som         = int(n_inh/2)
    n_pv         = int(n_inh/2)
    size          = float(1.)           # Size of the network
    simtime       = float(500)          # ms Simulation time for each trial
    se_lat        = 0.1                 # Spread of the lateral excitatory connections
    si_lat        = 0.1                 # Spread of the lateral inhibitory connections
    st_lat        = 0.2                 # Spread of the thalamic excitatory connections
    s_som         = 1.                  # Spread of somatostatin connections
    velocity      = 0.05                # mm/ms  velocity
    
    ### External Input Parameters ###
    m_xy          = [size/2., size/2.]  # center of the gaussian [x, y]
    s_xy          = [size/5., size/5.]  # sigma of the gaussian [sx, sy]
    m_time        = simtime/20.         # time to peak of the Gaussian
    s_time        = 20.                 # sigma in time of the Gaussian
    nb_repeats    = 6                  # number of repeated stimulation
    time_spacing  = 0.                  # time in between two stimulation
    #s_xyies       = [(np.array(s_xy)*i).tolist() for i in np.ones(nb_repeats)]
    
    ### Simulation parameters ###
    dt            = 0.1                     # ms Time step
    max_distance  = size/np.sqrt(2)      # Since this is a torus
    max_delay     = dt + max_distance/velocity # Needed for the connectors
    min_delay     = 0.1                     # ms
    #rngseed       = 92847459                # Random seed
    #parallel_safe = False                   # To have results independant on number of nodes
    #verbose       = True                    # To display the state of the connectors
    #space         = Space(periodic_boundaries=((0, 1), (0, 1), (0, 1))) # Specify the boundary conditions
    
    
    ### Initialize Kernel Parameters
    nest.resolution = dt
    nest.set(min_delay=min_delay, max_delay=max_delay+dt)
    nest.set(print_time=True)
    
    Ne         = int(epsilon*n_exc)
    Ni         = int(epsilon*n_inh/2)
    Nth          = int(epsilon*n_thalamus)
    
    exc_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                               extent=[1.,1.], edge_wrap=True)
    exc_neurons = nest.Create('iaf_cond_exp', n_exc, params=neuron_params, positions=exc_pos)
    
    pv_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                               extent=[1.,1.], edge_wrap=True)
    pv_neurons = nest.Create('iaf_cond_exp', n_pv, params=neuron_params, positions=pv_pos)
    
    som_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                               extent=[1.,1.], edge_wrap=True)
    som_neurons = nest.Create('iaf_cond_exp', n_som,params=neuron_params,  positions=som_pos)
    
    # noise_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
    #                            num_dimensions=2, edge_wrap=True)
    addnoise = nest.Create('poisson_generator', n_thalamus, {'rate':gext_baseline}) #, positions=noise_pos)
    
    # Connect external Poisson noise
    n2e_conn = {'rule': 'fixed_indegree', 'indegree': Nth}#, 'mask': {'circular': {'radius': s_som}}}
    n2e_syn = {'synapse_model':'static_synapse', 'weight': par_ext_syn_2*1.5e-3, 'delay': 0.1}
    n2e_synapses = nest.Connect(addnoise, exc_neurons, n2e_conn, syn_spec=n2e_syn)
    
    n2p_conn = {'rule': 'fixed_indegree', 'indegree': Nth}#, 'mask': {'circular': {'radius': s_som}}}
    n2p_syn = {'synapse_model':'static_synapse', 'weight': par_ext_syn_1*1.5e-3, 'delay': 0.1}
    n2p_synapses = nest.Connect(addnoise, pv_neurons, n2p_conn, syn_spec=n2p_syn)
    
    n2s_conne = {'rule': 'fixed_indegree', 'indegree': Nth}#, 'mask': {'circular': {'radius': s_som}}}
    n2s_syne = {'synapse_model':'static_synapse', 'weight': par_ext_syn_3*1.5e-3, 'delay': 0.1}
    n2s_e_synapses = nest.Connect(addnoise, som_neurons, n2s_conne, syn_spec=n2s_syne)
    
    n2s_conni = {'rule': 'fixed_indegree', 'indegree': Nth}#, 'mask': {'circular': {'radius': s_som}}}
    n2s_syni = {'synapse_model':'static_synapse', 'weight': -par_ext_syn_4*1.5e-3, 'delay': 0.1}
    n2s_i_synapses = nest.Connect(addnoise, som_neurons, n2s_conni, syn_spec=n2s_syni)
    
    
    ### Connectivity Parameters
    ## Connectivity from Campagnola, et al. 2022
    cprob_ee   = 0.06
    cprob_ep   = 0.42
    cprob_es   = 0.3
    cprob_pe   = 0.35
    cprob_pp   = 0.39
    cprob_ps   = 0  #0.09
    cprob_se   = 0.23
    cprob_sp   = 0.18
    cprob_ss   = 0  #0.05
    
    #p_far_e    = 0.3
    #p_far_p    = 0.3
    #p_far_s    = 0.6
    
    N_ee       = int(epsilon*n_exc*cprob_ee)
    N_ee_far   = int(N_ee*p_far_e)
    N_ee_close = N_ee - N_ee_far
    
    N_ep       = int(epsilon*n_exc*cprob_ep)
    N_ep_far   = int(N_ep*p_far_e)
    N_ep_close = N_ep - N_ep_far
    
    N_es       = int(epsilon*n_exc*cprob_es)
    
    N_pe       = int(epsilon*n_inh/2*cprob_pe)
    N_pe_far   = int(N_pe*p_far_p)
    N_pe_close = N_pe - N_pe_far
    
    N_pp       = int(epsilon*n_inh/2*cprob_pp)
    N_pp_far   = int(N_pp*p_far_p)
    N_pp_close = N_pp - N_pp_far
    
    N_se       = int(epsilon*n_inh/2*cprob_se)
    N_se_far   = int(N_se*p_far_s)
    N_se_close = N_se - N_se_far
    
    N_sp       = int(epsilon*n_inh/2*cprob_sp)
    N_sp_far   = int(N_sp*p_far_s)
    N_sp_close = N_sp - N_sp_far
    
    ratio_in = se_lat*se_lat*np.pi
    ratio_out = 1.0 - ratio_in
    p_in_ee = N_ee_close/(ratio_in * n_exc)
    p_out_ee =  N_ee_far/(ratio_out * n_exc)
    e2e_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_ee,p_out_ee)
    e2e_conn = {'rule': 'pairwise_bernoulli', 'p': e2e_mask}
    e2e_syn = {'synapse_model':'static_synapse', 'weight': g_syn_ee, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(exc_neurons, exc_neurons, e2e_conn, syn_spec = e2e_syn)
    
    p_in_ep = N_ep_close/(ratio_in * n_exc)
    p_out_ep =  N_ep_far/(ratio_out * n_exc)
    e2p_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_ep, p_out_ep)
    e2p_conn = {'rule': 'pairwise_bernoulli', 'p': e2p_mask}
    e2p_syn = {'synapse_model':'static_synapse', 'weight': g_syn_ep, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(exc_neurons, pv_neurons, e2p_conn, syn_spec = e2p_syn)
    
    e2s_conn = {'rule': 'pairwise_bernoulli', 'p': N_es/n_exc}
    e2s_syn = {'synapse_model':'static_synapse', 'weight': g_syn_es, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(exc_neurons, som_neurons, e2s_conn, syn_spec = e2s_syn)
    
    p_in_pe = N_pe_close/(ratio_in * n_pv)
    p_out_pe =  N_pe_far/(ratio_out * n_pv)
    p2e_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_pe,p_out_pe)
    p2e_conn = {'rule': 'pairwise_bernoulli', 'p': p2e_mask}
    p2e_syn = {'synapse_model':'static_synapse', 'weight': -g_syn_pe, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(pv_neurons, exc_neurons, p2e_conn, syn_spec = p2e_syn)
    
    p_in_pp = N_pp_close/(ratio_in * n_pv)
    p_out_pp =  N_pp_far/(ratio_out * n_pv)
    p2p_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_pp,p_out_pp)
    p2p_conn = {'rule': 'pairwise_bernoulli', 'p': p2p_mask}
    p2p_syn = {'synapse_model':'static_synapse', 'weight': -g_syn_pp, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(pv_neurons, pv_neurons, p2p_conn, syn_spec = p2p_syn)
    
    p_in_se = N_se_close/(ratio_in * n_som)
    p_out_se =  N_se_far/(ratio_out * n_som)
    s2e_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_se,p_out_se)
    s2e_conn = {'rule': 'pairwise_bernoulli', 'p': s2e_mask}
    s2e_syn = {'synapse_model':'static_synapse', 'weight': -g_syn_se, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(som_neurons, exc_neurons, s2e_conn, syn_spec = s2e_syn)
    
    p_in_sp = N_sp_close/(ratio_in * n_som)
    p_out_sp =  N_sp_far/(ratio_out * n_som)
    s2p_mask = nest.logic.conditional(nest.spatial.distance <= se_lat, p_in_sp,p_out_sp)
    s2p_conn = {'rule': 'pairwise_bernoulli', 'p': s2p_mask}
    s2p_syn = {'synapse_model':'static_synapse', 'weight': -g_syn_sp, 'delay': 0.1 + nest.spatial.distance/velocity}
    nest.Connect(som_neurons, pv_neurons, s2p_conn, syn_spec = s2p_syn)
    
    #### Stimulation 
    stim_type, contrast = condition
    if stim_type == 'Spont':
        print("Spontaneous Sim")
        sim_name = 'Spont'
        sim_spontaneous = True
    elif stim_type in ['PV', 'SOM']:
        print(stim_type, 'stimulation at contrast', contrast)
        sim_name = f'{stim_type}_{contrast}'
        sim_spontaneous = False
    else:
        raise Exception()
    
    ### Reset time and variables
    reset_sim()
    
    ### Visual and Chr2 stimulation ###
    if not sim_spontaneous:
        gext_rate     = par_gext_rate0 + par_gext_rate1*np.tanh(contrast/0.3)*0.3  #contrast varies between 0.02 and 1
        Chr2_times = np.array([500.+1000.*i for i in range(nb_repeats)]) # Induce an instantaneous synaptic conductance in the target population at time 0 every other trial
        Chr2_proba = 1.    # percentage of cells that will receive the conductance change
    
        th_pos = nest.spatial.free(pos=nest.random.uniform(min=0., max=1.),
                                   extent=[1.,1.], edge_wrap=True)
        thalamus = nest.Create('inhomogeneous_poisson_generator', n_thalamus, positions=th_pos)
        
        times     = np.arange(0, 200, 1.)
        times [0] = 0.1
        for ni in range(n_thalamus):
            x, y = nest.GetPosition(thalamus[ni])
            rate_times = []
            rate_values = []
    
            for repeat in range(nb_repeats):
                padding = repeat * (simtime+time_spacing)
                rates = gext_rate * np.exp(-((times-m_time)**2/(2*s_time**2)+(x-m_xy[0])**2/(2*s_xy[0]**2)+(y-m_xy[1])**2/(2*s_xy[1]**2)))
    
                rate_times += list(times+ padding)
                rate_values += list(rates)
    
            #print('GAUSSIAN', ni)
            #print(rate_times)
            #print(rate_values)
    
            nest.SetStatus(thalamus[ni], {'rate_times':rate_times, 'rate_values':rate_values})
    
        t2e_conn = {'rule': 'fixed_indegree', 'indegree': Nth, 'mask': {'circular': {'radius': st_lat}}}
        t2e_syn = {'synapse_model':'static_synapse', 'weight': g_tha_e, 'delay': 0.1}
        nest.Connect(thalamus, exc_neurons,t2e_conn, syn_spec=t2e_syn)
    
        t2p_conn = {'rule': 'fixed_indegree', 'indegree': Nth, 'mask': {'circular': {'radius': st_lat}}}
        t2p_syn = {'synapse_model':'static_synapse', 'weight': g_tha_p, 'delay': 0.1}
        nest.Connect(thalamus, pv_neurons,t2p_conn, syn_spec=t2p_syn)

        #print('THALAMUS EXC CONNECTIONS:', len(nest.GetConnections(thalamus, exc_neurons)))
        #print('THALAMUS PV CONNECTIONS:', len(nest.GetConnections(thalamus, pv_neurons)))
    
        source_chr2 = nest.Create('spike_generator', params={'spike_times': Chr2_times})
        if stim_type == 'PV':
            targets = pv_neurons
            chr2_str = chr2_str_pv
        elif stim_type == 'SOM':
            targets = som_neurons
            chr2_str = chr2_str_som
        else:
            raise Exception()
    
        chr_syn = {'synapse_model':'static_synapse', 'weight': chr2_str}
        for ni in range(n_pv):
            dist = nest.Distance(np.array([0.,0.]), targets[ni])
            if dist[0] < stim_range:
                #print(dist[0], 'CONNECT!', nest.GetPosition(targets[ni]))
                nest.Connect(source_chr2, targets[ni], syn_spec = chr_syn)
            else:
                if stim_type == 'PV': continue
                roll = rng.random()
                if p_chr2_random > roll:
                    nest.Connect(source_chr2, targets[ni], syn_spec = chr_syn)
    
        #print('STIM CONNECTIONS:', len(nest.GetConnections(source_chr2, targets)))
        #print(nest.GetStatus(source_chr2, 'spike_times'))
    
    ### Recording
    exc_sr = nest.Create('spike_recorder',)
    pv_sr = nest.Create('spike_recorder')
    som_sr = nest.Create('spike_recorder')
    #noise_sr = nest.Create('spike_recorder')
    
    nest.Connect(exc_neurons,exc_sr)
    nest.Connect(pv_neurons, pv_sr)
    nest.Connect(som_neurons, som_sr)
    #nest.Connect(addnoise, noise_spikes)
    
    ### CHECK Connections ###
    #print('e2e:',len(nest.GetConnections(exc_neurons, exc_neurons)))
    #print('e2p:',len(nest.GetConnections(exc_neurons, pv_neurons)))
    #print('e2s:',len(nest.GetConnections(exc_neurons, som_neurons)))
    #print('p2e:',len(nest.GetConnections(pv_neurons, exc_neurons)))
    #print('p2p:',len(nest.GetConnections(pv_neurons, pv_neurons)))
    #print('s2e:',len(nest.GetConnections(som_neurons, exc_neurons)))
    #print('s2p:',len(nest.GetConnections(som_neurons, pv_neurons)))
    
    simtime = nb_repeats * simtime + (nb_repeats - 1) * time_spacing
    nest.Simulate(simtime)
    
    ### Median FR ###
    exc_indices = np.arange(1,8001)
    pv_indices = np.arange(8001,9001)
    som_indices = np.arange(9001,10001)
    
    
    exc_spikes = []
    exc_rates = []
    for ni in exc_indices:
        spike_ids = np.where(exc_sr.events['senders'] == ni)[0]
        spike_times = exc_sr.events['times'][spike_ids]
        exc_spikes.append(spike_times)
        nspikes = len(spike_times)
        exc_rates.append(nspikes / simtime * 1000.0)
    
    pv_spikes = []
    pv_rates = []
    for ni in pv_indices:
        spike_ids = np.where(pv_sr.events['senders'] == ni)[0]
        spike_times = pv_sr.events['times'][spike_ids]
        pv_spikes.append(spike_times)
        nspikes = len(spike_times)
        pv_rates.append(nspikes / simtime * 1000.0)
    
    som_spikes = []
    som_rates = []
    for ni in som_indices:
        spike_ids = np.where(som_sr.events['senders'] == ni)[0]
        spike_times = som_sr.events['times'][spike_ids]
        som_spikes.append(spike_times)
        nspikes = len(spike_times)
        som_rates.append(nspikes / simtime * 1000.0)
    
    all_spikes = exc_spikes + pv_spikes + som_spikes
    with open('%s/%s_spikes.pickle'%(result_dir,sim_name),'wb') as f:
        pickle.dump(all_spikes,f)
    
    all_positions = np.concatenate([np.array(nest.GetPosition(exc_neurons)), np.array(nest.GetPosition(pv_neurons)), np.array(nest.GetPosition(som_neurons))])
    with open('%s/%s_positions.pickle'%(result_dir,sim_name), 'wb') as f:
        pickle.dump(all_positions, f)
    
    print("EXC:", np.median(exc_rates), np.mean(exc_rates))
    print("PV:", np.median(pv_rates), np.mean(pv_rates))
    print("SOM:", np.median(som_rates), np.mean(som_rates))



sim_parameters = read_sim_params('sim_parameters.txt')
sim_parameters['stim_type'] = 'SOM'
#sim_parameters['contrast'] = 0.5
contrast_values = [0.02, 0.05,0.1,0.18, 0.33]

seed_list = [ 12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 1234 ]
seed_list = []
for sstr in sys.argv[1:]:
    seed_list.append(int(sstr))
print("seeds: ", seed_list)

conditions =  [['Spont',0]] +[  ['PV', c] for c in contrast_values] + [ ['SOM', c] for c in contrast_values] 

print("conditions: ", conditions)

for rngseed in seed_list:
    result_dir = 'results_%s'%rngseed
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for condition in conditions:
        run_simulation(sim_parameters, condition, rngseed = rngseed)

