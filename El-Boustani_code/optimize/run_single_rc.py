import pickle
import os, socket, pylab
from pyNN.nest import *
from pyNN.utility import get_script_args, Timer
from pyNN.recording import files
from pyNN.space import *
from pyNN.random import NumpyRNG, RandomDistribution
import numpy, pyNN.connectors
from pyNN.parameters import LazyArray
from os.path import exists

from pylab import *

from lazyarray import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, exp, \
                      fabs, floor, fmod, hypot, ldexp, log, log10, modf, power, \
                      sin, sinh, sqrt, tan, tanh, maximum, minimum
from numpy import e, pi

import sys
from time import time
import multiprocessing as mp
import ctypes
import itertools

from help_funcs import *

### Trial 8sh: tune background firing now that contrast curves look qualitatively right
### halve the background excitatory input weight to the excitatory cells


use_power_law = False
result_dir = sys.argv[1]

### Define a class of connectors that draw random connections within a disk around each neuron with a radius sigma
class MyDistanceDependentProbabilityConnector(pyNN.connectors.MapConnector):    
    parameter_names = ('allow_self_connections', 'd_expression', 'n_connections')
    def __init__(self, sigma, allow_self_connections=False, 
                 rng=None, safe=True, n_connections=None, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.sigma = sigma
        self.n_connections = n_connections
        self.allow_self_connections = allow_self_connections
        self.rng = pyNN.connectors._get_rng(rng)
    def connect(self, projection):
        distance_map    = self._generate_distance_map(projection)
        generator       = RandomDistribution('uniform', (0, 1), rng=self.rng)        
        connection_map  = numpy.zeros(projection.shape, dtype=bool)        
        for i in range(projection.shape[1]):
            pidx = numpy.zeros(0, dtype=numpy.int32)
            while len(pidx) < self.n_connections:
                newidx = numpy.where(distance_map[:,i] < self.sigma)[0]
                pidx   = numpy.concatenate((pidx, newidx.astype(numpy.int32)))
                if projection.pre == projection.post:    
                    pidx = pidx[numpy.where(pidx!=i)[0]]               
            idx = numpy.random.permutation(pidx)[:self.n_connections]
            connection_map[idx, i] = True
        connection_map = LazyArray(connection_map)    
        self._connect_with_map(projection, connection_map, distance_map)

class MyDistanceDependentProbabilityConnector_far(pyNN.connectors.MapConnector):    
    parameter_names = ('allow_self_connections', 'd_expression', 'n_connections')
    def __init__(self, sigma, allow_self_connections=False, 
                 rng=None, safe=True, n_connections=None, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.sigma = sigma
        self.n_connections = n_connections
        self.allow_self_connections = allow_self_connections
        self.rng = pyNN.connectors._get_rng(rng)
    def connect(self, projection):
        distance_map    = self._generate_distance_map(projection)
        generator       = RandomDistribution('uniform', (0, 1), rng=self.rng)        
        connection_map  = numpy.zeros(projection.shape, dtype=bool)        
        for i in range(projection.shape[1]):
            pidx = numpy.zeros(0, dtype=numpy.int32)
            while len(pidx) < self.n_connections:
                newidx = numpy.where(distance_map[:,i] > self.sigma)[0]
                pidx   = numpy.concatenate((pidx, newidx.astype(numpy.int32)))
                if projection.pre == projection.post:    
                    pidx = pidx[numpy.where(pidx!=i)[0]]               
            idx = numpy.random.permutation(pidx)[:self.n_connections]
            connection_map[idx, i] = True
        connection_map = LazyArray(connection_map)    
        self._connect_with_map(projection, connection_map, distance_map)

def powerlaw(d,a):
    return numpy.power(d,-a)

def uni_powerlaw(d,a,c):
    if d < c: return 1.0
    else: return numpy.power(c,a)*numpy.power(d,-a)

def get_norm(positions,a,c):
    origin = numpy.array([0.5,0.5])
    distances=[]
    probs = []
    for ni in range(10000):
        pos = positions[:,ni][:2]
        dist = numpy.linalg.norm(origin-pos)
        distances.append(dist)
        probs.append(uni_powerlaw(dist,a,c))

    return numpy.sum(probs)

def pexec(sim_parameters):
    
    print_results = False

    ## Parameters for fitting
    gext_baseline = sim_parameters['gext_baseline']
    #g_exc         = sim_parameters['g_exc']  
    #g_inh         = sim_parameters['g_inh']
    g_tha_e        = sim_parameters['g_tha_e']
    g_tha_p        = sim_parameters['g_tha_p']
    par_gext_rate0 = sim_parameters['par_gext_rate0'] 
    par_gext_rate1 = sim_parameters['par_gext_rate1']
    
    par_ext_syn_1 = sim_parameters['par_ext_syn_1']
    par_ext_syn_2 = sim_parameters['par_ext_syn_2']
    par_ext_syn_3 = sim_parameters['par_ext_syn_3']
    par_ext_syn_4 = sim_parameters['par_ext_syn_4']
    chr2_str_som  = sim_parameters['chr2_str_som'] 
    chr2_str_pv   = sim_parameters['chr2_str_pv']  
    
    g_syn_ee      = sim_parameters['g_syn_ee']
    g_syn_ep      = sim_parameters['g_syn_ep']
    g_syn_es      = sim_parameters['g_syn_es']
    g_syn_pe      = sim_parameters['g_syn_pe']
    g_syn_pe_far  = sim_parameters['g_syn_pe_far']
    g_syn_pp      = sim_parameters['g_syn_pp']
    g_syn_pp_far  = sim_parameters['g_syn_pp_far']
    g_syn_se      = sim_parameters['g_syn_se']
    g_syn_sp      = sim_parameters['g_syn_sp']
    g_syn_ss      = sim_parameters['g_syn_ss']
    g_syn_se_far  = sim_parameters['g_syn_se_far']
    g_syn_sp_far  = sim_parameters['g_syn_sp_far']
    
    epsilon       = sim_parameters['epsilon']  # 0.02
    p_far_e       = sim_parameters['p_far_e']
    p_far_p       = sim_parameters['p_far_p']
    p_far_s       = sim_parameters['p_far_s']
    
    a_e = 3.0
    a_p = 3.0
    a_s = 1.0

    timer = Timer()
    ### Cell parameters ###
    tau_m     =  25.  # ms Membrane time constant
    c_m       =  0.25 # nF Capacitance
    tau_exc   =  6.   # ms Synaptic time constant (excitatory)
    tau_inh   =  20.  # ms Synaptic time constant (inhibitory)
    tau_ref   =  5.   # ms Refractory period
    El        = -70.  # mV Leak potential
    Vt        = -40.  # mV Spike Threhold 
    Vr        = -65.  # mV Spike Reset
    e_rev_E   =   0.  # mV Reversal Potential for excitatory synapses
    e_rev_I   = -75.  # mV Reversal Potential for inhibitory synapses
     
    default_params = {
        'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
        'v_rest'     : El,       'v_reset'    : Vr,       'v_thresh'   : Vt,
        'cm'         : c_m,      'tau_refrac' : tau_ref,  'e_rev_E'    : e_rev_E,
        'e_rev_I'    : e_rev_I}   
    
    ### Network Parameters ###         
    n_cells       = 10000               # Total number of cells in the recurrent network
    n_thalamus    = 1000                # Total number of cells in the input layer
    n_exc         = int(0.8 * n_cells)  # 4:1 ratio for exc/inh
    n_inh         = n_cells - n_exc         
    size          = float(1.)           # Size of the network
    simtime       = float(500)          # ms Simulation time for each trial
    #epsilon       = 0.02                # Probability of connections --> Probability of connections from x to y is now epsilon*cprob_xy
    se_lat        = 0.1                 # Spread of the lateral excitatory connections
    si_lat        = 0.1                 # Spread of the lateral inhibitory connections 
    st_lat        = 0.2                 # Spread of the thalamic excitatory connections
    s_som         = 1.                  # Spread of somatostatin connections
    #g_exc         = 1.5e-3              # nS Quantal Excitatory conductance
    #g_inh         = 20.*g_exc           # nS Quantal Inhibitory conductance
    #g_tha         = g_exc               # nS Quantal Thalamical conductance
    velocity      = 0.05                # mm/ms  velocity
    
    ### External Input Parameters ###
    #gext_baseline = 2.*6                  # Baseline noise
    #gext_rate     = par_ext_rate*2000.*contrast               # Hz max Rate of the Gaussian source
    #gext_rate     = 200.+400.*contrast               # Hz max Rate of the Gaussian source
    if not sim_spontaneous:
        gext_rate     = par_gext_rate0 + par_gext_rate1*contrast  #contrast varies between 0.02 and 1
    m_xy          = [size/2., size/2.]  # center of the gaussian [x, y]
    s_xy          = [size/5., size/5.]  # sigma of the gaussian [sx, sy] 
    m_time        = simtime/20.         # time to peak of the Gaussian
    s_time        = 20.                 # sigma in time of the Gaussian
    nb_repeats    = 6                  # number of repeated stimulation
    if not sim_spontaneous:
        intensities   = gext_rate*numpy.ones(nb_repeats)  
    time_spacing  = 0.                  # time in between two stimulation
    s_xyies       = [(numpy.array(s_xy)*i).tolist() for i in numpy.ones(nb_repeats)]
    
    ### Chr2 stimulation ### 
    #Chr2       = 'SOM' # Stimulated population with ChR2-like input. This can be either 'PV' or 'SOM'
    if not sim_spontaneous:
        Chr2_times = numpy.array([range(0,10,10)+epo for epo in numpy.arange(500,1000000,1000)]).flatten() # Induce an instantaneous synaptic conductance in the target population at time 0 every other trial
    #print('Chr2_times:', Chr2_times)
        Chr2_proba = 1.    # percentage of cells that will receive the conductance change
    
    #ChR2_str   = 0.08  # Strength of the ChR2 stimulation [Use 0.08 for SOM or PV for instance]
    #ChR2_str   = 0.03  # PV high
    #ChR2_str   = 0.04  # SOM low
    #ChR2_str   = 0.05  # SOM high
    #ChR2_str   = 0.02  # PV low
    #ChR2_str   = 0.05  # SOM high
    #ChR2_str   = 0.0  # SOM high
    
    ### Simulation parameters ###
    dt            = 0.1                     # ms Time step
    max_distance  = size/numpy.sqrt(2)      # Since this is a torus
    max_delay     = dt + max_distance/velocity # Needed for the connectors
    min_delay     = 0.1                     # ms
    rngseed       = 92847459                # Random seed
    parallel_safe = False                   # To have results independant on number of nodes
    verbose       = True                    # To display the state of the connectors
    space         = Space(periodic_boundaries=((0, 1), (0, 1), (0, 1))) # Specify the boundary conditions
    
    ### Simulation initialization
    node_id = setup(timestep=dt, min_delay=min_delay, max_delay=max_delay+dt, spike_precision='on_grid')
    np = num_processes()
    if node_id == 0:
        nest.SetStatus([0], {'print_time' : True})
    def nprint(string): 
        if node_id == 0:
            print(string)
    
    
    
    ### We create the cells and generate random positons in [0, size]x[0, size]
    square                 = RandomStructure(boundary=Cuboid(1, 1, 0), rng=NumpyRNG(seed=rngseed),origin=(0.5, 0.5, 0.5))
    all_cells              = Population(n_exc+n_inh, IF_cond_exp, default_params, square, label="All Cells")
    numpy.random.seed(12)
    exc_cells              = all_cells[0:n_exc]
    inh_cells              = all_cells[n_exc:n_cells]
    pv_cells               = inh_cells[:int(n_inh/2)]
    som_cells              = inh_cells[int(n_inh/2):]
    
    ### We initialize membrane potential v values to random number
    rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
    uniformDistr = RandomDistribution('uniform', [Vr, Vt], rng=rng)
    all_cells.initialize(v=uniformDistr)
    
    ### We create the external noise and the thalamic input
    numpy.random.seed(rngseed)
    addnoise = Population(n_thalamus, SpikeSourcePoisson(rate=gext_baseline), structure=square) # Create a population for additional external Poisson noise
    
    if not sim_spontaneous:
        def set_gaussian(population, m_time=m_time, s_time=s_time, m_xy=m_xy, s_xy=s_xyies, rate=intensities):
            from NeuroTools.stgen import StGen
            generator = StGen(seed=123327)
            times     = numpy.arange(0, simtime, 1.).astype(int)    
            for cell in population.local_cells:            
                x, y    = cell.position[[0, 1]]   
                spikes  = []             
                rate = intensities
                for repeat in range(nb_repeats):
                    padding = repeat*(simtime+time_spacing)
                    if x>-1:
                        rates   = rate[repeat]*numpy.exp(-((times-m_time)**2/(2*s_time**2)+(x-m_xy[0])**2/(2*s_xy[repeat][0]**2)+(y-m_xy[1])**2/(2*s_xy[repeat][1]**2)))
                    else:
                        rates = times*0
                    # print(rates, times,simtime)
                    generator.inh_poisson_generator(rates, times, t_stop=int(simtime), array=True) 
                    spikes += (numpy.round(generator.inh_poisson_generator(rates, times, t_stop=simtime, array=True) + padding, 1)).tolist() 
                if len(spikes) > 0:
                    data = numpy.sort(numpy.array(spikes).astype(numpy.float32))
                    cell.spike_times = data[data > dt]
        thalamus = Population(n_thalamus, SpikeSourceArray(), structure=square)                     # Create a population for Gaussian stimulation
        set_gaussian(thalamus) # comment out for no visual stimulus
    
        ### We create the ChR2 stimulation
        #chr2_point  = RandomStructure(boundary=Cuboid(0, 0, 0), rng=NumpyRNG(seed=rngseed),origin=(0.0, 0.0, 0.0))
        #chr2_point  = RandomStructure(boundary=Cuboid(0, 0, 0), rng=NumpyRNG(seed=rngseed),origin=(0.5, 0.5, 0.0)) # local stim
        source_chr2 = Population(1, SpikeSourceArray(spike_times=Chr2_times))
        if Chr2 == 'PV':
            targets = pv_cells
            ChR2_str = chr2_str_pv
        elif Chr2 == 'SOM':
            targets = som_cells
            ChR2_str = chr2_str_som
        else:
            raise Exception()
        target_pos = targets.positions
        #print(target_pos)
        distal = []
        #print(len(targets))
        #chr2_origin = 'local'#[0.0, 0.0] #distal
        chr2_origin = 'distal'#[0.5, 0.5] #local
        
        for ti in range(len(targets)):
            if chr2_origin == 'distal' :
                dd1 =  numpy.sqrt((target_pos[0][ti] - 0.0)**2 + (target_pos[1][ti] - 0.0)**2)
                dd2 =  numpy.sqrt((target_pos[0][ti] - 1.0)**2 + (target_pos[1][ti] - 0.0)**2)
                dd3 =  numpy.sqrt((target_pos[0][ti] - 0.0)**2 + (target_pos[1][ti] - 1.0)**2)
                dd4 =  numpy.sqrt((target_pos[0][ti] - 1.0)**2 + (target_pos[1][ti] - 1.0)**2)
                dist = numpy.min([dd1, dd2, dd3, dd4])
            elif chr2_origin == 'local':
                dist = numpy.sqrt((target_pos[0][ti] - 0.5)**2 + (target_pos[1][ti] - 0.5)**2)
            #print(ti, dist)
            if dist < 0.5:
                distal.append(ti)
        #print('DISTAL:',len(distal))
        chr_targets = targets[distal]
        print(len(chr_targets), 'TARGETED BY CHR2')
        
        syn = StaticSynapse(weight=ChR2_str)
        Projection(source_chr2, chr_targets, AllToAllConnector(), syn)
    
    ### Create all necessary connections in the network
    nprint("Connecting populations...")
    delays   = "%g + d/%g" %(dt, velocity)
    Ne         = int(epsilon*n_exc)
    Ni         = int(epsilon*n_inh/2)
    N          = int(epsilon*n_thalamus)
    
    if not use_power_law:
        #p_far = 0.6
        #Ne_far     = int(Ne * p_far/10) # /10 new
        #Ne_close   = Ne - Ne_far
        #Ni_far     = int(Ni * p_far)
        #Ni_close   = Ni - Ni_far
        #Ni_farPV   = int(Ni * 0.4) # originally 0.3
        #Ni_closePV = Ni - Ni_farPV
    
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
    
        #Note: old connectivities
        #N_ee_close  #se_lat, Ne_close
        #N_ep_close  #se_lat, Ne_close
        #N_es        #s_som, 4*Ne_close
        #
        #N_ee_far  #Ne_far
        #N_ep_far  #Ne_far
    
        #N_pe_close  #1.3*Ni_closePV
        #N_pp_close  #1.3*Ni_closePV
        #N_pe_far    #Ni_farPV
        #N_pp_far    #Ni_farPV
    
        #N_se_close  #Ni_close
        #N_sp_close  #Ni_close
        #N_ss_close  #0.3*Ni_close
    
        #N_se_far    #Ni_far
        #N_sp_far    #Ni_far
        #N_ss_far    #0.2*Ni_far
    
    # Excitatory Projection to other cells
    #exc_syn  = StaticSynapse(weight=g_exc, delay=delays)
    
    positions = all_cells.positions
    
    print('Exc Projections')
    if use_power_law:
        norm_ee = get_norm(positions, a_e, 0.2) * Ne
        exc_conn_ee = DistanceDependentProbabilityConnector(f'{norm_ee}*d**(-{a_e}) if d>0.02 else 1',allow_self_connections=False, rng=rng)
        exc_conn_ep = DistanceDependentProbabilityConnector(f'{norm_ee}*d**(-{a_e}) if d>0.02 else 1',allow_self_connections=False, rng=rng)
    
        norm_es = get_norm(positions, a_e, 0.2) * Ne*4
        exc_conn_es = DistanceDependentProbabilityConnector(f'{norm_es}*d**(-{a_e}) if d>0.005 else 1',allow_self_connections=False, rng=rng)
    else:
        exc_conn_ee = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=N_ee_close)
        exc_conn_ep = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=N_ep_close)
        exc_conn_es = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N_es)
        
        exc_conn_ee_far = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=N_ee_far)
        exc_conn_ep_far = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=N_ep_far)
    
    # PV Projection to other cells
    print('PV Projections')
    #inh_syn     = StaticSynapse(weight=g_inh*0.2, delay=delays)
    #inh_syn_weak     = StaticSynapse(weight=g_inh*0.1, delay=delays)
    #inh_syn_strong     = StaticSynapse(weight=g_inh*0.8, delay=delays)
    #inh_syn_stronger     = StaticSynapse(weight=g_inh*1.2, delay=delays)
    #inh_syn_con     = StaticSynapse(weight=g_inh, delay=delays)
    
    syn_ee = StaticSynapse(weight = g_syn_ee, delay = delays)  #exc_syn
    syn_ep = StaticSynapse(weight = g_syn_ep, delay = delays) #exc_syn
    syn_es = StaticSynapse(weight = g_syn_es, delay = delays) #exc_syn
    syn_pe = StaticSynapse(weight = g_syn_pe, delay = delays) #inh_syn_weak
    syn_pe_far = StaticSynapse(weight = g_syn_pe_far, delay = delays) #inh_syn_strong
    syn_pp = StaticSynapse(weight = g_syn_pp, delay = delays) #inh_syn
    syn_pp_far = StaticSynapse(weight = g_syn_pp_far, delay = delays) #inh_syn_strong
    syn_se = StaticSynapse(weight = g_syn_se, delay = delays) #inh_syn_con
    syn_sp = StaticSynapse(weight = g_syn_sp, delay = delays) #inh_syn_con
    syn_ss = StaticSynapse(weight = g_syn_ss, delay = delays) #inh_syn
    syn_se_far = StaticSynapse(weight = g_syn_se_far, delay = delays) #inh_syn
    syn_sp_far = StaticSynapse(weight = g_syn_sp_far, delay = delays) #inh_syn
    
    if use_power_law:
        norm_pe = get_norm(positions,a_p, 0.2)* Ni 
        inh_conn_pe = DistanceDependentProbabilityConnector(f'{norm_pe}*d**(-{a_p}) if d>0.02 else 1',allow_self_connections=False, rng=rng)
        inh_conn_pp = DistanceDependentProbabilityConnector(f'{norm_pe}*d**(-{a_p}) if d>0.02 else 1',allow_self_connections=False, rng=rng)
    else:
        inh_conn_pe = MyDistanceDependentProbabilityConnector(si_lat, allow_self_connections=False, rng=rng, n_connections=N_pe_close)
        inh_conn_pp = MyDistanceDependentProbabilityConnector(si_lat, allow_self_connections=False, rng=rng, n_connections=N_pp_close)
        
        inh_conn_pe_far = MyDistanceDependentProbabilityConnector_far(si_lat, allow_self_connections=False, rng=rng, n_connections=N_pe_far)
        inh_conn_pp_far = MyDistanceDependentProbabilityConnector_far(si_lat, allow_self_connections=False, rng=rng, n_connections=N_pp_far)
    
    # SOM Projection to other cells
    print('SOM Projections')
    if use_power_law:
        norm_se = get_norm(positions,a_s, 0.2)*Ni
        inh_conn_se = DistanceDependentProbabilityConnector(f'{norm_se}*d**(-{a_s}) if d>0.02 else 1',allow_self_connections=False, rng=rng)
        inh_conn_sp = DistanceDependentProbabilityConnector(f'{norm_se}*d**(-{a_s}) if d>0.02 else 1',allow_self_connections=False, rng=rng)
    
    else:
        inh_conn_se = MyDistanceDependentProbabilityConnector(si_lat,  allow_self_connections=False, rng=rng, n_connections=N_se_close)
        inh_conn_sp = MyDistanceDependentProbabilityConnector(si_lat,  allow_self_connections=False, rng=rng, n_connections=N_sp_close)
        
        inh_conn_se_far = MyDistanceDependentProbabilityConnector_far(si_lat,  allow_self_connections=False, rng=rng, n_connections=N_se_far)
        inh_conn_sp_far = MyDistanceDependentProbabilityConnector_far(si_lat,  allow_self_connections=False, rng=rng, n_connections=N_sp_far)
    
    # Thalamic Projection to other Cells for the Gaussian Stimulation
    if not sim_spontaneous:
        ext_conn_e = MyDistanceDependentProbabilityConnector(st_lat, allow_self_connections=False, rng=rng, n_connections=N)
        ext_syn_e  = StaticSynapse(weight=g_tha_e, delay=0.1)
        ext_syn_p  = StaticSynapse(weight=g_tha_p, delay=0.1)
    #ext_syn_e_2  = StaticSynapse(weight=(1.0+(contrast-0.1)*2)*g_tha, delay=0.1)
    
    # Additional external Poisson Noise
    ext_conn_1 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_conn_2 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_conn_3 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_conn_4 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_syn_1  = StaticSynapse(weight=par_ext_syn_1*1.5e-3, delay=0.1) # originally 10*g_exc (excitatory to PV)
    ext_syn_2  = StaticSynapse(weight=par_ext_syn_2*1.5e-3, delay=0.1) # originally 3*g_exc (excitatory to PC), just used 0.1,1
    ext_syn_3  = StaticSynapse(weight=par_ext_syn_3*1.5e-3 , delay=0.1) # (excitatory to SOM), implement contrast dependence here?
    ext_syn_4  = StaticSynapse(weight=par_ext_syn_4*1.5e-3 , delay=0.1) # originally 4*g_exc (inhibitory to SOM)
    
    # Creating the connections within the recurrent network
    print('Create Connections')
    if use_power_law:
        connections={}
        connections['exc2exc'] = Projection(exc_cells, exc_cells, exc_conn_ee, exc_syn, receptor_type='excitatory', space=space)
        connections['exc2pv']  = Projection(exc_cells, pv_cells,  exc_conn_ep, exc_syn, receptor_type='excitatory', space=space)
        connections['exc2som'] = Projection(exc_cells, som_cells, exc_conn_es, exc_syn, receptor_type='excitatory', space=space)
        connections['pv2exc']  = Projection(pv_cells, exc_cells,  inh_conn_pe, inh_syn, receptor_type='inhibitory', space=space)
        connections['pv2pv']   = Projection(pv_cells, pv_cells,   inh_conn_pp, inh_syn, receptor_type='inhibitory', space=space)
        connections['som2exc'] = Projection(som_cells, exc_cells, inh_conn_se, inh_syn, receptor_type='inhibitory', space=space)
        connections['som2pv']  = Projection(som_cells, pv_cells,  inh_conn_sp, inh_syn, receptor_type='inhibitory', space=space)
    
    else:
        connections={}
        connections['exc2exc'] = Projection(exc_cells, exc_cells, exc_conn_ee, syn_ee, receptor_type='excitatory', space=space)
        connections['exc2exc_far'] = Projection(exc_cells, exc_cells, exc_conn_ee_far, syn_ee, receptor_type='excitatory', space=space)
    
        connections['exc2pv']  = Projection(exc_cells, pv_cells,  exc_conn_ep, syn_ep, receptor_type='excitatory', space=space)
        connections['exc2pv_far']  = Projection(exc_cells, pv_cells,  exc_conn_ep_far, syn_ep, receptor_type='excitatory', space=space)
    
        connections['exc2som'] = Projection(exc_cells, som_cells, exc_conn_es, syn_es, receptor_type='excitatory', space=space)
    
        connections['pv2exc']  = Projection(pv_cells, exc_cells,  inh_conn_pe, syn_pe, receptor_type='inhibitory', space=space)
        connections['pv2exc_far']  = Projection(pv_cells, exc_cells,  inh_conn_pe_far, syn_pe_far, receptor_type='inhibitory', space=space)
    
        connections['pv2pv']   = Projection(pv_cells, pv_cells,   inh_conn_pp, syn_pp, receptor_type='inhibitory', space=space)
        connections['pv2pv_far']   = Projection(pv_cells, pv_cells,   inh_conn_pp_far, syn_pp_far, receptor_type='inhibitory', space=space) # trial 9
    
        connections['som2exc'] = Projection(som_cells, exc_cells, inh_conn_se, syn_se, receptor_type='inhibitory', space=space)
        connections['som2exc_far'] = Projection(som_cells, exc_cells, inh_conn_se_far, syn_se_far, receptor_type='inhibitory', space=space)
    
        connections['som2pv']  = Projection(som_cells, pv_cells,  inh_conn_sp, syn_sp, receptor_type='inhibitory', space=space)
        connections['som2pv_far']  = Projection(som_cells, pv_cells,  inh_conn_sp_far, syn_sp_far, receptor_type='inhibitory', space=space)
    
    # FeedForward Gaussian Input
    if not sim_spontaneous:
        connections['ext01']   = Projection(thalamus, exc_cells, ext_conn_e, ext_syn_e, receptor_type='excitatory', space=space)
        connections['ext02']   = Projection(thalamus, pv_cells,  ext_conn_e, ext_syn_p, receptor_type='excitatory', space=space)
    
    # Additional Noise
    connections['ext1']    = Projection(addnoise, pv_cells, ext_conn_1, ext_syn_1, receptor_type='excitatory', space=space)
    connections['ext2']    = Projection(addnoise, exc_cells, ext_conn_2, ext_syn_2, receptor_type='excitatory', space=space)
    connections['ext3']    = Projection(addnoise, som_cells, ext_conn_3, ext_syn_3, receptor_type='excitatory', space=space)
    connections['ext4']    = Projection(addnoise, som_cells, ext_conn_4, ext_syn_4, receptor_type='inhibitory', space=space)
    
    
    # read out time used for building
    buildCPUTime = timer.elapsedTime()
    timer.start()
    nprint("Setting the recorders..." )
    
    # Record spikes for all populations
    exc_cells.record(['spikes'], to_file=True)
    inh_cells.record(['spikes'], to_file=True)
    if not sim_spontaneous:
        thalamus.record(['spikes'], to_file=True)
    
    filedir = f'data'
    
    try:
        os.mkdir(filedir)
    except Exception:
        pass
    
    
    # run, measure computer time
    timer.start() # start timer on construction
    print("Running simulation for %g ms." %simtime, flush=True)
    simtime = nb_repeats * simtime + (nb_repeats - 1) * time_spacing
    run(simtime)
    simCPUTime = timer.elapsedTime()
    
    print('SIM DONE', flush=True)
    
    if use_power_law:
        sim_parameters['use_power_law'] = 'True'
        sim_parameters['a_e'] = a_e
        sim_parameters['a_p'] = a_p
        sim_parameters['a_s'] = a_s
    else:
        sim_parameters['use_power_law'] = 'False'
    print_params(filedir+'/parameters.txt', sim_parameters)
    
    # Save spikes and cell positions in the network
    all_spikes = all_cells.get_data(variables='spikes')
    all_spike_arrays = []
    for i in range(10000):
        all_spike_arrays.append(all_spikes.children[0].spiketrains[i].as_array())
        
    
    if print_results:
        with open(f'{filedir}/all_spikes.pickle','wb') as file:
            pickle.dump(all_spike_arrays, file)
        
        if not sim_spontaneous:
            thalamus_spikes = thalamus.get_data(variables='spikes')
            thalamus_spike_arrays = []
            for i in range(n_thalamus):
                thalamus_spike_arrays.append(thalamus_spikes.children[0].spiketrains[i].as_array())
            
            with open(f'{filedir}/thalamus_spikes.pickle','wb') as file:
                pickle.dump(thalamus_spike_arrays, file)
        
            numpy.save(f'{filedir}/thalamus_positions.npy',thalamus.positions)
    
    #all_cells.write_data("Results/all_cells_spikes.h5", gather=True, variables=('spikes'))
    #exc_cells.write_data("Results/exc_spikes.h5", gather=True, variables=('spikes'))
    #thalamus.write_data("Results/thalamus_spikes.h5", gather=True, variables=('spikes'))
    
    #all_cells.save_positions("Results/all_positions.txt")
    #exc_cells.save_positions("Results/exc_positions.txt")
    #inh_cells.save_positions("Results/inh_positions.txt")
    #thalamus.save_positions("Results/tha_positions.txt" )
    #nprint("Building time: %g s" %buildCPUTime)
    #nprint("Running  time: %g s" %simCPUTime)
    
        numpy.save(f'{filedir}/exc_positions.npy',exc_cells.positions)
        numpy.save(f'{filedir}/som_positions.npy',som_cells.positions)
        numpy.save(f'{filedir}/pv_positions.npy',pv_cells.positions)
    
    # s2e_far = connections['som2exc_far'].get(['weight'], format='list')
    # s2e = connections['som2exc'].get(['weight'], format='list')
    
    # s2p_far = connections['som2pv_far'].get(['weight'], format='list')
    # s2p = connections['som2pv'].get(['weight'], format='list')
    
    # s2s_far = connections['som2som_far'].get(['weight'], format='list')
    #s2s = connections['som2som'].get(['weight'], format='list')
    
    #som_projections = [s2e, s2e_far, s2p, s2p_far, s2s_far]
    #som_projections = [s2e, s2e_far, s2p, s2p_far, s2s, s2s_far]
    
    #with open(f'{filedir}/som_projections.pickle','wb') as file:
    #    pickle.dump(som_projections, file)

    if sim_spontaneous:
        return getSpontRateMedians(all_spike_arrays)
    else:
        PC_rates = getStimRateMeansPC(all_spike_arrays, exc_cells.positions)
        PV_rates = getStimRateMeansPV(all_spike_arrays, pv_cells.positions)
        SOM_rates = getStimRateMeansSOM(all_spike_arrays, som_cells.positions)
        return [PC_rates, PV_rates, SOM_rates]

def print_params(fname, sim_params):
    parnames = sim_params.keys()
    f = open(fname,'w')
    for par in parnames:
        f.write('%s    %s\n'%(par, sim_params[par]))
    f.close()

def parallelinit(num_P_c, num_r_c):
    global num_P_glo, num_r_glo
    num_P_glo = num_P_c
    num_r_glo = num_r_c

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

       
global simnum
simnum = 0

fitting_parameters = ['gext_baseline',# 'g_exc', 'g_inh',
        'par_gext_rate0', 'par_gext_rate1',
        'par_ext_syn_1', 'par_ext_syn_2', 'par_ext_syn_3', 'par_ext_syn_4',
        'g_syn_ee', 'g_syn_ep', 'g_syn_es',
        'g_syn_pe', 'g_syn_pe_far', 'g_syn_pp', 'g_syn_pp_far',
        'g_syn_se', 'g_syn_sp', 'g_syn_ss', 'g_syn_se_far', 'g_syn_sp_far']
        #'chr2_str_som', 'chr2_str_pv']

n_params = len(fitting_parameters)

parameter_rules = {} # List of lower boundary, upper boundary, and increments
parameter_rules['gext_baseline'] = [ 5., 30., 1.]
parameter_rules['g_exc'] = [ 0.2e-3, 5.0e-3, 0.1e-3]
parameter_rules['g_inh'] = [ 5.0e-3, 60.0e-3, 1e-3]
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

def single_run():
    sim_params = read_sim_params('rc_pars.txt')
    simnum, paramlist = get_paramlist(sim_params, simnum, epoch)
    npars = len(paramlist)
    pool = mp.Pool(npars) # Number of threads
    results = pool.map(pexec, paramlist)
    pool.close()
    rms = calculate_rms(results)

    print_log("Single run. RMS = %s\n"%(current_rms))
    plot_results(results,result_dir, epoch)

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


if __name__ == '__main__':

    t1 = time()

    mp.freeze_support()
    
    if not exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(result_dir+'/result_figs')

    single_run()

