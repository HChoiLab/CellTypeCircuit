import pickle
import os, socket, pylab
from pyNN.nest import *
from pyNN.utility import get_script_args, Timer
from pyNN.recording import files
from pyNN.space import *
from pyNN.random import NumpyRNG, RandomDistribution
import numpy, pyNN.connectors
from pyNN.parameters import LazyArray

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

### Trial 8sh: tune background firing now that contrast curves look qualitatively right
### halve the background excitatory input weight to the excitatory cells




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


def pexec(args):

    print("ARGS:", args)
    contrast = args[0]
    Chr2  = args[1]   # "SOM" or "PV"
    ChR2_str   = args[2]  # Strength of the ChR2 stimulation [Use 0.08 for SOM or PV for instance]
    p_far = 0.6 # proportion of SOM neurons with long-range connections


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
    epsilon       = 0.02                # Probability of connections
    se_lat        = 0.1                 # Spread of the lateral excitatory connections
    si_lat        = 0.1                 # Spread of the lateral inhibitory connections 
    st_lat        = 0.2                 # Spread of the thalamic excitatory connections
    s_som         = 1.                  # Spread of somatostatin connections
    g_exc         = 1.5e-3              # nS Quantal Excitatory conductance
    g_inh         = 20.*g_exc           # nS Quantal Inhibitory conductance
    g_tha         = g_exc               # nS Quantal Thalamical conductance
    velocity      = 0.05                # mm/ms  velocity
    
    ### External Input Parameters ###
    gext_baseline = 2.*6                  # Baseline noise
    gext_rate     = 2000.*contrast               # Hz max Rate of the Gaussian source
    m_xy          = [size/2., size/2.]  # center of the gaussian [x, y]
    s_xy          = [size/5., size/5.]  # sigma of the gaussian [sx, sy] 
    m_time        = simtime/20.         # time to peak of the Gaussian
    s_time        = 20.                 # sigma in time of the Gaussian
    nb_repeats    = 6                  # number of repeated stimulation
    time_spacing  = 0.                  # time in between two stimulation
    intensities   = gext_rate*numpy.ones(nb_repeats)  
    s_xyies       = [(numpy.array(s_xy)*i).tolist() for i in numpy.ones(nb_repeats)]
    
    ### Chr2 stimulation ### 
    #Chr2       = 'SOM' # Stimulated population with ChR2-like input. This can be either 'PV' or 'SOM'
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
    thalamus = Population(n_thalamus, SpikeSourceArray(), structure=square)                     # Create a population for Gaussian stimulation
    addnoise = Population(n_thalamus, SpikeSourcePoisson(rate=gext_baseline), structure=square) # Create a population for additional external Poisson noise
    
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
    set_gaussian(thalamus) # comment out for no visual stimulus
    
    
    ### We create the ChR2 stimulation
    #chr2_point  = RandomStructure(boundary=Cuboid(0, 0, 0), rng=NumpyRNG(seed=rngseed),origin=(0.0, 0.0, 0.0))
    #chr2_point  = RandomStructure(boundary=Cuboid(0, 0, 0), rng=NumpyRNG(seed=rngseed),origin=(0.5, 0.5, 0.0)) # local stim
    source_chr2 = Population(1, SpikeSourceArray(spike_times=Chr2_times))
    if Chr2 == 'PV':
        targets = pv_cells
    elif Chr2 == 'SOM':
        targets = som_cells
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
    Ne_far     = int(Ne * p_far/10) # /10 new
    Ne_close   = Ne - Ne_far
    Ni         = int(epsilon*n_inh/2)
    N          = int(epsilon*n_thalamus)
    Ni_far     = int(Ni * p_far)
    Ni_close   = Ni - Ni_far
    Ni_farPV   = int(Ni * 0.4) # originally 0.3
    Ni_closePV = Ni - Ni_farPV
    
    # Excitatory Projection to other cells
    exc_syn_strong  = StaticSynapse(weight=g_exc*0.4, delay=delays)
    exc_syn  = StaticSynapse(weight=g_exc*0.2, delay=delays)
    exc_syn_weak  = StaticSynapse(weight=g_exc*0.1, delay=delays)
    exc_conn_ee = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=Ne_close)
    exc_conn_ep = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=Ne_close)
    exc_conn_es = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=4*Ne_close)
    
    exc_conn_ee_far = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=Ne_far)
    exc_conn_ep_far = MyDistanceDependentProbabilityConnector(se_lat, allow_self_connections=False, rng=rng, n_connections=Ne_far)
    
    # PV Projection to other cells
    inh_syn     = StaticSynapse(weight=g_inh*0.2, delay=delays)
    inh_syn_weak     = StaticSynapse(weight=g_inh*0.1, delay=delays)
    inh_syn_strong     = StaticSynapse(weight=g_inh*0.8, delay=delays)
    inh_syn_stronger     = StaticSynapse(weight=g_inh*1.2, delay=delays)
    inh_syn_con     = StaticSynapse(weight=g_inh, delay=delays)
    inh_conn_pe = MyDistanceDependentProbabilityConnector(si_lat, allow_self_connections=False, rng=rng, n_connections=int(1.3*Ni_closePV))
    inh_conn_pp = MyDistanceDependentProbabilityConnector(si_lat, allow_self_connections=False, rng=rng, n_connections=int(1.3*Ni_closePV))
    
    inh_conn_pe_far = MyDistanceDependentProbabilityConnector_far(si_lat, allow_self_connections=False, rng=rng, n_connections=Ni_farPV)
    inh_conn_pp_far = MyDistanceDependentProbabilityConnector_far(si_lat, allow_self_connections=False, rng=rng, n_connections=Ni_farPV)
    
    # SOM Projection to other cells
    inh_conn_se = MyDistanceDependentProbabilityConnector(si_lat,  allow_self_connections=False, rng=rng, n_connections=Ni_close)
    inh_conn_sp = MyDistanceDependentProbabilityConnector(si_lat,  allow_self_connections=False, rng=rng, n_connections=Ni_close)
    inh_conn_ss = MyDistanceDependentProbabilityConnector(si_lat,  allow_self_connections=False, rng=rng, n_connections=int(Ni_close*0.3))
    
    inh_conn_se_far = MyDistanceDependentProbabilityConnector_far(si_lat,  allow_self_connections=False, rng=rng, n_connections=Ni_far)
    inh_conn_sp_far = MyDistanceDependentProbabilityConnector_far(si_lat,  allow_self_connections=False, rng=rng, n_connections=Ni_far)
    #inh_conn_ss_far = MyDistanceDependentProbabilityConnector_far(si_lat,  allow_self_connections=False, rng=rng, n_connections=int(Ni_far*0.2))
    
    # Thalamic Projection to other Cells for the Gaussian Stimulation
    ext_conn_e = MyDistanceDependentProbabilityConnector(st_lat, allow_self_connections=False, rng=rng, n_connections=N)
    ext_syn_e  = StaticSynapse(weight=g_tha*0.9, delay=0.1)
    ext_syn_e_2  = StaticSynapse(weight=(1.0+(contrast-0.1)*2)*g_tha, delay=0.1)
    
    # Additional external Poisson Noise
    ext_conn_1 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_conn_2 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_conn_3 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_conn_4 = MyDistanceDependentProbabilityConnector(s_som, allow_self_connections=False, rng=rng, n_connections=N)
    ext_syn_1  = StaticSynapse(weight=8.0*g_exc, delay=0.1) # originally 10*g_exc (excitatory to PV)
    ext_syn_2  = StaticSynapse(weight=8.0*g_exc, delay=0.1) # originally 3*g_exc (excitatory to PC), just used 0.1,1
    ext_syn_3  = StaticSynapse(weight=8*g_exc, delay=0.1) # (excitatory to SOM), implement contrast dependence here?
    ext_syn_4  = StaticSynapse(weight=4*g_exc, delay=0.1) # originally 4*g_exc (inhibitory to SOM)
    
    # Creating the connections within the recurrent network
    connections={}
    connections['exc2exc'] = Projection(exc_cells, exc_cells, exc_conn_ee, exc_syn, receptor_type='excitatory', space=space)
    connections['exc2pv']  = Projection(exc_cells, pv_cells,  exc_conn_ep, exc_syn, receptor_type='excitatory', space=space)
    connections['exc2som'] = Projection(exc_cells, som_cells, exc_conn_es, exc_syn, receptor_type='excitatory', space=space)
    connections['pv2exc']  = Projection(pv_cells, exc_cells,  inh_conn_pe, inh_syn_weak, receptor_type='inhibitory', space=space)
    connections['pv2exc_far']  = Projection(pv_cells, exc_cells,  inh_conn_pe_far, inh_syn_strong, receptor_type='inhibitory', space=space)
    connections['pv2pv']   = Projection(pv_cells, pv_cells,   inh_conn_pp, inh_syn, receptor_type='inhibitory', space=space)
    connections['pv2pv_far']   = Projection(pv_cells, pv_cells,   inh_conn_pp_far, inh_syn_strong, receptor_type='inhibitory', space=space) # trial 9
    connections['som2exc'] = Projection(som_cells, exc_cells, inh_conn_se, inh_syn_con, receptor_type='inhibitory', space=space)
    connections['som2pv']  = Projection(som_cells, pv_cells,  inh_conn_sp, inh_syn_con, receptor_type='inhibitory', space=space)
    connections['som2som'] = Projection(som_cells, som_cells, inh_conn_ss, inh_syn, receptor_type='inhibitory', space=space)
    
    # Long-range SOM projections (trial 7)
    connections['som2exc_far'] = Projection(som_cells, exc_cells, inh_conn_se_far, inh_syn, receptor_type='inhibitory', space=space)
    connections['som2pv_far']  = Projection(som_cells, pv_cells,  inh_conn_sp_far, inh_syn, receptor_type='inhibitory', space=space)
    #connections['som2som_far'] = Projection(som_cells, som_cells, inh_conn_ss_far, inh_syn, receptor_type='inhibitory', space=space)
    
    # FeedForward Gaussian Input
    connections['ext01']   = Projection(thalamus, exc_cells, ext_conn_e, ext_syn_e, receptor_type='excitatory', space=space)
    connections['ext02']   = Projection(thalamus, pv_cells,  ext_conn_e, ext_syn_e_2, receptor_type='excitatory', space=space)
    
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
    thalamus.record(['spikes'], to_file=True)
    
    filedir = 'v22b_%s_%.3f_%s_%.3f_%s'%(chr2_origin, contrast, Chr2, ChR2_str, p_far)
    
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
    
    
    # Save spikes and cell positions in the network
    all_spikes = all_cells.get_data(variables='spikes')
    all_spike_arrays = []
    for i in range(10000):
        all_spike_arrays.append(all_spikes.children[0].spiketrains[i].as_array())
    
    with open(f'{filedir}/all_spikes.pickle','wb') as file:
        pickle.dump(all_spike_arrays, file)
    
    thalamus_spikes = thalamus.get_data(variables='spikes')
    thalamus_spike_arrays = []
    for i in range(n_thalamus):
        thalamus_spike_arrays.append(thalamus_spikes.children[0].spiketrains[i].as_array())
    
    with open(f'{filedir}/thalamus_spikes.pickle','wb') as file:
        pickle.dump(thalamus_spike_arrays, file)
    
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
    numpy.save(f'{filedir}/thalamus_positions.npy',thalamus.positions)
    
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

def parallelinit(num_P_c, num_r_c):
    global num_P_glo, num_r_glo
    num_P_glo = num_P_c
    num_r_glo = num_r_c

if __name__ == '__main__':

    t1 = time()

    mp.freeze_support()

    num_P = 21 # Number of parameters
    num_r = 11 # Number of data points

    num_P_glo = mp.Value(ctypes.c_int, num_P)
    num_r_glo = mp.Value(ctypes.c_int, num_r)

    Chr2_list = ['SOM', 'PV'] 
    somlist = list(itertools.product( np.arange(0.1,0.3,0.02), ['SOM'], [0.05]))
    pvlist = list(itertools.product( np.arange(0.1,0.3,0.02), ['PV'], [0.1]))
    paramlist = somlist + pvlist
    print(paramlist)
    print(len(paramlist))

    #pool = mp.Pool(35, initializer=parallelinit, initargs=(num_P_glo, num_r_glo)) # Number of threads
    pool = mp.Pool(35) # Number of threads

    results = pool.map(pexec, paramlist)
    pool.close()

    t2 = time()

    print(t2-t1)

