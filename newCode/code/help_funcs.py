import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats

contrast_values = [0.02, 0.5, 1.0]
target_spontaneous_rates = [2.2, 4, 3]
target_exc_contrast = [5 + 6*ct for ct in  contrast_values]
target_pv_contrast = [12 + 16*ct for ct in contrast_values]
target_som_contrast = [8 + 15*ct for ct in contrast_values]

datapoints = 1 + 2 * len(contrast_values)

def getSpontRateMedians(all_spikes):
    
    # get the median spontaneous rates
    # PC
    exc_spikes = all_spikes[:8000]
    exc_rates = []
    for ni in range(8000): # loop through excitatory neurons
        temp = exc_spikes[ni][exc_spikes[ni]<2000]
        exc_rates = np.append(exc_rates, len(temp)/2)
    exc_med = st.median(exc_rates)

    # PV
    pv_spikes = all_spikes[8000:9000]
    pv_rates = []
    for ni in range(1000): # loop through pv neurons
        temp = pv_spikes[ni][pv_spikes[ni]<2000]
        pv_rates = np.append(pv_rates, len(temp)/2)
    pv_med = st.median(pv_rates)
    
    # SOM
    som_spikes = all_spikes[9000:]
    som_rates = []
    for ni in range(1000): # loop through som neurons
        temp = som_spikes[ni][som_spikes[ni]<2000]
        som_rates = np.append(som_rates, len(temp)/2)
    som_med = st.median(som_rates)
    
    return [exc_med, pv_med, som_med]


def getStimRateMeansPC(all_spikes, exc_positions):
    # only want the PC spikes here
    exc_spikes = all_spikes[:8000]
    # store the control times and the chr2 stimulation times
    control_times = []
    chr_times = []
    ii = 0
    is_control = True
    while True:
        start_time = ii * 500 + 2000
        stop_time = start_time + 40
        if is_control:
            control_times.append(start_time)
        else:
            chr_times.append(start_time)   
        ii += 1
        if is_control:
            is_control = False
        else:
            is_control = True        
        if stop_time > 5000: break
    # compute the rates of the cells distal to the chr2 stimulus
    control_spikes = []
    chr_spikes = []
    for ni in range(8000):
        pos_x, pos_y, pos_z = exc_positions[:,ni]
        dist = np.sqrt((0.5-pos_x)**2 + (0.5-pos_y)**2)
        if dist > 0.2:
            continue
        nspikes = 0
        for tim in control_times:
            aa = exc_spikes[ni] [exc_spikes[ni] > tim ]
            aa = aa [aa < tim+40]
            nspikes += len(aa)
        control_spikes.append(nspikes)

        nspikes = 0
        for tim in chr_times:
            aa = exc_spikes[ni] [exc_spikes[ni] > tim ]
            aa = aa [aa < tim+40]
            nspikes += len(aa)
        chr_spikes.append(nspikes)
    control_rates = np.divide(control_spikes,(40/1000)) # convert from spike count to rate
    chr_rates = np.divide(chr_spikes,(40/1000))
    ctrl_mean = np.mean(control_rates)
    chr2_mean = np.mean(chr_rates)
    
    ctrl_err = scipy.stats.sem(control_rates)
    chr2_err = scipy.stats.sem(chr_rates)
    
    return [ctrl_mean, chr2_mean, ctrl_err, chr2_err]


def getStimRateMeansPV(all_spikes, pv_positions):
    # only want the PC spikes here
    pv_spikes = all_spikes[8000:9000]
    # store the control times and the chr2 stimulation times
    control_times = []
    chr_times = []
    ii = 0
    is_control = True
    while True:
        start_time = ii * 500 + 2000
        stop_time = start_time + 40
        if is_control:
            control_times.append(start_time)
        else:
            chr_times.append(start_time)   
        ii += 1
        if is_control:
            is_control = False
        else:
            is_control = True        
        if stop_time > 5000: break
    # compute the rates of the cells distal to the chr2 stimulus
    control_spikes = []
    chr_spikes = []
    for ni in range(1000):
        pos_x, pos_y, pos_z = pv_positions[:,ni]
        dist = np.sqrt((0.5-pos_x)**2 + (0.5-pos_y)**2)
        if dist > 0.2:
            continue
        nspikes = 0
        for tim in control_times:
            aa = pv_spikes[ni] [pv_spikes[ni] > tim ]
            aa = aa [aa < tim+40]
            nspikes += len(aa)
        control_spikes.append(nspikes)

        nspikes = 0
        for tim in chr_times:
            aa = pv_spikes[ni] [pv_spikes[ni] > tim ]
            aa = aa [aa < tim+40]
            nspikes += len(aa)
        chr_spikes.append(nspikes)
    control_rates = np.divide(control_spikes,(40/1000)) # convert from spike count to rate
    chr_rates = np.divide(chr_spikes,(40/1000))
    ctrl_mean = np.mean(control_rates)
    chr2_mean = np.mean(chr_rates)
        
    ctrl_err = scipy.stats.sem(control_rates)
    chr2_err = scipy.stats.sem(chr_rates)
    
    return [ctrl_mean, chr2_mean, ctrl_err, chr2_err]


def getStimRateMeansSOM(all_spikes, som_positions):
    # only want the SOM spikes here
    som_spikes = all_spikes[9000:]
    # store the control times and the chr2 stimulation times
    control_times = []
    chr_times = []
    ii = 0
    is_control = True
    while True:
        start_time = ii * 500 + 2000
        stop_time = start_time + 40
        if is_control:
            control_times.append(start_time)
        else:
            chr_times.append(start_time)   
        ii += 1
        if is_control:
            is_control = False
        else:
            is_control = True        
        if stop_time > 5000: break
    # compute the rates of the cells distal to the chr2 stimulus
    control_spikes = []
    chr_spikes = []
    for ni in range(1000):
        pos_x, pos_y, pos_z = som_positions[:,ni]
        dist = np.sqrt((0.5-pos_x)**2 + (0.5-pos_y)**2)
        if dist > 0.2:
            continue
        nspikes = 0
        for tim in control_times:
            aa = som_spikes[ni] [som_spikes[ni] > tim ]
            aa = aa [aa < tim+40]
            nspikes += len(aa)
        control_spikes.append(nspikes)

        nspikes = 0
        for tim in chr_times:
            aa = som_spikes[ni] [som_spikes[ni] > tim ]
            aa = aa [aa < tim+40]
            nspikes += len(aa)
        chr_spikes.append(nspikes)
    control_rates = np.divide(control_spikes,(40/1000)) # convert from spike count to rate
    chr_rates = np.divide(chr_spikes,(40/1000))
    ctrl_mean = np.mean(control_rates)
    chr2_mean = np.mean(chr_rates)
    
    ctrl_err = scipy.stats.sem(control_rates)
    chr2_err = scipy.stats.sem(chr_rates)
    
    return [ctrl_mean, chr2_mean, ctrl_err, chr2_err]
    
def plot_results(results, result_dir, trialnum):
    testSpon = results[0]
    testPVstim = np.array(results[1:1+len(contrast_values)])
    testExc = testPVstim[:,0,0]
    testPV = testPVstim[:,1,0]
    testSST = testPVstim[:,2,0]

    testSSTstim = np.array(results[1:1+len(contrast_values)])
    testExc += testSSTstim[:,0,0]
    testPV += testSSTstim[:,1,0]
    testSST += testSSTstim[:,2,0]

    testExc = np.array(testExc)
    testPV = np.array(testPV)
    testSST = np.array(testSST)

    testExc = testExc/2
    testPV = testPV/2
    testSST = testSST/2
    
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots(1,4,figsize = (12,3))

    contVec = contrast_values

    #testExc = np.random.uniform(np.min(target_exc_contrast),np.max(target_exc_contrast),6)
    ax[0].plot(contVec,target_exc_contrast,linestyle='-',marker='^',color='black',label='target',markersize=10)
    ax[0].plot(contVec,testExc,linestyle='-',marker='o',color='black',label='model',markersize=10)
    ax[0].set_title('Exc',fontsize=15)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[0].set_xlabel('contrast %',fontsize=18)
    ax[0].set_ylabel('firing rate (Hz)',fontsize=15)

    #testPV = np.random.uniform(np.min(target_pv_contrast),np.max(target_pv_contrast),6)
    ax[1].set_title('PV',fontsize=15)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[1].plot(contVec,target_pv_contrast,linestyle='-',marker='^',color='black',label='target',markersize=10)
    ax[1].plot(contVec,testPV,linestyle='-',marker='o',color='black',label='model',markersize=10)
    ax[1].set_xlabel('contrast %',fontsize=18)

    #testSST= np.random.uniform(np.min(target_som_contrast),np.max(target_som_contrast),6)
    ax[2].set_title('SST',fontsize=15)
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax[2].plot(contVec,testSST,linestyle='-',marker='o',color='black',label='model',markersize=10)
    ax[2].plot(contVec,target_som_contrast,linestyle='-',marker='^',color='black',label='target',markersize=10)
    ax[2].set_xlabel('contrast %',fontsize=18)

    dum = [1,2,3] # equally-spaced
    #testSpon = np.random.uniform(np.min(target_spontaneous_rates),np.max(target_spontaneous_rates),3)
    ax[3].plot(dum,testSpon,linestyle='',marker='o',color='black',label='model',markersize=10)
    ax[3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[3].plot(dum,target_spontaneous_rates,linestyle='',marker='^',color='black',label='targets',markersize=10)
    ax[3].set_xticks([1,2,3])
    ax[3].set_xticklabels(['PC','PV','SOM'])
    ax[3].set_title('spontaneous',fontsize=15)
    ax[3].legend(fontsize=15,bbox_to_anchor=[1.8,0.5],loc='center right')
    ax[3].set_xlabel('cell type',fontsize=18)
    
    plt.savefig(f'{result_dir}/result_figs/{trialnum}.png',bbox_inches='tight',dpi=200)
    plt.close()

#sim_parameters = {}
#sim_parameters['gext_baseline'] = 18.
#sim_parameters['g_exc'] = 1.5e-3
#sim_parameters['g_inh'] = 20.*1.5e-3
#sim_parameters['par_gext_rate0'] = 200.
#sim_parameters['par_gext_rate1'] = 400.
#sim_parameters['par_ext_syn_1'] = 10.
#sim_parameters['par_ext_syn_2'] = 3.
#sim_parameters['par_ext_syn_3'] = 2.
#sim_parameters['par_ext_syn_4'] = 8.
#sim_parameters['chr2_str_som']  = 0.1
#sim_parameters['chr2_str_pv']  = 0.05

def print_params(fname, sim_params):
    parnames = sim_params.keys()
    f = open(fname,'w')
    for par in parnames:
        f.write('%s    %s\n'%(par, sim_params[par]))
    f.close()


def read_sim_params(fname):
    togo = {}
    f = open(fname)
    for l in f:
        if l[0] == '#': continue
        stuff = l.split()
        if len(stuff) <1: continue
        togo[stuff[0]] = float(stuff[1]) 
    return togo


def periodic_distance(p1, p2, L=1.0):
    """
    Calculates the periodic distance between two points on a 1x1 2-dimensional plane.

    Parameters
    ----------
    p1 : numpy array
        The first point, with shape (2,).
    p2 : numpy array
        The second point, with shape (2,).
    L : float
        The length of the periodic boundary, default value is 1.0.

    Returns
    -------
    distance : float
        The periodic distance between two points.
    """
    delta = np.abs(p1 - p2)
    delta = np.where(delta > L/2, L - delta, delta)
    return np.sqrt(np.sum(delta**2))


import scipy.stats
# def getStimRateMeans(exc_spikes, exc_positions, rates=False, nsecs = 10, delay=0, binlen = 100):
#     # store the control times and the chr2 stimulation times
#     control_times = []
#     chr_times = []
#     ii = 0
#     is_control = True
#     while True:
#         start_time = ii * 500 + 1000 + delay
#         stop_time = start_time + binlen +delay
#         if stop_time > nsecs*1000: break
#         if is_control:
#             control_times.append(start_time)
#         else:
#             chr_times.append(start_time)   
#         ii += 1
#         if is_control:
#             is_control = False
#         else:
#             is_control = True        
        
#     assert len(control_times) == len(chr_times)
# #     print("NTIMES:", control_times, chr_times)
#     # compute the rates of the cells distal to the chr2 stimulus
#     control_spikes = []
#     chr_spikes = []
#     center_pos = np.array([0.5,0.5])
#     for ni in range(len(exc_positions)):
#         pos_x, pos_y = exc_positions[ni,:]
#         dist = np.sqrt((0.5-pos_x)**2 + (0.5-pos_y)**2)
#         if dist > 0.25:
#             continue
#         nspikes = 0
#         for tim in control_times:
#             aa = exc_spikes[ni] [exc_spikes[ni] > tim ]
#             aa = aa [aa < tim+binlen]
#             nspikes += len(aa)
#         control_spikes.append(nspikes)

#         nspikes = 0
#         for tim in chr_times:
#             aa = exc_spikes[ni] [exc_spikes[ni] > tim ]
#             aa = aa [aa < tim+binlen]
#             nspikes += len(aa)
#         chr_spikes.append(nspikes)
#     control_rates = np.divide(control_spikes,(binlen/1000)*len(control_times)) # convert from spike count to rate
#     chr_rates = np.divide(chr_spikes,(binlen/1000)*len(chr_times))
#     ctrl_mean = np.mean(control_rates)
#     chr2_mean = np.mean(chr_rates)
    
#     ctrl_err = scipy.stats.sem(control_rates)
#     chr2_err = scipy.stats.sem(chr_rates)
    
#     if rates:
#         return [ctrl_mean, chr2_mean, ctrl_err, chr2_err, control_rates, chr_rates]
#     else:
#         return [ctrl_mean, chr2_mean, ctrl_err, chr2_err]
    
def getStimRateMeans_v2(exc_spikes, exc_positions, rates=False, nsecs = 10, delay=0, binlen = 200):
    # store the control times and the chr2 stimulation times
    control_times = []
    chr_times = []
    ii = 2
    is_control = True
    while True:
        start_time = ii * 1000 + 100 + delay
        stop_time = start_time + binlen +delay
        if stop_time > nsecs*1000: break
        if is_control:
            control_times.append(start_time)
        else:
            chr_times.append(start_time)   
        ii += 1
        if is_control:
            is_control = False
        else:
            is_control = True        
        
    assert len(control_times) == len(chr_times)
#     print("NTIMES:", control_times, chr_times)
    # compute the rates of the cells distal to the chr2 stimulus
    control_spikes = []
    chr_spikes = []
    center_pos = np.array([0.5,0.5])
    for ni in range(len(exc_positions)):
        pos_x, pos_y = exc_positions[ni,:]
        dist = np.sqrt((0.5-pos_x)**2 + (0.5-pos_y)**2)
        if dist > 0.25:
            continue
        nspikes = 0
        for tim in control_times:
            aa = exc_spikes[ni] [exc_spikes[ni] > tim ]
            aa = aa [aa < tim+binlen]
            nspikes += len(aa)
        control_spikes.append(nspikes)

        nspikes = 0
        for tim in chr_times:
            aa = exc_spikes[ni] [exc_spikes[ni] > tim ]
            aa = aa [aa < tim+binlen]
            nspikes += len(aa)
        chr_spikes.append(nspikes)
    control_rates = np.divide(control_spikes,(binlen/1000)*len(control_times)) # convert from spike count to rate
    chr_rates = np.divide(chr_spikes,(binlen/1000)*len(chr_times))
    ctrl_mean = np.mean(control_rates)
    chr2_mean = np.mean(chr_rates)
    
    ctrl_err = scipy.stats.sem(control_rates)
    chr2_err = scipy.stats.sem(chr_rates)
    
    if rates:
        return [ctrl_mean, chr2_mean, ctrl_err, chr2_err, control_rates, chr_rates]
    else:
        return [ctrl_mean, chr2_mean, ctrl_err, chr2_err]
   
getStimRateMeans = getStimRateMeans_v2

def getSpontMedians(all_spikes, start_time = 1000., stop_time = 10000.):
    
    exc_spikes = all_spikes[:8000]
    pv_spikes = all_spikes[8000:9000]
    sst_spikes = all_spikes[9000:]
    
    tlen = (stop_time - start_time)/1000.
    
    exc_nspikes = []
    for spikes in exc_spikes:
        aa = spikes[spikes > start_time ]
        aa = aa [aa < stop_time]
        exc_nspikes.append(len(aa)/tlen)
        
    pv_nspikes = []
    for spikes in pv_spikes:
        aa = spikes[spikes > start_time ]
        aa = aa [aa < stop_time]
        pv_nspikes.append(len(aa)/tlen)
        
    sst_nspikes = []
    for spikes in sst_spikes:
        aa = spikes[spikes > start_time ]
        aa = aa [aa < stop_time]
        sst_nspikes.append(len(aa)/tlen)
        
    return np.median(exc_nspikes), np.median(pv_nspikes), np.median(sst_nspikes)


from scipy.optimize import curve_fit

def linear_model(x, a, b):
    return a * x + b

import numpy as np
from scipy.optimize import curve_fit

# naka-rushton = (m+((Rm.*(x.^n))./((x.^n) + (c50.^n)))))
# parameter = [Rm, n, c50, m]
# p0 = [150, 5, 0.1, 50]
# Upper = [250, 30, 0.26, 200]
# Lower = [0, 0, 0.04, 0]
p0 = [50, 150, 5, 0.1] 
Upper = [200, 250, 15, 0.2]
# Lower = [0, 0, 0, 0.04]
Lower = [0, 0, 0, 0.05]
bounds = (Lower, Upper)

def naka_rushton(x, m, C, n, k):
    return m + C * (x**n) / (x**n + k**n)

def fit_naka_rushton(x, y, sigma=None, p0 = p0, bounds=bounds):
    params, cov = curve_fit(naka_rushton, x, y, bounds=bounds, sigma=sigma, p0=p0, maxfev = 1000000)
    return params, cov
