import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    
    return [ctrl_mean, chr2_mean]


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
    
    return [ctrl_mean, chr2_mean]


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
    
    return [ctrl_mean, chr2_mean]
    
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


def read_sim_params(fname):
    togo = {}
    f = open(fname)
    for l in f:
        stuff = l.split()
        if len(stuff) <1: continue
        togo[stuff[0]] = float(stuff[1]) 
    return togo
