import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    
