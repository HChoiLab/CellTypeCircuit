# PV/SST Circuit model

This code reproduces the modeling results for the paper ["Lateral inhibition in V1 controls neural & perceptual contrast sensitivity"](https://www.biorxiv.org/content/10.1101/2023.11.10.566605v2) by Joseph Del Rosario, Stefano Coletta, Soon Ho Kim, Zach Mobille, Kayla Peelman, Brice Williams, Alan J Otsuki, Alejandra Del Castillo Valerio, Kendell Worden, Lou T. Blanpain, Lyndah Lovell, Hannah Choi, Bilal Haider.

<div style="text-align: center;">
<img src="https://github.com/user-attachments/assets/83eac280-935d-41bb-aacd-e6018a2a4192" alt="image" width="400">
</div>

## Required Installations

Running simulations require installation of [PyNEST](https://nest-simulator.readthedocs.io/en/v3.3/ref_material/pynest_apis.html) (version 3.3.0 used) and [NESTML](https://nestml.readthedocs.io/en/latest/) (version 5.1.0). (NOTE: We used the aforementioned NESTML version to produce the figures in the manuscript, but the current versions of pyNEST and NESTML cause issues when installing the dendritic integration model. We are working on a revised script to address this.)

NESTML is used to install the dendritic integration model ```iaf_cond_exp_dend.nestml```. To install the model, first follow the directions [here](https://nestml.readthedocs.io/en/latest/installation.html) to configure settings. Then, use the commands 
```
cd code/custom_model
python install_dendritic.py
```
to install the model. Some scripts also use matplotlib and scipy for visualization and analysis.

## Fig. 4C and D

To reproduce Figs. 4C-D, run the distal stimulation protocol for PV and SST stimulation and 5 contrast values. Repeat for 100 seeds (which uses 100 different random seeds for network initialization and input spike generation).

Figs. 4C-D use alpha = 0.07 (strength of dendritic nonlinearity) and pfar_sst = 0.8 (proportion of sst projections which are long-range). To initialize the working directory use the command
```
python initiallize_parameters.py 0.07 0.8
cd work_dir/base_a0.07_pf0.8
```
then run
```
python ../../code/run_distal_stim.py 1 1
```
where the first argument is the RNG seed and the second is the stimulation condition (vary from 1-10 for each seed). We recommend running simulations in parallel if you intend on running many seeds. After running simulations, the contrast response curves can be plotted using ```Fig3.ipynb```.

## Fig. 5

Repeat distal stimulations for alpha 0.04-0.07 (0.002 linear spacing) and pf 0-0.8 (0.1 spacing)  over 100 seeds. The notebook computes slope changes and plots distributions. Pre-compiled MI changes are stored in ```compiled_data/Exc_heatmaps.mat```.

## Fig. S9

Plots spatial profile of network activity using simualtions from the above.

## Fig. S10

Test for inhibition stabilization. Use ```code/run_delin.py``` to run simulations with inhibition deleted from the network. Inhibition can be deleted locally or globally by varying the ```del_range``` parameter. Initialize using
```
python initialize_delete_inhibition.py
cd work_dir/dl_0.0
python ../../run_delin.py 1 1
```
etc.

## Fig. S11
To produce the data for figure S11, run the ```run_distal_stim.py``` script with parameter values pfar_sst = 0.8, alpha = 0.07 in the spontaneous condition and at the contrast values = 0.05 for both PV and SST stimulation (30 network seeds). To produce the figures, run the notebook ```FigS11.ipynb```

## Fig. S12
To produce the data for figure S12, run the scripts ```run_local_stim.py```, ```run_0p25_stim.py```, and ```run_distal_stim.py``` over the parameter values pfar_sst = 0.2, 0.5, 0.8 and alpha = 0.04, 0.05, 0.06, 0.07, 0.08 with 10 different network seeds across contrast values = 0.02, 0.05, 0.1, 0.18, 0.33. To produce the plots, run the jupyter notebook ```FigS12.ipynb```

## Fig. S13
To produce the data for figure S13, run the ```run_distal_stim.py``` script with parameter values pfar_sst = 0.2, 0.8 and alpha = 0.07 across contrast values = 0.02, 0.05, 0.1, 0.18, 0.33 for both PV and SST stimulation protocols (9 network seeds). To produce the plots, run the notebook ```FigS13.ipynb```

## Fig. S22
To produce the data for figure S22, run the script ```run_varyDist_stim.py``` with pfar_sst = 0.8, alpha = 0.08 over 11 network seeds across contrast values = 0.02, 0.05, 0.1, 0.18, 0.33. The fourth argument to ```run_varyDist_stim.py``` is the laser position, which should range from 0 to 0.5. We use 11 evenly-spaced values in Fig. S22. To produce the plots, run the notebook ```FigS22.ipynb```

## Fig S24
To produce the data for figure S24, run ```run_distal_stim_noAnti.py``` with pfar = 0.2, 0.8, alpha = 0.07 over 9 network seeds across contrast values = 0.02, 0.05, 0.1, 0.18, 0.33 for both PV and SST stimulation protocols. To produce the plots, run the notebook ```FigS24.ipynb```
