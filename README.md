# PV/SST Circuit model

This code reproduces the modeling results for the paper ["Cell-type specific lateral inhibition distinctly transforms perceptual and corresponding neural sensitivity"](https://www.biorxiv.org/content/10.1101/2023.11.10.566605v1) by Joseph Del Rosario, Stefano Coletta, Soon Ho Kim, Zach Mobille, Kayla Peelman, Brice Williams, Alan J Otsuki, Alejandra Del Castillo Valerio, Kendell Worden, Lou T. Blanpain, Lyndah Lovell, Hannah Choi, Bilal Haider.

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
