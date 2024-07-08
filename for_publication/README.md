# PV/SST Circuit model

This code reproduces the modeling results for the paper ["Cell-type specific lateral inhibition distinctly transforms perceptual and corresponding neural sensitivity"](https://www.biorxiv.org/content/10.1101/2023.11.10.566605v1) by Joseph Del Rosario, Stefano Coletta, Soon Ho Kim, Zach Mobille, Kayla Peelman, Brice Williams, Alan J Otsuki, Alejandra Del Castillo Valerio, Kendell Worden, Lou T. Blanpain, Lyndah Lovell, Hannah Choi, Bilal Haider.

## Required Installations

Running simulations require installation of [PyNEST](https://nest-simulator.readthedocs.io/en/v3.3/ref_material/pynest_apis.html) (version 3.3.0 used) and [NESTML](https://nestml.readthedocs.io/en/latest/) (version 5.1.0).

NESTML is used to install the dendritic integration model ```iaf_cond_exp_dend.nestml```. To install the model, first follow the directions [here](https://nestml.readthedocs.io/en/latest/installation.html) to configure settings. Then, use the commands 
```
cd code/custom_model
python install_dendritic.py
```
to install the model.
