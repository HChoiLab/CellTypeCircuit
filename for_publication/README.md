# PV/SST Circuit model

This code reproduces the figures from the paper "Cell-type specific lateral inhibition distinctly transforms perceptual and corresponding neural sensitivity" by ... (paper details).

## Required Installations

Running simulations require installation of [PyNEST](https://nest-simulator.readthedocs.io/en/v3.3/ref_material/pynest_apis.html) (version 3.3.0 used) and [NESTML](https://nestml.readthedocs.io/en/latest/) (version 5.1.0).

NESTML is used to install the dendritic integration model ```iaf_cond_exp_dend.nestml```. To install the model, use 
```
cd code/custom_model
python install_dendritic.py
```

```run.py``` is for the base model without the dendritic nonlinearity, ```nlrun.py``` uses the dendritic model for excitatory neurons.
