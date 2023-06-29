# PV/SST Circuit model

This code reproduces the figures from the paper "Cell-type specific lateral inhibition distinctly transforms perceptual and corresponding neural sensitivity" by ... (paper details).

## Required Installations

Running simulations require installation of [```PyNEST```](https://nest-simulator.readthedocs.io/en/v3.3/ref_material/pynest_apis.html) (version 3.3.0 used) and [NESTML](https://nestml.readthedocs.io/en/latest/) (version 


# (Old Version)

## El-Boustani
```El-Boustani``` contains original and modified code of the grid model received from Sami El-Boustani.

Trial 3: Fixed capacitance from 0.2 to 0.25nF. Spontaneous, PV stim, and SOM Stim code.

Trial 4: Added multiplier to external stimulation. Use, for example: ```python gmult.py 0.1```, in which the default height of the Gaussian will be multiplied by the factor 0.1.

Trial 5:

Trial 6: Local/Distal ChR2 stimulation of SST/PV with contrast curves.

Trial 7: add long-range SOM connections.

Trial 9: LRv22b and LRv22 contain versions used in results produced around COSYNE submission

## newCode

Code is rewritten in pyNEST3.3.0. NESTML required for installing the dendritic integration model ```iaf_cond_exp_dend.nestm```.

```python iaf_p39.py``` to install the model.

```run.py``` is for the base model without the dendritic nonlinearity, ```nlrun.py``` uses the dendritic model for excitatory neurons.
