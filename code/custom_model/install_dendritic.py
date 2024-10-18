#import matplotlib.pyplot as plt
import nest
import numpy as np
import os
from pynestml.frontend.pynestml_frontend import generate_nest_target

NEST_SIMULATOR_INSTALL_LOCATION = nest.ll_api.sli_func("statusdict/prefix ::")
print(NEST_SIMULATOR_INSTALL_LOCATION)

generate_nest_target(input_path="iaf_cond_exp_dend.nestml",
                     target_path="/tmp/nestml-target",
                     module_name="nestml_iaf_module",
                     suffix="_nestml",
                     logging_level="ERROR",  # try "INFO" for more debug information
                     codegen_opts={"nest_path": NEST_SIMULATOR_INSTALL_LOCATION})

nest.Install("nestml_iaf_module")
