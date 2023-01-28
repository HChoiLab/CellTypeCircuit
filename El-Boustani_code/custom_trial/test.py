import matplotlib.pyplot as plt
import nest
import numpy as np
import os
from pynestml.frontend.pynestml_frontend import generate_nest_target

NEST_SIMULATOR_INSTALL_LOCATION = nest.hl_api.sli_func("statusdict/prefix ::")

nestml_active_dend_model = '''
neuron iaf_psc_exp_active_dendrite:
  state:
    V_m mV = 0 mV     # membrane potential
    t_dAP ms = 0 ms   # dendritic action potential timer
    I_dAP pA = 0 pA   # dendritic action potential current magnitude
  end

  equations:
    # alpha shaped postsynaptic current kernel
    kernel syn_kernel = (e / tau_syn) * t * exp(-t / tau_syn)
    recordable inline I_syn pA = convolve(syn_kernel, spikes_in)
    V_m' = -(V_m - E_L) / tau_m + (I_syn + I_dAP + I_e) / C_m
  end

  parameters:
    C_m pF = 250 pF          # capacity of the membrane
    tau_m ms = 20 ms         # membrane time constant
    tau_syn ms = 10 ms       # time constant of synaptic current
    V_th mV = 25 mV          # action potential threshold
    V_reset mV = 0 mV        # reset voltage
    I_e    pA = 0 pA         # external current
    E_L    mV = 0 mV         # resting potential

    # dendritic action potential
    I_th pA = 100 pA         # current threshold for a dendritic action potential
    I_dAP_peak pA = 150 pA   # current clamp value for I_dAP during a dendritic action potential
    T_dAP ms = 10 ms         # time window over which the dendritic current clamp is active
  end

  input:
    spikes_in pA <- spike
  end

  output: spike

  update:
    # solve ODEs
    integrate_odes()

    if t_dAP > 0 ms:
      t_dAP -= resolution()
      if t_dAP <= 0 ms:
        # end of dendritic action potential
        t_dAP = 0 ms
        I_dAP = 0 pA
      end
    end

    if I_syn > I_th:
      # current-threshold, emit a dendritic action potential
      t_dAP = T_dAP
      I_dAP = I_dAP_peak
    end

    # emit somatic action potential
    if V_m > V_th:
      emit_spike()
      V_m = V_reset
    end
  end
end
'''

with open("iaf_psc_exp_active_dendrite.nestml", "w") as nestml_model_file:
    print(nestml_active_dend_model, file=nestml_model_file)

generate_nest_target(input_path="iaf_psc_exp_active_dendrite.nestml",
                     target_path="/tmp/nestml-active-dend-target",
                     module_name="nestml_active_dend_module",
                     suffix="_nestml",
                     logging_level="ERROR",  # try "INFO" for more debug information
                     codegen_opts={"nest_path": NEST_SIMULATOR_INSTALL_LOCATION})
nest.Install("nestml_active_dend_module")

