'''
33 bus distribution network with DGs and loads participating in DR program.

'''

import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.run import runpp
from utils import *
from collections import defaultdict
from setting import *
import matplotlib.pyplot as plt
import pandapower.timeseries as timeseries
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly

def network_comp(TIMESTEPS = TIMESTEPS, curtailed = np.zeros(5)):
    net = pp.create_empty_network()
    for i in range(N_BUS): #zero-indexed  
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")  
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Substation")

    # TODO c_nf_per_km, max_i_ka ?
    for to_bus, from_bus, r_ohm, x_ohm in line_data:
        pp.create_line_from_parameters(net, c_nf_per_km = 10, max_i_ka = 0.4, 
                                       from_bus = from_bus, to_bus = to_bus, length_km = 1, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm) 

    #  gen -> voltage controlled PV nodes. sgen -> no voltage control
    pv1 = pp.create_sgen(net, bus=13, p_mw=4.231, q_mvar=0, name="PV1", index = 0)

    # Add wind-based DG at Bus 5 (p_mw = p_rated?)
    wt1 = pp.create_gen(net, bus=4, p_mw=0.5, min_p_mw = 0, max_p_mw = WTRATED, vm_pu=1.0, name="WT1", index = 1)

    # Add conventional DG at Bus 12 #min = 35, max = 300, p rated up, p rated down
    cdg1 = pp.create_gen(net, bus=11, p_mw=0.07, min_p_mw=PGEN_MIN, max_p_mw=PGEN_MAX, vm_pu=1.0, name="CDG1", index = 2)

    # TODO add storage
    #pp.create_storage(net, bus, p_mw, max_e_mwh, q_mvar=0, sn_mva=nan, 
    # soc_percent=nan, min_e_mwh=0.0, name=None, index=None, scaling=1.0, type=None, in_service=True, 
    # max_p_mw=nan, min_p_mw=nan, max_q_mvar=nan, min_q_mvar=nan, controllable=nan)
    
    consumers = {}

    for i, bus in enumerate(range(0, 32)):  # Range for 0-based indexing
        consumers[f'C{i}'] = pp.create_load(
            net, 
            bus=bus,  # Use 0-based indexing for buses
            p_mw=0.05, 
            max_p_mw=load_data[i][1] / 1000,  
            max_q_mw=load_data[i][2] / 1000,  
            name=f"C{i}",  
            index = f"C{i}", 
            controllable=True
        )
    ConstControl(net, element='gen', variable='p_mw', element_index=wt1, 
                 profile_name="P_wind", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_wind)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv1, 
                 profile_name="P_solar", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_sun)
    for i, element_index in enumerate(consumers.values(), start = 1):
        ConstControl(net, element='load', variable='p_mw', element_index=element_index, 
                     profile_name=f"Customer_{i}", data_source=data_source_consumers, initial_powerflow=False, recycle=False, run_control=True)  
    # Run power flow analysis
    ow = create_output_writer(net, TIMESTEPS, output_dir=filepath_results)  
    timeseries.run_timeseries(net, time_steps = TIMESTEPS)
    print("Time series simulation completed.")
    line_losses = (net.res_line['p_from_mw'] - net.res_line['p_to_mw']).sum()

    net.gen.at[cdg1, 'p_mw'] = line_losses

    return line_losses, net
if __name__ == "__main__":  
    line_losses, net = network_comp()
    #pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=True, ax=None)
    #pf_res_plotly(net)
    
    #plot_results(filepath_results)