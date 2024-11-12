'''
33 bus distribution network with DGs and loads participating in DR program.

'''

import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from utils import *
from collections import defaultdict
from setting import *
import matplotlib.pyplot as plt
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.plotting.plotly import simple_plotly
filepath_results = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived"
power_data_path_wind = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\wt_profile.csv"
power_data_path_sun = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\pv_profile.csv"
power_data_consumers =r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\load_profile.csv"
datasource_wind = pd.read_csv(power_data_path_wind)
datasource_sun = pd.read_csv(power_data_path_sun)
datasource_consumers = pd.read_csv(power_data_consumers) * PEAK_P_DEMAND / 100 #individual consumer profiles in percentage

data_source_wind = DFData(datasource_wind)
data_source_sun = DFData(datasource_sun)
data_source_consumers = DFData(datasource_consumers)

def network_comp():
    net = pp.create_empty_network()
    for i in range(N_NODE):  
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")  
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Substation")
    # TODO c_nf_per_km, max_i_ka ?
    for to_bus, from_bus, r_ohm, x_ohm in line_data:
        pp.create_line_from_parameters(net, c_nf_per_km = 10, max_i_ka = 0.4, 
                                       from_bus = from_bus, to_bus = to_bus, length_km = 1, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm) 

    #  gen -> voltage controlled PV nodes. sgen -> no voltage control
    pv1 = pp.create_sgen(net, bus=13, p_mw=4.231, q_mvar=0, name="PV1")

    # Add wind-based DG at Bus 5 (p_mw = p_rated?)
    wt1 = pp.create_gen(net, bus=4, p_mw=0.5, min_p_mw = 0, max_p_mw = WTRATED, vm_pu=1.0, name="WT1")

    # Add conventional DG at Bus 12 #min = 35, max = 300, p rated up, p rated down
    cdg1 = pp.create_gen(net, bus=11, p_mw=0.07, min_p_mw=PGEN_MIN, max_p_mw=PGEN_MAX, vm_pu=1.0, name="CDG1")

    # TODO add storage
    #pp.create_storage(net, bus, p_mw, max_e_mwh, q_mvar=0, sn_mva=nan, 
    # soc_percent=nan, min_e_mwh=0.0, name=None, index=None, scaling=1.0, type=None, in_service=True, 
    # max_p_mw=nan, min_p_mw=nan, max_q_mvar=nan, min_q_mvar=nan, controllable=nan)
    #add consumers
    c1= pp.create_load(net, bus= 8, p_mw = 0.05, max_p_mw =PEAK_P_DEMAND, max_q_mw = PEAK_Q_DEMAND, name="C1",controllable=True)
    c2 = pp.create_load(net, bus= 21, p_mw = 0.05,max_p_mw =PEAK_P_DEMAND, max_q_mw = PEAK_Q_DEMAND, name="C2",controllable=True)
    c3 = pp.create_load(net, bus= 13, p_mw = 0.05,max_p_mw =PEAK_P_DEMAND, max_q_mw = PEAK_Q_DEMAND, name="C3",controllable=True)
    c4 = pp.create_load(net, bus= 29, p_mw = 0.05,max_p_mw =PEAK_P_DEMAND, max_q_mw = PEAK_Q_DEMAND, name="C4",controllable=True)
    c5 = pp.create_load(net, bus= 24, p_mw = 0.05,max_p_mw =PEAK_P_DEMAND, max_q_mw = PEAK_Q_DEMAND, name="C5",controllable=True)
    for bus, p_kw, q_kvar in load_data:
        pp.create_load(net, bus= bus, p_mw=p_kw / 1000, q_mvar=q_kvar / 1000)

    ConstControl(net, element='gen', variable='p_mw', element_index=wt1, 
                 profile_name="P_wind", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_wind)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv1, 
                 profile_name="P_solar", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_sun)
    ConstControl(net, element='load', variable='p_mw', element_index=c1, 
                 profile_name="C1", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_consumers)   
    ConstControl(net, element='load', variable='p_mw', element_index=c2, 
                 profile_name="C2", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_consumers)  
    ConstControl(net, element='load', variable='p_mw', element_index=c3, 
                 profile_name="C3", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_consumers)  
    ConstControl(net, element='load', variable='p_mw', element_index=c4, 
                 profile_name="C4", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_consumers)  
    ConstControl(net, element='load', variable='p_mw', element_index=c5, 
                 profile_name="C5", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_consumers)  
    res = defaultdict(list)
    # Run power flow analysis

    ow = create_output_writer(net, TIMESTEPS, output_dir=filepath_results)  # Step 4: Create output writer
    timeseries.run_timeseries(net, TIMESTEPS)  # Step 5: Run time series simulation
    print("Time series simulation completed.")

    ids = {
    'pv1': pv1,
    'wt1': wt1,
    'cdg1': cdg1,
    'c1': c1,'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}
    
    return net, ids, res
if __name__ == "__main__":  
    net, ids, res = network_comp()
    plot_results(filepath_results)
    