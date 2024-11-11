'''
33 bus distribution network with DGs and loads participating in DR program.

'''
import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from collections import defaultdict
from setting import *
import matplotlib.pyplot as plt
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.plotting.plotly import simple_plotly
power_data_path_wind = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\wt_profile.csv"
power_data_path_sun = r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\pv_profile.csv"
power_data_consumers =r"C:\Users\jlhb83\Desktop\Python Projects\powersystemRL\data\derived\load_profile.csv"
datasource_wind = pd.read_csv(power_data_path_wind)
datasource_sun = pd.read_csv(power_data_path_sun)
datasource_consumers = pd.read_csv(power_data_consumers) * PEAK_P_DEMAND / 100
datasource_consumers.drop(columns=['hourly_load%'], inplace=True)
data_source_wind = DFData(datasource_wind)
data_source_sun = DFData(datasource_sun)
data_source_consumers = DFData(datasource_consumers)
# TODO how to control generator outputs based on decision variables
def network_comp():
    net = pp.create_empty_network()
    for i in range(N_NODE):  
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")  
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Substation")
    #(to_bus, from_bus, r_ohm(total), x_ohm)
    
    line_data = [
    (0, 1, 0.0922, 0.0470), ##SS
    (1, 2, 0.4930, 0.2511),
    (2, 3, 0.3660, 0.1864),
    (3, 4, 0.3811, 0.1941),
    (4, 5, 0.8190, 0.7070),
    (5, 6, 0.1872, 0.6188),
    (6, 7, 1.7114, 0.2351),
    (7, 8, 1.0300, 0.7400),
    (8, 9, 1.0400, 0.7400),
    (9, 10, 0.1966, 0.0650),
    (10, 11, 0.3744, 0.1238),
    (11, 12, 1.4680, 0.1550),
    (12, 13, 0.5416, 0.7129),
    (13, 14, 0.5910, 0.5260),
    (14, 15, 0.7463, 0.5450),
    (15, 16, 1.2890, 1.7210),
    (16, 17, 0.7320, 0.5740),
    (18, 1, 0.1640, 1.1565),
    (18, 19, 1.5402, 1.3554),
    (19, 20, 0.4095, 0.4784),
    (20, 21, 0.7089, 0.9373),
    (2, 22, 0.4512, 0.3083),
    (22, 23, 0.8980, 0.7091),
    (23, 24, 0.8960, 0.7011),
    (5, 25, 0.2030, 0.1034),
    (25, 26, 0.2842, 0.1447),
    (26, 27, 1.0590, 0.9337),
    (27, 28, 0.8042, 0.7006),
    (28, 29, 0.5075, 0.2585),
    (29, 30, 0.9744, 0.9630),
    (30, 31, 0.3105, 0.3619),
    (31, 32, 0.3410, 0.5302)]
    #(bus,PL,QL)
    load_data = [
    (1, 100, 60),
    (2, 90, 40),
    (3, 120, 80),
    (4, 60, 30),
    (5, 60, 20),
    (6, 200, 100),
    (7, 200, 100),
    (8, 60, 20),
    (9, 60, 20),
    (10, 45, 30),
    (11, 60, 35),
    (12, 60, 35),
    (13, 120, 80),
    (14, 60, 10),
    (15, 60, 20),
    (16, 60, 20),
    (17, 90, 40),
    (18, 90, 40),
    (19, 90, 40),
    (20, 90, 40),
    (21, 90, 40),
    (22, 90, 50),
    (23, 420, 200),
    (24, 420, 200),
    (25, 60, 25),
    (26, 60, 25),
    (27, 60, 20),
    (28, 120, 70),
    (29, 200, 600),
    (30, 150, 70),
    (31, 210, 100),
    (32, 60, 40)]

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

    timeseries.run_timeseries(net)
    res['bus_voltage'].append(net.res_bus["vm_pu"].values)
    res['gen_output'].append(net.res_gen["p_mw"].values)
    res['sgen_output'].append(net.res_sgen["p_mw"].values)
   
    ids = {
    'pv1': pv1,
    'wt1': wt1,
    'cdg1': cdg1,
    'c1': c1,'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}
    return net, ids, res
if __name__ == "__main__":  
    network_comp()