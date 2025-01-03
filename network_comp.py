'''
33 bus distribution network with DGs and loads participating in DR program.

'''

import pandapower as pp
import pandas as pd
from pandapower.control import ConstControl
from pandapower.run import runpp
from utils import *
from setting import *
import matplotlib.pyplot as plt
import pandapower.timeseries as timeseries
from pandapower.plotting.plotly import simple_plotly
from pandapower.plotting.plotly import pf_res_plotly
from pandapower.control.basic_controller import Controller

class DG_controller(Controller):
    """
    Custom DG Controller for managing distributed generation with curtailment and line loss control.
    """
    def __init__(self, net, element_index, scaled_action, prev_line_losses, 
                 data_source=None, order=0, level=0, **kwargs):
        super().__init__(net, in_service=True, recycle=False, order=order, level=level, **kwargs)

        # Attributes for distributed generation
        self.element_index = element_index  # Index of the DG in the network
        self.curtailment_indices = ["C8", "C21", "C13", "C29", "C24"]
        self.scaled_action = scaled_action
        self.prev_line_losses = prev_line_losses
        self.applied = False

        # Profile-related attributes
        self.data_source = data_source
        self.last_time_step = None

    def is_converged(self, net):
        """
        Check if the control step has already been applied in this iteration.
        """
        return self.applied

    def write_to_net(self, net):
        """
        Write updated power and state to the net.
        """
        # Update curtailed loads
        load_values = net.load.loc[self.curtailment_indices, "p_mw"].to_numpy()
        updated_load_values = np.maximum(load_values - self.scaled_action[:-1], 0)
        net.load.loc[self.curtailment_indices, "p_mw"] = updated_load_values

        # Update DG power based on previous line losses
        net.gen.loc[self.element_index, "p_mw"] = self.prev_line_losses

    def control_step(self, net):
        """
        Execute the control logic to adjust loads and generation based on action and state.
        """
        self.write_to_net(net)
        self.applied = True

    def time_step(self, net, time):
        """
        Update action or state variables at each simulation timestep.
        """
        if self.last_time_step is not None:
            # Custom logic to manage attributes over time
            pass  # Update logic for time-series attributes if needed
        self.last_time_step = time
        self.applied = False  # Reset for next control step

def network_comp(TIMESTEPS,scaled_action,prev_line_losses):
    net = pp.create_empty_network()
    for i in range(N_BUS): #zero-indexed  
        pp.create_bus(net, vn_kv=12.66, name=f"Bus {i}")  
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Substation")

    # TODO c_nf_per_km, max_i_ka ?
    for to_bus, from_bus, r_ohm, x_ohm in line_data:
        pp.create_line_from_parameters(net, c_nf_per_km = 10, max_i_ka = 0.4, 
                                       from_bus = from_bus, to_bus = to_bus, length_km = 1, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm) 

    #  gen -> voltage controlled PV nodes. sgen -> no voltage control
    pv1 = pp.create_sgen(net, bus=13, p_mw=0, q_mvar=0, name="PV1", index = 0)
    wt1 = pp.create_gen(net, bus=4, p_mw=0.5, min_p_mw = 0, max_p_mw = WTRATED, vm_pu=1.0, name="WT1", index = 1)
    cdg1 = pp.create_gen(net, bus=11, p_mw=0, min_p_mw=PGEN_MIN, max_p_mw=PGEN_MAX, vm_pu=1.0, name="CDG1", index = 2)
    dg_controller = DG_controller(net=net,element_index=2, scaled_action=scaled_action,prev_line_losses=prev_line_losses,order=1)
    single_step = False
    for i, bus in enumerate(range(32)):
        pp.create_load(
            net, 
            bus=bus,  # Use 0-based indexing for buses
            p_mw=0.05, 
            max_p_mw=load_data[i][1],  
            max_q_mw=load_data[i][2],  
            name=f"C{i}",  
            index = f"C{i}", 
            controllable=True
        )
        ConstControl(net, element='load', variable='p_mw', element_index=f"C{i}", 
                     profile_name=f"C{i}", data_source=data_source_consumers_original, initial_powerflow=False, recycle=False, run_control=True)  
    ConstControl(net, element='gen', variable='p_mw', element_index=wt1, 
                 profile_name="P_wind", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_wind)
    ConstControl(net, element='sgen', variable='p_mw', element_index=pv1, 
                 profile_name="P_solar", recycle=False, run_control=True, initial_powerflow=False, data_source=data_source_sun)
    # Run power flow analysis
    
    if not single_step: 
        #ow = create_output_writer(net, TIMESTEPS, output_dir=filepath_results)  
        
        #print(pp.diagnostic(net))
        timeseries.run_timeseries(net, time_steps = TIMESTEPS, numba = False)
        #print("Time series simulation completed.")
        line_losses = net.res_line['pl_mw'].sum()
    else:
        #print("Starting simulation with control loop...")
        
        pp.control.run_control(
            net,
            ctrl_variables=None,  
            max_iter=30,          
            continue_on_lf_divergence=False  
        )
        #print("Control loop simulation completed.")

        # Calculate line losses
        line_losses = net.res_line['pl_mw'].sum()

    return line_losses, net
 
#if __name__ == "__main__":
#    line_losses, net = network_comp((0,0),scaled_action=[0.01,0.01,0.01,0.01,0.01,0.1],prev_line_losses=0)
    #pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=True, ax=None)  
    
    #
    #pf_res_plotly(net)
    
    #plot_results(filepath_results)