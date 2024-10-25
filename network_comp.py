'''
33 bus distribution network with DGs and loads participating in DR program.

'''
import pandapower as pp
from pandapower.control import ConstControl
from setting import *
from pandapower.timeseries.data_sources.frame_data import DFData
def network_comp(wind_data,solar_data):
    net = pp.create_empty_network()
    for i in range(N_NODE+1):  
        pp.create_bus(net, vn_kv=11, name=f"Bus {i}")  
    pp.create_ext_grid(net, bus=0, vm_pu=1.0, name="Substation")
    #(from_bus, to_bus, r_ohm(total), x_ohm)
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
    (15, 16, 0.7320, 1.7210),
    (16, 17, 0.7320, 1.7210),
    (17, 18, 0.7320, 1.7210),
    (18, 19, 0.7320, 1.7210),
    (19, 20, 0.7320, 1.7210),
    (20, 21, 0.7320, 1.7210),
    (21, 22, 0.7320, 1.7210),
    (22, 23, 0.7320, 1.7210),
    (23, 24, 0.7320, 1.7210),
    (24, 25, 0.7320, 1.7210),
    (25, 26, 0.7320, 1.7210),
    (26, 27, 0.7320, 1.7210),
    (27, 28, 0.7320, 1.7210),
    (28, 29, 0.7320, 1.7210),
    (29, 30, 0.7320, 1.7210),
    (30, 31, 0.7320, 1.7210),
    (31, 32, 0.7320, 1.7210),
    (32, 33, 0.7320, 1.7210),
   # (33, 34, 0.7320, 1.7210),
   ]
    #(bus,PL,QL)
    load_data = [
    (1, 100, 60)
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
    (18, 60, 40),
    (19, 120, 40),
    (20, 60, 40),
    (21, 60, 40),
    (22, 60, 50),
    (23, 90, 200),
    (24, 60, 200),
    (25, 60, 25),
    (26, 60, 25),
    (27, 60, 20),
    (28, 60, 70),
    (29, 60, 600),
    (30, 60, 70),
    (31, 60, 100),
    (32, 60, 40),
    #(33, 60, 20),
    ]
    for from_bus, to_bus, r_ohm, x_ohm in line_data:
        pp.create_line(net, from_bus = from_bus, to_bus = to_bus, r_ohm=r_ohm, x_ohm=x_ohm) #max_i_ka=0.2?
pp.create_bus()
    # Add solar-based DG at Bus 14
    # TODO interval optimisation here
    pp.create_sgen(net, bus=14, p_mw=4.231, q_mvar=0, name="Solar DG")

    # Add wind-based DG at Bus 5
    pp.create_gen(net, bus=5, p_mw=0.5, vm_pu=1.0, name="Wind DG")

    # Add conventional DG at Bus 12
    pp.create_gen(net, bus=12, p_mw=0.07, min_p_mw=0.035, max_p_mw=0.3, vm_pu=1.0, name="Conventional DG")

    for bus, p_kw, q_kvar in load_data:
        pp.create_load(net, bus=bus-1, p_mw=p_kw / 1000, q_mvar=q_kvar / 1000)

    # Run power flow analysis
    pp.runpp(net)

    # Output the results
    print(net.res_bus)
    print(net.res_line)
    print(net.res_gen)