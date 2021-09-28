import matplotlib.pyplot as plt
import numpy as np
import csv
import qcodes.instrument_drivers.rohde_schwarz.ZNB as ZNB
from qcodes.instrument_drivers.rohde_schwarz.ZNB import ZNBChannel
import qcodes



def get_data(param, freq_start = 100e6, freq_stop = 1e9, freq_num = 200, vna_power=-30):
    ZNB.ZNB.close_all()
    vna = ZNB.ZNB('VNA', f'TCPIP0::{IP}::INSTR', init_s_params=False)
    station = qcodes.Station(vna)

    vna.add_channel(param)

    vna.cont_meas_on()
    vna.display_single_window()
    vna.rf_on()

    freq_start = freq_start
    freq_stop = freq_stop
    freq_num = freq_num
    freq = np.linspace(freq_start, freq_stop, freq_num)

    vna.channels.format('Complex')

    vna.channels.start(freq_start)
    vna.channels.stop(freq_stop)
    vna.channels.npts(freq_num)

    vna.channels.power(vna_power)
    vna.channels.autoscale()

    trace = vna.channels.trace.get()[0]

    return {'freq' :freq, 'trace' :trace}
