import matplotlib.pyplot as plt
import numpy as np
import csv, os
import qcodes.instrument_drivers.rohde_schwarz.ZNB as ZNB
import qcodes


def get_data(param, plot_phase, exp_path, freq_start = 100e6, freq_stop = 1e9, freq_num = 200, vna_power=-30):
    IP = '10.208.234.8'
    ZNB.ZNB.close_all()
    vna = ZNB.ZNB('VNA', f"TCPIP0::{IP}::INSTR", init_s_params=False)
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

    get_pic(freq=freq, trace=trace, title='IF Reflection', plot_phase=plot_phase, exp_path=exp_path)

    # return {'freq' :freq, 'trace' :trace}

def save_data(pic_path, exp_path):
    if not os.path.exists('{}/data'.format(exp_path)):
         os.mkdir('{}/data'.format(exp_path))
    if os.path.exists(f'{exp_path}/current/data.csv') and pic_path:
        os.replace(f'{exp_path}/current/data.csv', f"{pic_path}")


def get_pic(freq, trace, title, plot_phase, exp_path):

    if plot_phase:
        plt.figure(figsize=(18, 6))
        plt.suptitle(title)

        plt.subplot(121)
        plt.plot(freq, 20*np.log10(np.abs(trace)))
        plt.xlabel('frequency, GHz')
        plt.ylabel('Amp, dB')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()

        plt.subplot(122)
        plt.plot(freq, np.angle(trace))
        plt.xlabel('frequency, GHz')
        plt.ylabel(r'phase, $rad$')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()

    else:
        plt.figure(figsize=(10,6))
        plt.plot(freq, 20*np.log10(np.abs(trace)))
        plt.xlabel('frequency, GHz')
        plt.ylabel('Amp, dB')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()

    if not os.path.exists(f'{exp_path}/current'):
        os.mkdir(f'{exp_path}/current')

    with open(f'{exp_path}/current/data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['freq-Hz', 're-im'])
        for i in range(len(freq)):
            writer.writerow([freq[i], trace[i]])

    plt.show()




