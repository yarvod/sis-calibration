import matplotlib.pyplot as plt
import numpy as np
import csv, os
import qcodes.instrument_drivers.rohde_schwarz.ZNB as ZNB
import qcodes


def get_data(param, exp_path, freq_start=3.5e9, freq_stop=8.5e9, freq_num=201, vna_power=-30, aver=False, num=None,
             mov_aver=False, span=None, plot=False, plot_phase=False, save=True):
    title = 'IF Reflection'
    plot_phase = plot_phase
    exp_path = exp_path
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

    if aver:
        trace = av(num, vna)
        db = 20 * np.log10(np.abs(trace))
    if mov_aver:
        trace = vna.channels.trace.get()[0]
        db = 20 * np.log10(np.abs(mov_av(list(trace), span)))
        freq = freq[span-1:]
    else:
        trace = vna.channels.trace.get()[0]
        db = 20 * np.log10(np.abs(trace))

    if plot and plot_phase:
        plt.figure(figsize=(18, 6))
        plt.suptitle(title)

        plt.subplot(121)
        plt.plot(freq, db)
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
        plt.show()

    elif plot:
        plt.figure(figsize=(10, 6))
        plt.plot(freq, db)
        plt.xlabel('frequency, GHz')
        plt.ylabel('Amp, dB')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()
        plt.show()

    if not os.path.exists(f'{exp_path}/current'):
        os.mkdir(f'{exp_path}/current')

    if save:
        with open(f'{exp_path}/current/data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['freq-Hz', 're', 'im'])
            for i in range(len(freq)):
                writer.writerow([freq[i], trace[i].real, trace[i].imag])

    return {'freq': freq, 'trace': trace}

def save_data(pic_path, exp_path):
    if os.path.exists(f'{exp_path}/current/data.csv') and pic_path:
        os.replace(f'{exp_path}/current/data.csv', f"{pic_path}")

def save_calibrated_data(data_path, exp_path):
    if os.path.exists(f'{exp_path}/current/calibrated_data.csv') and data_path:
        os.replace(f'{exp_path}/current/calibrated_data.csv', f"{data_path}")

def av(num, vna):
    all_data = []
    for i in range(num):
        all_data.append(list(vna.channels.trace.get()[0]))
    all_data = np.array(all_data)
    return np.mean(all_data, axis=0)


def mov_av(a, n=3):
    ret = np.cumsum(a, dtype=np.complex)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
