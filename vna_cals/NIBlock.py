import nidaqmx as ni
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import csv
import time

from vna import get_data
from logger import logger


class NIBlock:

    curr_input_chan = "Dev1/ai3"
    volt_input_chan = "Dev1/ai4"
    mag_curr_input_chan = "Dev1/ai5"
    volt_output_chan = "Dev1/ao2"
    mag_curr_output_chan = "Dev1/ao3"

    def __init__(self) -> None:
        self.iv = defaultdict(list)
        self.refl = defaultdict(list)

    def set_volt(self, volt: float):

        with ni.Task() as rtask, ni.Task() as wtask:

            rtask.ai_channels.add_ai_current_chan(self.curr_input_chan)
            wtask.ao_channels.add_ao_voltage_chan(self.volt_output_chan)

            wtask.write(volt)
            curr = rtask.read(number_of_samples_per_channel=1157)
            
            return np.mean(curr)

    def set_mag_curr(self, curr: float):

        with ni.Task() as rtask, ni.Task() as wtask:

            wtask.ao_channels.add_ao_current_chan(self.mag_curr_output_chan)
            wtask.write(curr)

    def get_mag_curr(self):

        with ni.Task() as rtask:

            rtask.ai_channels.add_ai_current_chan(self.mag_curr_input_chan)

            data = rtask.read(number_of_samples_per_channel=1157)
            return np.mean(data)

    def get_volt(self):

        with ni.Task() as rtask:

            rtask.ai_channels.add_ai_voltage_chan(self.volt_input_chan)

            data = rtask.read(number_of_samples_per_channel=1157)
            return np.mean(data)

    def get_curr(self):

        with ni.Task() as rtask:

            rtask.ai_channels.add_ai_current_chan(self.curr_input_chan)

            data = rtask.read(number_of_samples_per_channel=1157)
            return np.mean(data)

    def measure_iv(self, volt_range: list):
        for volt in volt_range:
            i = self.set_volt(volt)
            v = self.get_volt()
            self.iv['I'].append(i)
            self.iv['V'].append(v)
            self.iv['V_set'].append(volt)
            logger.info(f"i = {i} v = {v}; {ind/len(volt_range) * 100} %")
        return self.iv

    def measure_reflection(
            self,
            volt_range,
            f_from,
            f_to,
            f_points,
            s_par,
            exp_path,
            avg,
            vna_ip,
    ):
        self.refl = defaultdict(list)
        for ind, volt in enumerate(volt_range):
            if ind == 0:
                time.sleep(0.1)
            i = self.set_volt(volt)
            v = self.get_volt()
            self.iv['I'].append(i)
            self.iv['V'].append(v)
            self.iv['V_set'].append(volt)
            res = get_data(
                param=s_par,
                vna_ip=vna_ip,
                plot=False,
                plot_phase=False,
                freq_start=f_from, freq_stop=f_to,
                exp_path=exp_path,
                freq_num=f_points or 201,
                avg=int(avg)
            )
            self.refl[f"{v};{i}"] = res['trace']
            self.refl['freq'] = res['freq']
            logger.info(f"i = {i} v = {v}; {ind/len(volt_range) * 100} %")

        return self.iv, self.refl

    def plot_iv(self, iv=None):
        if not iv:
            iv = self.iv
        plt.figure(figsize=(9,5))
        plt.plot(iv['V'], iv['I'])
        plt.xlabel('voltage, mV')
        plt.ylabel('current, mA')
        plt.grid()
        plt.show()


class NIContainer:

    PARAMS = [2.10505744, 0.01640703]

    def __init__(self):
        self.params = self.PARAMS
        self.volt_range = []
        self.iv = defaultdict(list)
        self.refl = defaultdict(list)

    @property
    def block(self):
        return NIBlock()

    def set_zero(self):
        self.block.set_volt(0)
        delta = self.block.get_volt()
        self.block.set_volt(-delta/self.params[0])

    def update_params(self):
        self.measure_iv(np.linspace(0, 1, 300))
        lin = lambda x, a, b: a*x + b
        opt, cov = curve_fit(lin, self.volt_range, self.iv['V'])
        self.params = opt
        self.block.set_volt(0)

    def measure_iv(self, volt_range):
        initial = self.block.get_volt()
        self.set_zero()
        self.volt_range = volt_range
        self.iv = self.block.measure_iv(volt_range=self.volt_range)
        self.block.set_volt(initial)

    def plot_iv(self):
        self.block.plot_iv(self.iv)

    def write_IV_csv(self, path):
        with open(f'{path}', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['I','V', 'V_set'])
            for i, v, vbias in zip(self.iv['I'], self.iv['V'], self.iv['V_set']):
                writer.writerow([i, v, vbias])

    def measure_reflection(
            self,
            volt_range,
            f_from,
            f_to,
            f_points,
            s_par,
            exp_path,
            avg,
            vna_ip,
    ):
        initial = self.block.get_volt()
        self.set_zero()
        self.volt_range = volt_range
        self.iv, self.refl = self.block.measure_reflection(
            volt_range,
            f_from,
            f_to,
            f_points,
            s_par,
            exp_path,
            avg,
            vna_ip,
        )
        self.block.set_volt(initial / self.params[0] - self.params[1])

    def write_refl_csv(self, path):
        df = pd.DataFrame(self.refl)
        df.to_csv(path, index=False)

        
if __name__ == '__main__':
    volt_range = np.linspace(1, 1.7, 300)
    container = NIContainer()
    # container.update_params()
    container.measure_iv(volt_range)
    container.plot_iv()
    