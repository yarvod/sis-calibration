import nidaqmx as ni
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class NIBlock:

    curr_input_chan = "Dev1/ai3"
    volt_input_chan = "Dev1/ai4"
    volt_output_chan = "Dev1/ao2"

    def __init__(self) -> None:
        self.iv = defaultdict(list)

    def set_volt(self, volt: float):

        with ni.Task() as rtask, ni.Task() as wtask:

            rtask.ai_channels.add_ai_current_chan(self.curr_input_chan)
            wtask.ao_channels.add_ao_voltage_chan(self.volt_output_chan)

            wtask.write(volt)
            curr = rtask.read(number_of_samples_per_channel=157)
            
            return np.mean(curr)

    def get_volt(self):

        with ni.Task() as rtask:

            rtask.ai_channels.add_ai_voltage_chan(self.volt_input_chan)

            data = rtask.read(number_of_samples_per_channel=157)
            return np.mean(data)

    def get_curr(self):

        with ni.Task() as rtask:

            rtask.ai_channels.add_ai_current_chan(self.curr_input_chan)

            data = rtask.read(number_of_samples_per_channel=157)
            return np.mean(data)

    def measure_iv(self, volt_range: list):
    
        for volt in volt_range:
            i = self.set_volt(volt)
            v = self.get_volt()
            self.iv['I'].append(i)
            self.iv['V'].append(v)
            self.iv['V_set'].append(volt)
        return self.iv

    def plot_iv(self, iv=None):
        if not iv:
            iv = self.iv
        plt.figure(figsize=(9,5))
        plt.plot(iv['V'], iv['I'])
        plt.xlabel('voltage, mV')
        plt.ylabel('current, mkA')
        plt.grid()
        plt.show()


class NIContainer:

    PARAMS = [2.10505744, 0.01640703]

    def __init__(self):
        self.params = self.PARAMS
        self.volt_range = []
        self.iv = []

    @property
    def block(self):
        return NIBlock()

    def update_params(self):
        self.measure_iv(np.linspace(0, 1, 500))
        lin = lambda x, a, b: a*x + b
        opt, cov = curve_fit(lin, self.volt_range, self.iv['V'])
        self.params = opt

    def measure_iv(self, volt_range):
        self.volt_range = volt_range
        self.iv = self.block.measure_iv(volt_range=self.volt_range)

        
if __name__ == '__main__':
    volt_range = np.linspace(0, 3, 10)
    block = NIBlock()

    block.measure_iv(volt_range)
    block.plot_iv()
    