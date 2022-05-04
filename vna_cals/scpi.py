import socket
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from vna import get_data
from logger import logger
import time
import csv

ST_OK = 'OK'


class Block:

    HOST = '169.254.190.83'
    PORT = 9876

    IV = defaultdict(list)

    @classmethod
    def manipulate(cls, cmd: str) -> str:
        if type(cmd) != bytes:
            cmd = bytes(cmd, 'utf-8')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((cls.HOST, cls.PORT))
            s.sendall(cmd)
            data = s.recv(1024)

        return data.decode().rstrip()

    def get_current(self):
        return self.manipulate('BIAS:DEV2:CURR?')

    def get_voltage(self):
        return self.manipulate('BIAS:DEV2:VOLT?')

    def set_voltage(self, volt: float):
        self.manipulate(f'BIAS:DEV2:VOLT {volt}')

    def measure_IV(self, v_from: float, v_to: float, points: int):
        iv = defaultdict(list)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            
            s.sendall(bytes(f'BIAS:DEV2:VOLT?', 'utf-8'))
            init_v = float(s.recv(1024).decode().rstrip())
            for v in np.linspace(v_from, v_to, points):

                s.sendall(bytes(f'BIAS:DEV2:VOLT {v}', 'utf-8'))
                if v == v_from or (v>2.6e-3 and v<2.9e-3) or (v>-2.9e-3 and v<-2.6e-3):
                    time.sleep(0.1)
                status = s.recv(1024).decode().rstrip()
                if status == 'OK':
                    s.sendall(b'BIAS:DEV2:CURR?')
                    i = s.recv(1024).decode().rstrip()
                    try:
                        i = float(i)
                    except ValueError:
                        continue  # FIXME: try again 
                    iv['I'].append(float(i))
                    iv['V'].append(v)
                    logger.info(f"volt {v}; curr {float(i)}")
            s.sendall(bytes(f'BIAS:DEV2:VOLT {init_v}', 'utf-8'))
        return iv

    def write_IV_csv(self, path, iv):
        with open(f'{path}', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['I','V'])
            for i, v in zip(iv['I'], iv['V']):
                writer.writerow([i,v])

    def write_refl_csv(self, path, refl):
        df = pd.DataFrame(refl)
        df.to_csv(path, index=False)

    def calc_offset(self):
        iv1 = self.measure_IV(v_from=0, v_to=7e-3, points=300)
        iv2 = self.measure_IV(v_from=7e-3, v_to=0, points=300)

    def measure_reflection(self, v_from: float, v_to: float, v_points: int,
                           f_from: float, f_to: float, f_points: int, s_par, exp_path):
        refl = defaultdict(list)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))

            s.sendall(bytes(f'BIAS:DEV2:VOLT?', 'utf-8'))    
            init_v = float(s.recv(1024).decode().rstrip())
            for v in np.linspace(v_from, v_to, v_points):
                s.sendall(bytes(f'BIAS:DEV2:VOLT {v}', 'utf-8'))
                status = s.recv(1024).decode().rstrip()
                if status == 'OK':
                    time.sleep(0.2)
                    s.sendall(b'BIAS:DEV2:CURR?')
                    i = s.recv(1024).decode().rstrip()
                    try:
                        i = float(i)
                    except ValueError:
                        continue  # FIXME: try again

                    res = get_data(
                        param=s_par,
                        plot=False,
                        plot_phase=False,
                        freq_start=f_from, freq_stop=f_to,
                        exp_path=exp_path,
                        freq_num=f_points or 201)

                    refl[f"v={v};i={i}"] = res['trace']

        refl['freq'] = res['freq']
        return refl

    def plot_iv(self, iv):
        plt.figure(figsize=(10, 6))
        plt.scatter(np.array(iv['V'])*1e3, np.array(iv['I'])*1e3, s=2)
        plt.xlabel('SIS Voltage, mV')
        plt.ylabel('SIS Current, m A')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()
        plt.show()


