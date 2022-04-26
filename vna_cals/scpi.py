import socket
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from vna import get_data

ST_OK = 'OK'


class Block:

    HOST = '169.254.190.83'
    PORT = 9876

    IV = defaultdict(list)

    @classmethod
    def manipulate(cls, cmd: str) -> str:
        if type(cmd) != bytes:
            cmd = bytes(cmd, 'utf-8')
        with socket.socket(socket.AF_UNIX, socket.socket.SOCK_STREAM) as s:
            s.connect((cls.HOST, cls.PORT))
            s.sendall(cmd)
            data = s.recv(1024)

        return data.decode().rstrip()

    def get_current(self):
        self.manipulate('BIAS:DEV2:CURR?')

    def get_voltage(self):
        self.manipulate('BIAS:DEV2:VOLT?')

    def set_voltage(self, volt: float):
        self.manipulate(f'BIAS:DEV2:VOLT {volt}')

    def measure_IV(self, v_from: float, v_to: float, points: int) -> defaultdict[list]:
        iv = defaultdict(list)
        with socket.socket(socket.AF_UNIX, socket.socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            for v in np.linspace(v_from, v_to, points):
                s.sendall(bytes(f'BIAS:DEV2:VOLT {v}', 'utf-8'))
                status = s.recv(1024).decode().rstrip()
                if status == 'OK':
                    s.sendall(b'BIAS:DEV2:CURR?')
                    i = s.recv(1024).decode().rstrip()
                    iv['I'].append(float(i))
                    iv['V'].append(v)
        return iv

    def write_IV_csv(self, path):
        with open(f'{path}', 'w') as f:
            f.write('I,V')
            for i, v in zip((self.IV['I'], self.IV['V'])):
                f.write(f"{i},{v}")

    def write_refl_csv(self, path):
        with open(f'{path}', 'w') as f:
            f.write(['freq'])
            for i, v in zip((self.IV['I'], self.IV['V'])):
                f.write(f"{i},{v}")

    def calc_offset(self):
        iv1 = self.measure_IV(v_from=0, v_to=7e-3, points=300)
        iv2 = self.measure_IV(v_from=7e-3, v_to=0, points=300)

    def measure_reflection(self, v_from: float, v_to: float, v_points: int,
                           f_from: float, f_to: float, f_points: int, s_par, exp_path):
        refl = defaultdict(list)
        with socket.socket(socket.AF_UNIX, socket.socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            for v in np.linspace(v_from, v_to, v_points):
                s.sendall(bytes(f'BIAS:DEV2:VOLT {v}', 'utf-8'))
                status = s.recv(1024).decode().rstrip()
                if status == 'OK':
                    res = get_data(
                        param=s_par,
                        plot=False,
                        plot_phase=False,
                        freq_start=f_from, freq_stop=f_to,
                        exp_path=exp_path,
                        freq_num=f_points or 201)
                    refl[v] = res['trace']
                    if not refl['freq']:
                        refl['freq'] = res['freq']

        return refl

    def plot_iv(self, iv):
        plt.figure(figsize=(10, 6))
        plt.plot(iv['V']*1000, iv['I']*1000)
        plt.xlabel('SIS Voltage, mV')
        plt.ylabel(r'SIS Current, $\mu A$')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()


