import socket
from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from vna import get_data
import time
import csv
import logger as mylogger
import logging

from config import BLOCK_IP

logger = logging.getLogger(__name__)

ST_OK = 'OK'


class Block:

    HOST = BLOCK_IP
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
        ats = 10
        iv = defaultdict(list)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            
            init_v = 0
            at=0
            while at<ats:
                try:
                    at += 1
                    time.sleep(0.3)
                    s.sendall(b'BIAS:DEV2:VOLT?')
                    init_v = s.recv(1024)
                    init_v.decode().rstrip()
                    init_v = float(init_v)
                    logger.debug(f'Recieved value: {init_v}; attempt {at}')
                    break
                except Exception as e:
                    logger.warning(f'Exception: {e}; att {at}')
                    continue
            init_time = time.time()
            for v in np.linspace(v_from, v_to, points):

                status = ''
                at=0
                while at<ats:
                    try:
                        at += 1
                        s.sendall(bytes(f'BIAS:DEV2:VOLT {v}', 'utf-8'))
                        status = s.recv(1024)
                        status.decode().rstrip()
                        logger.debug(f'Recieved value: {status}; attempt {at}')
                        break
                    except Exception as e:
                        logger.warning(f'Exception: {e}; att {at}')
                        continue

                a_v = 0
                at=0
                while at<ats:
                    try:
                        at += 1
                        s.sendall(b'BIAS:DEV2:VOLT?')
                        time.sleep(0.1)
                        a_v = s.recv(1024)
                        a_v.decode().rstrip()
                        a_v = float(a_v)
                        logger.debug(f'Recieved value: {a_v}; attempt {at}')
                        break
                    except Exception as e:
                        logger.warning(f'Exception: {e}; att {at}')
                        continue

                i = 0
                at = 0
                while at<ats:
                    try:
                        at += 1
                        s.sendall(b'BIAS:DEV2:CURR?')
                        time.sleep(0.1)
                        i = s.recv(1024)
                        i.decode().rstrip()
                        i = float(i)
                        logger.debug(f'Recieved value: {i}; attempt {at}')
                        break
                    except Exception as e:
                        logger.warning(f'Exception: {e}; att {at}')
                        continue

                delta_time = time.time()-init_time

                iv['I'].append(float(i))
                iv['V'].append(a_v)
                iv['V_bias'].append(v)
                iv['time'].append(delta_time)

                logger.info(f"FINISH; time {delta_time}; volt {a_v}; curr {i}")

            status = ''
            at=0
            while at<ats:
                try:
                    at += 1
                    s.sendall(bytes(f'BIAS:DEV2:VOLT {init_v}', 'utf-8'))
                    status = s.recv(1024)
                    status.decode().rstrip()
                    logger.debug(f'Recieved value: {status}; attempt {at}')
                    break
                except Exception as e:
                    logger.warning(f'Exception: {e}; att {at}')
                    continue
        return iv

    def write_IV_csv(self, path, iv):
        with open(f'{path}', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['I','V', 'V_bias', 'time'])
            for i, v, vbias, t in zip(iv['I'], iv['V'], iv['V_bias'], iv['time']):
                writer.writerow([i, v, vbias, t])

    def write_refl_csv(self, path, refl):
        df = pd.DataFrame(refl)
        df.to_csv(path, index=False)

    def calc_offset(self):
        iv1 = self.measure_IV(v_from=0, v_to=7e-3, points=300)
        iv2 = self.measure_IV(v_from=7e-3, v_to=0, points=300)

    def measure_reflection(self, v_from: float, v_to: float, v_points: int, vna_ip: str,
                           f_from: float, f_to: float, f_points: int, s_par, exp_path, avg: int):
        refl = defaultdict(list)
        ats = 10
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))

            init_v = 0
            at=0
            while at<ats:
                try:
                    at += 1
                    time.sleep(0.3)
                    s.sendall(b'BIAS:DEV2:VOLT?')
                    init_v = s.recv(1024)
                    init_v.decode().rstrip()
                    init_v = float(init_v)
                    logger.debug(f'Recieved value: {init_v}; attempt {at}')
                    break
                except Exception as e:
                    logger.warning(f'Exception: {e}; att {at}')
                    continue
            init_time = time.time()
            for v in np.linspace(v_from, v_to, v_points):

                status = ''
                at=0
                while at<ats:
                    try:
                        at += 1
                        s.sendall(bytes(f'BIAS:DEV2:VOLT {v}', 'utf-8'))
                        status = s.recv(1024)
                        status.decode().rstrip()
                        logger.debug(f'Recieved value: {status}; attempt {at}')
                        break
                    except Exception as e:
                        logger.warning(f'Exception: {e}; att {at}')
                        continue

                a_v = 0
                at=0
                while at<ats:
                    try:
                        at += 1
                        s.sendall(b'BIAS:DEV2:VOLT?')
                        time.sleep(0.1)
                        a_v = s.recv(1024)
                        a_v.decode().rstrip()
                        a_v = float(a_v)
                        logger.debug(f'Recieved value: {a_v}; attempt {at}')
                        break
                    except Exception as e:
                        logger.warning(f'Exception: {e}; att {at}')
                        continue

                i = 0
                at = 0
                while at<ats:
                    try:
                        at += 1
                        s.sendall(b'BIAS:DEV2:CURR?')
                        time.sleep(0.1)
                        i = s.recv(1024)
                        i.decode().rstrip()
                        i = float(i)
                        logger.debug(f'Recieved value: {i}; attempt {at}')
                        break
                    except Exception as e:
                        logger.warning(f'Exception: {e}; att {at}')
                        continue

                delta_time = time.time()-init_time
                res = {}
                at = 0
                while at<ats:
                    try:
                        at += 1
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
                        break
                    except:
                        continue
                refl[f"{a_v};{i}"] = res['trace']
                logger.info(f"FINISH; time {delta_time}; volt {a_v}; curr {i}")

            status = ''
            at=0
            while at<ats:
                try:
                    at += 1
                    s.sendall(bytes(f'BIAS:DEV2:VOLT {init_v}', 'utf-8'))
                    status = s.recv(1024)
                    status.decode().rstrip()
                    logger.debug(f'Recieved value: {status}; attempt {at}')
                    break
                except Exception as e:
                    logger.warning(f'Exception: {e}; att {at}')
                    continue

            s.sendall(bytes(f'BIAS:DEV2:VOLT {init_v}', 'utf-8'))

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


