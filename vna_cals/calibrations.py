from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv


def reim(df):
    data_z = np.zeros(shape=(NUM), dtype=np.complex128)
    data_z.real = np.array(df['re'], dtype=np.float64)
    data_z.imag = np.array(df['im'], dtype=np.float64)
    return data_z


def to_db(vec):
    return 20 * np.log10(np.abs(vec))


def get_v(data_dict, freq_point):
    bias_v = data_dict.keys()
    db = []
    freq_ind = np.where(freq>freq_point)[0][0]
    for key in data_dict:
        db.append(data_dict[key][freq_ind])
    return db


def moving_average(a, n=3):
    """
    :param list a: vector to average
    :param int n: points window
    :return: averaged vector
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def deriv(x, y):
    """
    :param list x:
    :param list y:
    :return: derivative vector
    """
    der = []
    for i in range(len(y)-1):
        der.append((x[i+1]-x[i])/(y[i+1]-y[i]))

    der = np.array(der)
    return 1/der


def to_db2(dwl):
    for key in dwl:
        dwl[key] = to_db(dwl[key])
    return dwl


def calibrate(meas_path, cal_path, resistance, point_num, rho=50):
    """
    :param list of str meas_path: list of global paths to measurements (csv)
    :param dict cal_path: global paths to 3 calibrations
    :param dict resistance: open, short, load impedance
    :param int point_num:
    :param float rho: Impedance of supply line
    :return: calibrated data
    """
    calibrated_data = defaultdict(list)
    for meas in meas_path:
        key = meas.split('/')[-1].split('.csv')[0]
        cal = Calibration(
            meas_path=meas,
            cal_path=cal_path,
            resistance=resistance,
            point_num=point_num,
            rho=rho
        )
        cal.calibrate()
        calibrated_data[key] = np.array(cal.point_calibrated)
    return calibrated_data


class Calibration:

    def __init__(self, meas_path, cal_path, resistance, point_num, rho=50):
        self.open_csv_path = cal_path.get('open')
        self.short_csv_path = cal_path.get('short')
        self.load_csv_path = cal_path.get('load')
        self.point_csv_path = meas_path

        self.open_r = resistance.get('open')
        self.short_r = resistance.get('short')
        self.load_r = resistance.get('load')
        self.rho = rho

        self.point_num = point_num

    def _parse_csv(self, csv_table_path):
        with open(csv_table_path, 'r') as csv_file:
            table = csv.reader(csv_file)
            table_list = []
            for row in table:
                if '#' not in row[0]:
                    if '' in row:
                        row.remove('')
                    table_list.append(row)
            df_data = pd.DataFrame(table_list[1:], columns=table_list[0])
            return df_data

    def _get_z(self):
        self.cal_load_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.cal_load_z.real = np.array(self._parse_csv(self.load_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.cal_load_z.imag = np.array(self._parse_csv(self.load_csv_path).iloc(axis=1)[2], dtype=np.float64)

        self.cal_open_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.cal_open_z.real = np.array(self._parse_csv(self.open_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.cal_open_z.imag = np.array(self._parse_csv(self.open_csv_path).iloc(axis=1)[2], dtype=np.float64)

        self.cal_short_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.cal_short_z.real = np.array(self._parse_csv(self.short_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.cal_short_z.imag = np.array(self._parse_csv(self.short_csv_path).iloc(axis=1)[2], dtype=np.float64)

        self.point_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.point_z.real = np.array(self._parse_csv(self.point_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.point_z.imag = np.array(self._parse_csv(self.point_csv_path).iloc(axis=1)[2], dtype=np.float64)

        self.freq_list = np.array(self._parse_csv(self.point_csv_path).iloc(axis=1)[0], dtype=np.float64) / 1000000000

    @staticmethod
    def _E_matrix(C, V):
        C_H = np.matrix.getH(C)
        inv_CC_H = np.linalg.inv(np.dot(C_H, C))
        result = np.dot(np.dot(inv_CC_H, C_H), V.T)
        return result

    @staticmethod
    def _Error_Coeffs(vector, cals):
        cals['D'].append(vector[1])
        cals['S'].append(vector[2])
        cals['R'].append(vector[0] + vector[1] * vector[2])

    @staticmethod
    def _Gamma(att, cal_frame):
        gamma = (att - cal_frame['D']) / (cal_frame['R'] + cal_frame['S'] * att - cal_frame['S'] * cal_frame['D'])
        return gamma

    def _gamma_cal(self, Z_n):
        return (Z_n - self.rho) / (Z_n + self.rho)

    def calibrate(self):
        G_a_1 = self._gamma_cal(self.load_r)  # Actual match load
        G_a_2 = self._gamma_cal(self.open_r)  # Actual open
        G_a_3 = self._gamma_cal(self.short_r)  # Actual short

        self._get_z()

        cals = {'D': [], 'S': [], 'R': []}
        for i in range(self.point_num):
            G_m_1 = self.cal_load_z[i]  # Measured match load
            G_m_2 = self.cal_open_z[i]  # Measured open
            G_m_3 = self.cal_short_z[i]  # Measured short

            C = np.array([[G_a_1, 1, G_a_1 * G_m_1], [G_a_2, 1, G_a_2 * G_m_2], [G_a_3, 1, G_a_3 * G_m_3]])
            V = np.array([G_m_1, G_m_2, G_m_3])

            self._Error_Coeffs(self._E_matrix(C, V), cals)

        cals_frame = pd.DataFrame(cals)
        self.point_calibrated = self._Gamma(self.point_z, cals_frame)


    def plot(self, pic_name='SIS_IF_Ref', plot_phase = None, title='SIS IF Reflection', pic_path = '', save = False, start=None, stop=None):

        if plot_phase:
            plt.figure(figsize=(19, 6))
            plt.suptitle(title)

            plt.subplot(121)
            plt.plot(self.freq_list[start:stop], 20 * np.log10(np.abs(self.point_calibrated))[start:stop])
            plt.xlabel('frequency, GHz')
            plt.ylabel('Amp, dB')
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':')
            plt.grid()

            plt.subplot(122)
            plt.plot(self.freq_list[start:stop], np.angle(self.point_calibrated)[start:stop])
            plt.xlabel('frequency, GHz')
            plt.ylabel(r'phase, $rad$')
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':')
            plt.grid()
        if not plot_phase:
            plt.figure(figsize=(10, 6))
            plt.title(title)

            plt.plot(self.freq_list[start:stop], 20 * np.log10(np.abs(self.point_calibrated))[start:stop])
            plt.xlabel('frequency, GHz')
            plt.ylabel('Amp, dB')
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':')
            plt.grid()

        if save:
            plt.savefig(pic_path + pic_name + '.pdf', dpi=400)
        plt.show()

    def plot_cals(self, pic_name='Cals', title='Calibrations', pic_path = '', save = False, plot_measured = False, plot_calibrated = False):

        plt.figure(figsize=(10, 6))
        plt.title(title)

        lgnd = ['open', 'short', 'load']

        plt.plot(self.freq_list, 20 * np.log10(np.abs(self.cal_open_z)))
        plt.plot(self.freq_list, 20 * np.log10(np.abs(self.cal_short_z)))
        plt.plot(self.freq_list, 20 * np.log10(np.abs(self.cal_load_z)))

        if plot_calibrated:
            plt.plot(self.freq_list, 20 * np.log10(np.abs(self.point_calibrated)))
            lgnd.append('point_calibrated')
        if plot_measured:
            plt.plot(self.freq_list, 20 * np.log10(np.abs(self.point_z)))
            lgnd.append('point_measured')

        plt.legend(lgnd)

        plt.xlabel('frequency, GHz')
        plt.ylabel('Amp, dB')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':')
        plt.grid()

        if save:
            plt.savefig(pic_path + pic_name + '.pdf', dpi=400)
        plt.show()


