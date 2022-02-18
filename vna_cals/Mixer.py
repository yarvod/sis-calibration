from collections import defaultdict

import numpy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv

from mpmath import besselj
from qmix.respfn import RespFnFromIVData
from qmix.mathfn.misc import slope
from scipy.constants import e, hbar
from scipy.optimize import curve_fit


def reim(df, num):
    data_z = np.zeros(shape=(num), dtype=np.complex128)
    data_z.real = np.array(df['re'], dtype=np.float64)
    data_z.imag = np.array(df['im'], dtype=np.float64)
    return data_z


def to_db(vec):
    return 20 * np.log10(np.abs(vec))


def get_v(data_dict, freq_point, freq):
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
        self.point_csv_path = meas_path or None

        self.open_z = np.vectorize(resistance.get('open')) or None
        self.short_z = np.vectorize(resistance.get('short')) or None
        self.load_z = np.vectorize(resistance.get('load')) or None
        self.rho = rho

        self.point_num = point_num

    @staticmethod
    def _parse_csv(csv_table_path):
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

    def _get_z(self, calibrate=False):
        self.cal_load_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.cal_load_z.real = np.array(self._parse_csv(self.load_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.cal_load_z.imag = np.array(self._parse_csv(self.load_csv_path).iloc(axis=1)[2], dtype=np.float64)

        self.cal_open_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.cal_open_z.real = np.array(self._parse_csv(self.open_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.cal_open_z.imag = np.array(self._parse_csv(self.open_csv_path).iloc(axis=1)[2], dtype=np.float64)

        self.cal_short_z = np.zeros(shape=(self.point_num), dtype=np.complex128)
        self.cal_short_z.real = np.array(self._parse_csv(self.short_csv_path).iloc(axis=1)[1], dtype=np.float64)
        self.cal_short_z.imag = np.array(self._parse_csv(self.short_csv_path).iloc(axis=1)[2], dtype=np.float64)

        if calibrate:
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

        self._get_z(calibrate=True)

        G_a_1 = self._gamma_cal(self.load_z(self.freq_list*1e9))  # Actual match load
        G_a_2 = self._gamma_cal(self.open_z(self.freq_list*1e9))  # Actual open
        G_a_3 = self._gamma_cal(self.short_z(self.freq_list*1e9))  # Actual short

        cals = {'D': [], 'S': [], 'R': []}
        for i in range(self.point_num):
            G_m_1 = self.cal_load_z[i]  # Measured match load
            G_m_2 = self.cal_open_z[i]  # Measured open
            G_m_3 = self.cal_short_z[i]  # Measured short

            C = np.array([[G_a_1[i], 1, G_a_1[i] * G_m_1], [G_a_2[i], 1, G_a_2[i] * G_m_2], [G_a_3[i], 1, G_a_3[i] * G_m_3]])
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

    def plot_cals(self, pic_name='Cals', title='Calibrations', pic_path='', save=False, plot_measured=False, plot_calibrated=False):

        plt.figure(figsize=(10, 6))
        plt.title(title)

        lgnd = ['open', 'short', 'load']
        self._get_z(plot_calibrated or plot_measured)
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


class Mixer:

    def __init__(self, IV_csv_path, meas_path, cal_path, V_bias, point_num, rho=50):
        self.csv_path = IV_csv_path
        self.table = pd.read_csv(IV_csv_path)
        self.I = np.array(self.table['I'])
        self.V = np.array(self.table['V'])

        self.V_bias = V_bias
        self.rho = rho

        self.resp = RespFnFromIVData(self.Vn, self.In)
        self.freq_list = np.array(Calibration._parse_csv(meas_path).iloc(axis=1)[0], dtype=np.float64)
        self.meas_path = meas_path
        self.cal_path = cal_path
        self.point_num = point_num

        self._calibration = None
        self.cal_impedance = None

    @property
    def calibration(self):
        return self._calibration

    def set_calibration(self, recalculate=False):
        if recalculate or not self.cal_impedance:
            self.resistance
        self._calibration = Calibration(self.meas_path, self.cal_path, self.cal_impedance, point_num=self.point_num,
                                        rho=self.rho)

    @property
    def Vn(self):
        return self.V / self.Vgap

    @property
    def In(self):
        return self.I / self.Igap

    @property
    def Igap(self):
        cond = slope(self.V, self.I)
        ind = np.where(cond == np.max(cond))
        return self.I[ind]

    @property
    def Vgap(self):
        cond = slope(self.V, self.I)
        ind = np.where(cond == np.max(cond))
        return self.V[ind]

    @staticmethod
    def kron(a, b):
        if a == b:
            return 1
        else:
            return 0

    @property
    def resistance(self):
        res = {
            'open': {'re': [], 'im': []},
            'short': {'re': [], 'im': []},
            'load': {'re': [], 'im': []}
        }
        par = {
            'open': {'opt_re': [], 'cov_re': [], 'opt_im': [], 'cov_im': []},
            'short': {'opt_re': [], 'cov_re': [], 'opt_im': [], 'cov_im': []},
            'load': {'opt_re': [], 'cov_re': [], 'opt_im': [], 'cov_im': []}
        }
        nu0_range = self.freq_list[::15]
        f_re = lambda x, a1, a2, a3, a4: a1 * x ** 3 + a2 * x ** 2 + a3 * x + a4
        f_im = lambda x, a1, a2, a3, a4: a1 * x ** 3 + a2 * x ** 2 + a3 * x + a4

        for nu0 in nu0_range:
            z_open = self.Z(nu=600 * 10 ** 9, nu0=nu0, V0=self.V_bias['open'], al=0.01)
            z_short = self.Z(nu=600 * 10 ** 9, nu0=nu0, V0=self.V_bias['short'], al=0.01)
            z_load = self.Z(nu=600 * 10 ** 9, nu0=nu0, V0=self.V_bias['load'], al=0.01)

            res['open']['re'].append(z_open.real)
            res['open']['im'].append(z_open.imag)
            res['short']['re'].append(z_short.real)
            res['short']['im'].append(z_short.imag)
            res['load']['re'].append(z_load.real)
            res['load']['im'].append(z_load.imag)

        for key in par.keys():
            par[key]['opt_re'], par[key]['cov_re'] = curve_fit(f_re, nu0_range, res[key]['re'])
            par[key]['opt_im'], par[key]['cov_im'] = curve_fit(f_im, nu0_range, res[key]['im'])

        imp = {
            'open': lambda x: f_re(x, *par['open']['opt_re']) + f_im(x, *par['open']['opt_im']) * 1j,
            'short': lambda x: f_re(x, *par['short']['opt_re']) + f_im(x, *par['short']['opt_im']) * 1j,
            'load': lambda x: f_re(x, *par['load']['opt_re']) + f_im(x, *par['load']['opt_im']) * 1j
        }
        self.cal_impedance = imp

    def _G(self, nu, nu0, V0, al, lim=10, mrange=[0]):
        """
        :param float nu: FFO rate
        :param float nu0: IF rate
        """
        om = nu * np.pi * 2
        om0 = nu0 * 2 * np.pi
        omm = lambda m: m * om + om0

        g = np.zeros((len(mrange), len(mrange)))
        d = max(mrange)

        for m in mrange:
            for m1 in mrange:
                for n in np.arange(-lim, lim + 1, 1):
                    for n1 in np.arange(-lim, lim + 1, 1):
                        g[m + d][m1 + d] += float(
                            besselj(n, al) * besselj(n1, al) * self.kron(m - m1, n1 - n) * \
                            ((self.resp.idc((V0 + n1 * hbar * om / e + hbar * omm(m1) / e) / self.Vgap) - self.resp.idc(
                                (V0 + n1 * hbar * om / e) / self.Vgap)) + \
                             (self.resp.idc((V0 + n * hbar * om / e) / self.Vgap) - self.resp.idc(
                                (V0 + n * hbar * om / e - hbar * omm(m1) / e) / self.Vgap))) * self.Igap
                        )
                g[m + d][m1 + d] *= e / (2 * hbar * om0)
        return g

    def _B(self, nu, nu0, V0, al, lim=10, mrange=[0]):
        """
        :param float nu: FFO rate
        :param float nu0: IF rate
        """
        om = nu * np.pi * 2
        om0 = nu0 * 2 * np.pi
        omm = lambda m: m * om + om0

        b = np.zeros((len(mrange), len(mrange)))
        d = max(mrange)

        for m in mrange:
            for m1 in mrange:
                i = 0
                for n in np.arange(-lim, lim + 1, 1):
                    for n1 in np.arange(-lim, lim + 1, 1):
                        b[m + d][m1 + d] += float(
                            besselj(n, al) * besselj(n1, al) * self.kron(m - m1, n1 - n) * \
                            ((self.resp.idc((V0 + n1 * hbar * om / e + hbar * omm(m1) / e) / self.Vgap) - self.resp.idc(
                                (V0 + n1 * hbar * om / e) / self.Vgap)) - \
                             (self.resp.idc((V0 + n * hbar * om / e) / self.Vgap) - self.resp.idc(
                                 (V0 + n * hbar * om / e - hbar * omm(m1) / e) / self.Vgap))) * self.Igap
                        )
                b[m + d][m1 + d] *= e / (2 * hbar * om0)
        return b

    def Z(self, nu, nu0, V0, al, lim=10, mrange=[0]):
        """
        :param float nu: FFO rate
        :param float nu0: IF rate
        :param float V0: V bias
        :param float al: pumping level
        """
        g = np.array(self._G(nu, nu0, V0, al, lim, mrange))[max(mrange)][max(mrange)]
        b = np.array(self._B(nu, nu0, V0, al, lim, mrange))[max(mrange)][max(mrange)]
        y = g + b * 1j

        return 1 / y

    def Ip(self, V0, al=0.2, nu=600 * 10 ** 9, lim=10):
        """
        :param float V0: SIS Bias
        :param float al: Pumping level
        :param float om: FFO rate
        :param int lim: summ limit
        """
        res = 0
        om = 2 * np.pi * nu
        for n in np.arange(-lim, lim + 1, 1):
            res += besselj(n, al) ** 2 * self.resp.idc((V0 + n * hbar * om / e) / self.Vgap) * self.Igap
        return res

    def plot_Ip(self, al, nu,  V=None):
        """
        :param float al: pumping parameter
        :param float nu: Heterodyne rate
        :param float V: bias voltages range
        """
        if not V:
            V = self.V

        plt.figure(figsize=(10, 6))
        plt.plot(self.V * 1000, self.I * 1000, label='autonomus')
        plt.plot(self.V * 1000, self.Ip(V, al=al, nu=nu) * 1000, label='pumped')
        plt.ylabel('I, mA')
        plt.xlabel('V, mV')
        plt.legend()
        plt.grid()
        plt.show()
