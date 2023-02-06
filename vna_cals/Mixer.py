from datetime import datetime
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from mpmath import besselj
from qmix.respfn import RespFnFromIVData
from qmix.mathfn.misc import slope
from scipy.constants import e, hbar
from scipy.optimize import curve_fit

import logging

logger = logging.getLogger(__name__)
debug = logger.debug


class Measure:

    def __init__(self, meas, cals, cal_impedance, point_num, freq_list, rho=50):
        self.meas = meas
        self.cals = cals
        self.open_z = np.vectorize(cal_impedance.get('open')) or None
        self.short_z = np.vectorize(cal_impedance.get('short')) or None
        self.load_z = np.vectorize(cal_impedance.get('load')) or None
        self.rho = rho

        self.freq_list = freq_list / 1e9

        self.point_num = point_num

    def _get_z(self):
        self.cal_load_z = self.cals.iloc[:,2]
        self.cal_open_z = self.cals.iloc[:,0]
        self.cal_short_z = self.cals.iloc[:,1]

        self.point_z = np.array(self.meas, dtype=np.complex128)

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

        self._get_z()

        G_a_1 = self._gamma_cal(self.load_z(self.freq_list * 1e9))  # Actual match load
        G_a_2 = self._gamma_cal(self.open_z(self.freq_list * 1e9))  # Actual open
        G_a_3 = self._gamma_cal(self.short_z(self.freq_list * 1e9))  # Actual short

        cals = {'D': [], 'S': [], 'R': []}
        for i in range(self.point_num):
            G_m_1 = self.cal_load_z[i]  # Measured match load
            G_m_2 = self.cal_open_z[i]  # Measured open
            G_m_3 = self.cal_short_z[i]  # Measured short

            C = np.array(
                [[G_a_1[i], 1, G_a_1[i] * G_m_1], [G_a_2[i], 1, G_a_2[i] * G_m_2], [G_a_3[i], 1, G_a_3[i] * G_m_3]])
            V = np.array([G_m_1, G_m_2, G_m_3])

            self._Error_Coeffs(self._E_matrix(C, V), cals)

        cals_frame = pd.DataFrame(data=cals, index=self.freq_list.round(4))
        self.point_calibrated = self._Gamma(self.point_z, cals_frame)


    def plot(self, pic_name='SIS_IF_Ref', plot_phase=None, title='SIS IF Reflection', pic_path='', save=False,
             start=None, stop=None):

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

    def plot_cals(self, pic_name='Cals', title='Calibrations', pic_path='', save=False, plot_measured=False,
                  plot_calibrated=False):

        plt.figure(figsize=(10, 6))
        plt.title(title)

        lgnd = ['open', 'short', 'load']
        self._get_z()
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
            plt.savefig(pic_path + pic_name + '.png', dpi=400)
        plt.show()


class Mixer:

    def __init__(self, meas_table, cal_table, V_bias, point_num, LO_rate, Ym=None, offset=(0,0), rho=50):
        self.meas_table = meas_table
        self.cal_table = cal_table
        self.offset = offset
        self.Ym = Ym

        self.IV_curve = dict(
            (
                float(s.split(';')[0]) + self.offset[0], 
                float(s.split(';')[1]) + self.offset[1]
            ) 
            for s in self.cal_table.columns[:-1]
        )

        self.IV_pumped = dict(
            (
                float(s.split(';')[0]) + self.offset[0],
                float(s.split(';')[1]) + self.offset[1]
            )
            for s in self.meas_table.columns[:-1]
        )

        self.LO_rate = LO_rate

        self.I = np.array(list(self.IV_curve.values()))
        self.V = np.array(list(self.IV_curve.keys()))

        self.I_pumped = np.array(list(self.IV_pumped.values()))
        self.V_pumped = np.array(list(self.IV_pumped.keys()))

        self.V_bias = V_bias
        self.rho = rho

        self.resp = RespFnFromIVData(self.Vn, self.In)
        self.point_num = point_num

        self.freq_list = np.array(self.meas_table.pop('freq'), dtype=np.float64)

        self._measures = defaultdict(Measure)
        self.cal_impedance = None

    def calibrate(self):
        for key in self._measures.keys():
            self._measures[key].calibrate()

    @property
    def measures(self):
        return self._measures

    def set_measures(self, recalculate=False):
        if recalculate or not self.cal_impedance:
            self.set_cal_impedance()
        for vi, meas in self.meas_table.items():
            key = f"{float(vi.split(';')[0]) + self.offset[0]};{float(vi.split(';')[1]) + self.offset[1]}"
            self._measures[key] = Measure(meas, self.cals, self.cal_impedance,
                                          point_num=self.point_num, rho=self.rho, freq_list=self.freq_list)

    def remove_measures(self):
        self._measures = defaultdict(Measure)

    @property
    def cals(self):
        open = self.cal_table.filter(like=f"{self.V_bias['open']}").iloc[:,0]
        short = self.cal_table.filter(like=f"{self.V_bias['short']}").iloc[:,0]
        load = self.cal_table.filter(like=f"{self.V_bias['load']}").iloc[:,0]
        return pd.DataFrame((open, short, load), dtype=complex).T

    @property
    def Vn(self):
        return self.V / self.Vgap

    @property
    def In(self):
        return self.I / (self.Igap * 2)

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

    def set_cal_impedance(self):
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
        nu0_range = self.freq_list[::20]
        f_re = lambda x, a1, a2, a3, a4, a5: a1 * x ** 4 + a2 * x ** 3 + a3 * x ** 2 + a4 * x + a5
        f_im = lambda x, a1, a2, a3, a4, a5: a1 * x ** 4 + a2 * x ** 3 + a3 * x ** 2 + a4 * x + a5

        range_time = datetime.now()
        for nu0 in nu0_range:
            iter_time = datetime.now()
            z_open = self.Z(nu=self.LO_rate, nu0=nu0, V0=self.V_bias['open'], al=0.01)
            z_short = self.Z(nu=self.LO_rate, nu0=nu0, V0=self.V_bias['short'], al=0.01)
            z_load = self.Z(nu=self.LO_rate, nu0=nu0, V0=self.V_bias['load'], al=0.01)
            delta = datetime.now() - iter_time
            debug(f'Z calculation time: {delta}')

            res['open']['re'].append(z_open.real)
            res['open']['im'].append(z_open.imag)
            res['short']['re'].append(z_short.real)
            res['short']['im'].append(z_short.imag)
            res['load']['re'].append(z_load.real)
            res['load']['im'].append(z_load.imag)

        delta = datetime.now() - range_time
        debug(f'Z calculation time: {delta}')

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
        start_time = datetime.now()
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
                                (V0 + n1 * hbar * om / e) / self.Vgap)) +
                             (self.resp.idc((V0 + n * hbar * om / e) / self.Vgap) - self.resp.idc(
                                 (V0 + n * hbar * om / e - hbar * omm(m1) / e) / self.Vgap))) * self.Igap * 2
                        )
                g[m + d][m1 + d] *= e / (2 * hbar * om0)

        delta = datetime.now() - start_time
        debug(f'G calculation time: {delta}')
        return g

    def _B(self, nu, nu0, V0, al, lim=10, mrange=[0]):
        """
        :param float nu: FFO rate
        :param float nu0: IF rate
        """
        start_time = datetime.now()
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
                                 (V0 + n * hbar * om / e - hbar * omm(m1) / e) / self.Vgap))) * self.Igap * 2
                        )
                b[m + d][m1 + d] *= e / (2 * hbar * om0)

        delta = datetime.now() - start_time
        debug(f'B calculation time: {delta}')
        return b

    def Z(self, nu, nu0, V0, al, lim=10, mrange=None):
        """
        :param int lim:
        :param list mrange:
        :param float nu: FFO rate
        :param float nu0: IF rate
        :param float V0: V bias
        :param float al: pumping level
        """
        if mrange is None:
            mrange = [-1, 0, 1]

        start_time = datetime.now()
        if self.Ym:
            g = np.array(self._G(nu, nu0, V0, al, lim, mrange))
            b = np.array(self._B(nu, nu0, V0, al, lim, mrange))
            y = g + np.eye(3,3) * self.Ym + b * 1j
            res = np.linalg.inv(y)[1][1]

        else:
            g = np.array(self._G(nu, nu0, V0, al, lim, [0]))
            b = np.array(self._B(nu, nu0, V0, al, lim, [0]))
            y = g[0][0] + b[0][0] * 1j
            res = 1/y

        delta = datetime.now() - start_time
        debug(f'Z calculation time: {delta}')
        return res

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
            res += besselj(n, al) ** 2 * self.resp.idc((V0 + n * hbar * om / e) / self.Vgap) * self.Igap * 2
        return float(res[0])

    def plot_Ip(self, al, nu, V=None):
        """
        :param float al: pumping parameter
        :param float nu: Heterodyne rate
        :param float V: bias voltages range
        """
        if not V:
            V = self.V

        plt.figure(figsize=(10, 6))
        plt.plot(self.V * 1000, self.I * 1000, label='autonomus')
        plt.plot(self.V * 1000, np.vectorize(self.Ip)(V, al=al, nu=nu) * 1000, label='pumped')
        plt.ylabel('I, mA')
        plt.xlabel('V, mV')
        plt.legend()
        plt.grid()
        plt.show()


def mixing(meas_table, cal_table, V_bias, LO_rate, Ym=None, offset=(0,0), point_num=300, rho=50):
    mixer = Mixer(
        meas_table=meas_table,
        cal_table=cal_table,
        V_bias=V_bias,
        point_num=point_num,
        Ym=Ym,
        offset=offset,
        LO_rate=LO_rate,
        rho=rho
    )
    mixer.set_measures()
    mixer.calibrate()
    return mixer
