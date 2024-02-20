from datetime import datetime
from typing import List, Union, Dict, Tuple, Callable

import numpy as np

from mpmath import besselj
from .respfn import RespFnFromIVData
from scipy.constants import e, hbar
from scipy.optimize import curve_fit

import logging

from .utils import get_gap

logger = logging.getLogger(__name__)
debug = logger.debug


class Measure:
    def __init__(
        self,
        reflection: List[complex],
        cals: Dict[str, List],
        cal_impedance: Dict[str, Callable],
        voltage: float,
        current: float,
        if_list: List[float],
        rho: float = 50
    ):
        """

        :param list of complex reflection:
        :param dict cals: Reflection in calibration points {"open": [1 + 1j, ...], "short": ..., "load": ...}
        :param dict of Callable cal_impedance: {"open": func(if), "short": ..., "load": ...} func calculate impedance
        :param float voltage: bias voltage,
        :param float current: bias current,
        :param if_list: List of IFs
        :param rho:
        """
        self.reflection = reflection
        self.calibrated_reflection = None
        self.cals = cals
        self.open_z = np.vectorize(cal_impedance.get("open")) or None
        self.short_z = np.vectorize(cal_impedance.get("short")) or None
        self.load_z = np.vectorize(cal_impedance.get("load")) or None

        self.voltage = voltage
        self.current = current

        self.if_list = if_list
        self.point_num = len(if_list)
        self.rho = rho

    def __str__(self):
        return f"{self.__class__.__name__}(V={self.voltage}, I={self.current}, rho={self.rho})"

    __repr__ = __str__

    def _set_z(self):
        self.cal_load_z = np.array(self.cals["load"], dtype=np.complex128)
        self.cal_open_z = np.array(self.cals["open"], dtype=np.complex128)
        self.cal_short_z = np.array(self.cals["short"], dtype=np.complex128)
        self.reflection_z = np.array(self.reflection, dtype=np.complex128)

    @staticmethod
    def _error_matrix(C, V):
        C_H = np.matrix.getH(C)
        inv_CC_H = np.linalg.inv(np.dot(C_H, C))
        result = np.dot(np.dot(inv_CC_H, C_H), V.T)
        return result

    @staticmethod
    def _append_cals_coefs(vector, cals):
        cals["D"].append(vector[1])
        cals["S"].append(vector[2])
        cals["R"].append(vector[0] + vector[1] * vector[2])

    @staticmethod
    def calibrate_reflection(raw_reflection, cal_frame):
        gamma = (raw_reflection - cal_frame["D"]) / (
                cal_frame["R"] + cal_frame["S"] * raw_reflection - cal_frame["S"] * cal_frame["D"]
        )
        return gamma

    def impedance_to_reflection(self, z):
        return (z - self.rho) / (z + self.rho)

    def calibrate(self):
        self._set_z()

        G_a_1 = self.impedance_to_reflection(self.load_z(self.if_list))  # Actual match load
        G_a_2 = self.impedance_to_reflection(self.open_z(self.if_list))  # Actual open
        G_a_3 = self.impedance_to_reflection(self.short_z(self.if_list))  # Actual short

        cals = {"D": [], "S": [], "R": []}
        for i in range(self.point_num):
            G_m_1 = self.cal_load_z[i]  # Measured match load
            G_m_2 = self.cal_open_z[i]  # Measured open
            G_m_3 = self.cal_short_z[i]  # Measured short

            C = np.array(
                [
                    [G_a_1[i], 1, G_a_1[i] * G_m_1],
                    [G_a_2[i], 1, G_a_2[i] * G_m_2],
                    [G_a_3[i], 1, G_a_3[i] * G_m_3],
                ]
            )
            V = np.array([G_m_1, G_m_2, G_m_3])

            self._append_cals_coefs(self._error_matrix(C, V), cals)

        self.calibrated_reflection = self.calibrate_reflection(self.reflection_z, cals)


class MeasureList(list):
    def first(self) -> Union["Measure", None]:
        try:
            return self[0]
        except IndexError:
            return None

    def last(self) -> Union["Measure", None]:
        try:
            return self[-1]
        except IndexError:
            return None

    def _filter(self, **kwargs) -> filter:
        def _filter(item):
            for key, value in kwargs.items():
                if not getattr(item, key, None) == value:
                    return False
            return True

        return filter(_filter, self)

    def filter(self, **kwargs) -> "MeasureList":
        return self.__class__(self._filter(**kwargs))

    def get_by_voltage(self, voltage: float) -> "Measure":
        get_voltage = np.array([i.voltage for i in self])
        diff = np.abs(get_voltage - voltage)
        ind = np.where(diff == np.min(diff))[0][0]  # TODO: add exception catcher
        return self[ind]

    def delete_by_index(self, index: int) -> None:
        del self[index]


class CalibrationList(list):
    def get_by_voltage(self, voltage: float) -> Dict:
        get_voltage = np.array([i["voltage"] for i in self])
        diff = np.abs(get_voltage - voltage)
        ind = np.where(diff == np.min(diff))[0][0]
        return self[ind]


class Mixer:
    """
    SIS mixer implementation

    How to use
    ----------
    >>> mixer = Mixer(...)
    >>> mixer.set_measures()
    >>> mixer.calibrate()

    If you want to set custom calibration impedance before call ``mixer.set_measures()`` you should set:
    ``mixer.cal_impedance = {"open": func(if), "short": ..., "load": ...}`` Then:

    >>> mixer.set_measures(calculate_cal_impedance=False)
    >>> mixer.calibrate()

    As a result you can get calibrated measures data:

    >>> mixer.measures  # Returns list of measures MeasureList
    >>> measure = mixer.measures.get_by_voltage(voltage=0.002)  # Returns closable Measure to voltage 0.002 V
    >>> measure.calibrated_reflection  # Returns list of calibrated complex reflection. E.g. [1 + 2j, ...]
    """
    def __init__(
        self,
        measure: List[Dict],
        calibration: List[Dict],
        v_bias: Dict[str, float],
        if_list: List[float],
        lo_rate: float,
        ym: float = None,
        offset: Tuple[float] = (0, 0),
        rho: float = 50,
        gap_params: Dict[str, float] = None,
        i_gap: float = None,
        v_gap: float = None,
    ):
        """
        :param list of dict measure: [{"voltage": 0.02, "current": 0.001, "reflection": [1 + 1j, ...]}, ...]
        :param list of dict calibration: [{"voltage": 0.02, "current": 0.001, "reflection": [1 + 1j, ...]}, ...]
        :param dict v_bias: dict of used bias voltages [V] for Open, Short and Load cal. {"open": 0.1, "short": 0.2. "load": 0.3}
        :param list of float if_list: IF frequency [Hz] values list
        :param float lo_rate: Local oscillator frequency [Hz]
        :param list of float ym: High frequency admittance vector [Ohm]
        :param tuple of float offset: offset by voltage [V] and current [A]
        :param float rho: Impedance of Y_l [Ohm]
        :param dict gap_params: parameters of gap searching
        """
        self.measure_data = measure
        self.calibration_data = CalibrationList(calibration)
        self.offset = offset
        self.ym = ym

        self.iv_curve = {
            s["voltage"] + self.offset[0]: s["current"] + self.offset[1]
            for s in self.calibration_data
        }

        self.i = np.array(list(self.iv_curve.values()))
        self.v = np.array(list(self.iv_curve.keys()))

        self.iv_pumped = self.iv_curve = {
            s["voltage"] + self.offset[0]: s["current"] + self.offset[1]
            for s in self.measure_data
        }

        self.i_pumped = np.array(list(self.iv_pumped.values()))
        self.v_pumped = np.array(list(self.iv_pumped.keys()))

        self.lo_rate = lo_rate

        self.v_gap, self.i_gap = v_gap, i_gap
        if gap_params is None:
            gap_params = {
                "voltage_gap_start": 0.0022,
                "voltage_gap_end": 0.003,
                "voltage_rn_start": 0.004,
                "sgf_window": 50,
                "sgf_degree": 5,
            }
        # calculate gap if v_gap, i_gap not passed
        if not self.v_gap and not self.i_gap:
            self.v_gap, self.i_gap = get_gap(self.v, self.i, **gap_params)

        self.v_bias = v_bias
        self.rho = rho

        self.resp = RespFnFromIVData(self.Vn, self.In)

        self.if_list = if_list
        self.point_num = len(if_list)

        self._measures: MeasureList["Measure"] = MeasureList()
        self.cal_impedance = None

    def calibrate(self):
        for measure in self._measures:
            measure.calibrate()

    @property
    def measures(self):
        return self._measures

    def set_measures(self, calculate_cal_impedance: bool = True):
        if calculate_cal_impedance:
            self.cal_impedance = self.calculate_cal_impedance()
        assert self.cal_impedance is not None, "Calibration impedance can't be None"
        for meas in self.measure_data:
            measure = Measure(
                reflection=meas["reflection"],
                cals=self.cals,
                cal_impedance=self.cal_impedance,
                voltage=meas["voltage"],
                current=meas["current"],
                if_list=self.if_list,
                rho=self.rho,
            )
            self._measures.append(measure)

    def remove_measures(self):
        self._measures = MeasureList()

    @property
    def cals(self):
        return {
            "open": self.calibration_data.get_by_voltage(self.v_bias['open'])['reflection'],
            "short": self.calibration_data.get_by_voltage(self.v_bias['short']),
            "load": self.calibration_data.get_by_voltage(self.v_bias['load']),
        }

    @property
    def Vn(self):
        return self.v / self.v_gap

    @property
    def In(self):
        return self.i / self.i_gap

    @staticmethod
    def kron(a, b):
        return 1 if a == b else 0

    def calculate_cal_impedance(self) -> Dict[str, Callable]:
        """
        This method calculate and fit (poly 4) SIS mixer impedance for calibration points in self.v_bias

        Returns: Dict of functions

        """
        res = {
            "open": {"re": [], "im": []},
            "short": {"re": [], "im": []},
            "load": {"re": [], "im": []},
        }
        par = {
            "open": {"opt_re": [], "cov_re": [], "opt_im": [], "cov_im": []},
            "short": {"opt_re": [], "cov_re": [], "opt_im": [], "cov_im": []},
            "load": {"opt_re": [], "cov_re": [], "opt_im": [], "cov_im": []},
        }
        nu0_range = self.if_list[::20]
        f_re = (
            lambda x, a1, a2, a3, a4, a5: a1 * x**4
            + a2 * x**3
            + a3 * x**2
            + a4 * x
            + a5
        )
        f_im = (
            lambda x, a1, a2, a3, a4, a5: a1 * x**4
            + a2 * x**3
            + a3 * x**2
            + a4 * x
            + a5
        )

        range_time = datetime.now()
        for ind, nu0 in enumerate(nu0_range):
            iter_time = datetime.now()
            z_open = self.Z(nu=self.lo_rate, nu0=nu0, V0=self.v_bias["open"], al=0.001)
            z_short = self.Z(nu=self.lo_rate, nu0=nu0, V0=self.v_bias["short"], al=0.001)
            z_load = self.Z(nu=self.lo_rate, nu0=nu0, V0=self.v_bias["load"], al=0.001)
            delta = datetime.now() - iter_time
            debug(f"[calculate_cal_impedance][{ind}/{len(nu0_range)}]Z calculation time: {delta}")

            res["open"]["re"].append(z_open.real)
            res["open"]["im"].append(z_open.imag)
            res["short"]["re"].append(z_short.real)
            res["short"]["im"].append(z_short.imag)
            res["load"]["re"].append(z_load.real)
            res["load"]["im"].append(z_load.imag)

        delta = datetime.now() - range_time
        debug(f"[calculate_cal_impedance] Finish. Z calculation time: {delta}")

        for key in par.keys():
            par[key]["opt_re"], par[key]["cov_re"] = curve_fit(
                f_re, nu0_range, res[key]["re"]
            )
            par[key]["opt_im"], par[key]["cov_im"] = curve_fit(
                f_im, nu0_range, res[key]["im"]
            )

        imp = {
            "open": lambda x: f_re(x, *par["open"]["opt_re"])
            + f_im(x, *par["open"]["opt_im"]) * 1j,
            "short": lambda x: f_re(x, *par["short"]["opt_re"])
            + f_im(x, *par["short"]["opt_im"]) * 1j,
            "load": lambda x: f_re(x, *par["load"]["opt_re"])
            + f_im(x, *par["load"]["opt_im"]) * 1j,
        }
        return imp

    def _G(self, nu, nu0, V0, al, lim=10, mrange=(-1, 0, 1)):
        """
        :param float nu: LO rate
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
                            besselj(n, al)
                            * besselj(n1, al)
                            * self.kron(m - m1, n1 - n)
                            * (
                                (
                                    self.resp.idc(
                                        (V0 + n1 * hbar * om / e + hbar * omm(m1) / e)
                                        / self.v_gap
                                    )
                                    - self.resp.idc(
                                        (V0 + n1 * hbar * om / e) / self.v_gap
                                    )
                                )
                                + (
                                    self.resp.idc((V0 + n * hbar * om / e) / self.v_gap)
                                    - self.resp.idc(
                                        (V0 + n * hbar * om / e - hbar * omm(m1) / e)
                                        / self.v_gap
                                    )
                                )
                            )
                            * self.i_gap
                        )
                g[m + d][m1 + d] *= e / (2 * hbar * omm(m1))

        delta = datetime.now() - start_time
        debug(f"G calculation time: {delta}")
        return g

    def _B(self, nu, nu0, V0, al, lim=10, mrange=(-1, 0, 1)):
        """
        :param float nu: LO rate
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
                            besselj(n, al)
                            * besselj(n1, al)
                            * self.kron(m - m1, n1 - n)
                            * (
                                (
                                    self.resp.ikk(
                                        (V0 + n1 * hbar * om / e + hbar * omm(m1) / e)
                                        / self.v_gap
                                    )
                                    - self.resp.ikk(
                                        (V0 + n1 * hbar * om / e) / self.v_gap
                                    )
                                )
                                - (
                                    self.resp.ikk((V0 + n * hbar * om / e) / self.v_gap)
                                    - self.resp.ikk(
                                        (V0 + n * hbar * om / e - hbar * omm(m1) / e)
                                        / self.v_gap
                                    )
                                )
                            )
                            * self.i_gap
                        )
                b[m + d][m1 + d] *= e / (2 * hbar * omm(m1))

        delta = datetime.now() - start_time
        debug(f"B calculation time: {delta}")
        return b

    def Z(self, nu, nu0, V0, al, lim=10, mrange=None):
        """
        :param int lim:
        :param list mrange:
        :param float nu: LO rate
        :param float nu0: IF rate
        :param float V0: V bias
        :param float al: pumping level
        """
        if mrange is None:
            mrange = [-1, 0, 1]

        start_time = datetime.now()
        if self.ym:
            g = np.array(self._G(nu, nu0, V0, al, lim, mrange))
            b = np.array(self._B(nu, nu0, V0, al, lim, mrange))
            y = g + np.eye(3, 3) * self.ym + b * 1j
            res = np.linalg.inv(y)[1][1]

        else:
            g = np.array(self._G(nu, nu0, V0, al, lim, [0]))
            b = np.array(self._B(nu, nu0, V0, al, lim, [0]))
            y = g[0][0] + b[0][0] * 1j
            res = 1 / y

        delta = datetime.now() - start_time
        debug(f"Z calculation time: {delta}")
        return res

    def Ip(self, V0, al=0.2, nu=600 * 10**9, lim=10):
        """
        :param float V0: SIS Bias
        :param float al: Pumping level
        :param float om: LO rate
        :param int lim: summ limit
        """
        res = 0
        om = 2 * np.pi * nu
        for n in np.arange(-lim, lim + 1, 1):
            res += (
                besselj(n, al) ** 2
                * self.resp.idc((V0 + n * hbar * om / e) / self.v_gap)
                * self.i_gap
            )
        return float(res)
