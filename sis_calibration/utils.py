import re
from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy.interpolate import splrep, splev, BSpline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt


def reim(df, num=None):
    if not num:
        num = len(df)
    data_z = np.zeros(shape=(num), dtype=np.complex128)
    data_z.real = np.array(df["re"], dtype=np.float64)
    data_z.imag = np.array(df["im"], dtype=np.float64)
    return data_z


def to_db(vec):
    return 20 * np.log10(np.abs(vec))


def moving_average(a, n=3):
    """
    :param list a: vector to average
    :param int n: points window
    :return: averaged vector
    """
    a = np.array(a)
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def power_aver(vec, n):
    return 10 * np.log10(moving_average((np.abs(vec)) ** 2, n))


def calc_offset(V, I):
    I = np.array(list(I))
    V = np.array(list(V))

    slp = slope(V, I)
    nans_inds = np.isnan(slp)
    infinit_inds = ~np.isfinite(slp)
    bad_inds = nans_inds & infinit_inds
    I = I[~bad_inds]
    V = V[~bad_inds]
    slp = slp[~bad_inds]

    ind_v_pos = np.where(((V > 0.002) & (V < 0.04)))
    ind_v_neg = np.where(((V > -0.004) & (V < -0.002)))

    ind_max_right = np.where(slp == np.max(slp[ind_v_pos]))
    ind_max_left = np.where(slp == np.max(slp[ind_v_neg]))

    aver = (V[ind_max_right] - V[ind_max_left]) / 2

    offset_v = aver - V[ind_max_right]  # for addition

    V = V + offset_v
    v_nearest2zero = min(abs(V))
    ind_0 = np.where((V == v_nearest2zero) | (V == -v_nearest2zero))
    offset_i = -I[ind_0]  # for addition

    return offset_v, offset_i


def carve_iv(lst):
    v = np.array([float(it.split("\t")[0].replace(",", ".")) for it in lst])
    i = np.array([float(it.split("\t")[1].replace(",", ".")) for it in lst])
    return dict(V=v, I=i)


def parse_iv(name, split):
    lst = []
    with open(name, "r") as f:
        lst = f.readlines()
        lst = lst[split[0] : split[1]]

    iv_dict = carve_iv(lst)

    return iv_dict


def parse_ivs(name):
    splits = defaultdict(list)
    data = dict()
    with open(name, "r") as f:

        curve_num = 0
        lines = f.readlines()
        for i, line in enumerate(lines):
            if re.search(r"#START Curve Data", line):
                curve_num += 1
                splits[curve_num].append(i + 1)
            if re.search(r"#END Curve [0-9]", line):
                splits[curve_num].append(i)

        data = {
            num: carve_iv(lines[split[0] : split[1]]) for num, split in splits.items()
        }

    return data


def slope(x, y):
    der = np.zeros(len(x), dtype=float)

    rise = y[2:] - y[:-2]
    run = x[2:] - x[:-2]

    with np.errstate(divide="ignore"):
        der[1:-1] = rise / run
        der[0] = (y[1] - y[0]) / (x[1] - x[0])
        der[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return der


def filter_increase(x: np.ndarray):
    """
    Returns: Appropriate indexes of initial data
    """
    indx = [0]
    for i, el in enumerate(x):
        if i == 0:
            continue
        mask = x < el
        if all(mask[:i]):
            indx.append(i)
    return indx


def spline_curve(x: np.ndarray, y: np.ndarray):
    indx = filter_increase(x)
    tck = splrep(x[indx], y[indx], s=0)
    return BSpline(*tck)


def derivative_spline_curve(
    x: np.ndarray,
    y: np.ndarray,
):
    indx = filter_increase(x)
    tck = splrep(x[indx], y[indx], s=0)
    return lambda xnew: splev(xnew, tck, der=1)


def get_gap(
    voltage: np.ndarray,
    current: np.ndarray,
    voltage_gap_start: float = 0.0022,
    voltage_gap_end: float = 0.003,
    voltage_rn_start: float = 0.004,
    sgf_window: int = 50,
    sgf_degree: int = 5,
    plot: bool = False,
) -> Tuple[float, float]:
    """
    :param voltage: Voltage [V]
    :param current: Current [A]
    :param voltage_gap_start: Voltage [V] where gap starts
    :param voltage_gap_end: Voltage [V] where gap ends
    :param voltage_rn_start: Voltage [V] where Normal resistance starts
    :param sgf_window
    :param sgf_degree
    :param plot
    """
    voltage = savgol_filter(voltage, sgf_window, sgf_degree)
    current = savgol_filter(current, sgf_window, sgf_degree)

    linear = lambda x, a, b: a * x + b

    iv_fun = spline_curve(voltage, current)

    voltage = np.linspace(voltage[0], voltage[-1], 2001)
    current = iv_fun(voltage)

    rd = 1 / slope(voltage, current)

    opt, _ = curve_fit(
        linear, voltage[voltage > voltage_rn_start], current[voltage > voltage_rn_start]
    )
    rn = 1 / opt[0]

    # find low current
    voltage_gap = voltage[
        np.where(
            rd
            == np.min(rd[(voltage > voltage_gap_start) & (voltage < voltage_gap_end)])
        )
    ]
    mask = (voltage > voltage_gap_start) & (voltage < voltage_gap)
    rd_mask = rd[mask]
    delta = np.abs(rd_mask - rn / 2)
    closest_to_rn = rd_mask[delta == np.min(delta)]
    current_low = current[mask][np.where((rd_mask == closest_to_rn))]
    voltage_low = voltage[mask][np.where((rd_mask == closest_to_rn))]

    # find high current
    voltage_range = voltage[(voltage > voltage_gap_start) & (voltage < voltage_gap_end)]
    current_range = current[(voltage > voltage_gap_start) & (voltage < voltage_gap_end)]
    rd_lin = np.vectorize(lambda x: linear(x, *opt))(voltage_range)
    cross_indx = np.argwhere(np.diff(np.sign(rd_lin - current_range))).flatten()
    min_cross_ind = np.min(cross_indx)
    current_high = current_range[min_cross_ind]
    voltage_high = voltage_range[min_cross_ind]

    i_gap = current_high - current_low

    # find vgap
    center_igap = current_low + i_gap / 2
    voltage_range_2 = np.linspace(voltage_gap_start, voltage_gap_end, 1001)
    delta = np.abs(iv_fun(voltage_range_2) - center_igap)

    v_gap = voltage_range_2[np.where(delta == np.min(delta))]

    if plot:
        plt.figure()
        plt.title(f"V_gap = {v_gap[0] * 1e3:.3} mV; I_gap = {i_gap[0] * 1e3:.3} mA")
        plt.plot(voltage * 1e3, current * 1e3, label="I-V curve")
        plt.plot(voltage_range * 1e3, rd_lin * 1e3, label="Rn")
        plt.scatter(voltage_high * 1e3, current_high * 1e3, label="current high", c="r")
        plt.scatter(
            voltage_low * 1e3, current_low * 1e3, label="current low", c="green"
        )
        plt.scatter(
            v_gap * 1e3,
            (current_low + i_gap / 2) * 1e3,
            label=f"v_gap & i_gap / 2",
            c="black",
        )
        plt.xlabel("Voltage, mV")
        plt.ylabel("Current, mA")
        plt.grid()
        plt.legend()
        plt.show()

    return v_gap[0], i_gap[0]
