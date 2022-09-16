import re
from collections import defaultdict

import numpy as np
from qmix.mathfn.misc import slope


def reim(df, num=None):
    if not num:
        num = len(df)
    data_z = np.zeros(shape=(num), dtype=np.complex128)
    data_z.real = np.array(df['re'], dtype=np.float64)
    data_z.imag = np.array(df['im'], dtype=np.float64)
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
    return ret[n - 1:] / n


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
    offset_i = - I[ind_0]  # for addition

    return offset_v, offset_i


def carve_iv(lst):
    v = np.array([float(it.split('\t')[0].replace(',', '.')) for it in lst])
    i = np.array([float(it.split('\t')[1].replace(',', '.')) for it in lst])
    return dict(V=v, I=i)


def parse_iv(name, split):
    lst = []
    with open(name, 'r') as f:
        lst = f.readlines()
        lst = lst[split[0]: split[1]]

    iv_dict = carve_iv(lst)

    return iv_dict


def parse_ivs(name):
    splits = defaultdict(list)
    data = dict()
    with open(name, 'r') as f:

        curve_num = 0
        lines = f.readlines()
        for i, line in enumerate(lines):
            if re.search(r'#START Curve Data', line):
                curve_num += 1
                splits[curve_num].append(i + 1)
            if re.search(r'#END Curve [0-9]', line):
                splits[curve_num].append(i)

        data = {num: carve_iv(lines[split[0]:split[1]]) for num, split in splits.items()}

    return data
