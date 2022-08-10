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
    return 10*np.log10(moving_average((np.abs(vec))**2, n))


def calc_offset(V, I):
    I = np.array(list(I))
    V = np.array(list(V))
    
    slp = slope(V, I)
    
    ind_v_pos = np.where(V>0.002)
    ind_v_neg = np.where(V<-0.002)
    
    ind_max_right = np.where(slp==np.max(slp[ind_v_pos]))
    ind_max_left = np.where(slp==np.max(slp[ind_v_neg]))
    
    aver = (V[ind_max_right] - V[ind_max_left]) / 2
    
    offset_v = aver - V[ind_max_right]  # for addition
    
    V = V + offset_v
    v_nearest2zero = min(abs(V))
    ind_0 = np.where((V==v_nearest2zero)|(V==-v_nearest2zero))
    offset_i = - I[ind_0]  # for addition
    
    return offset_v, offset_i