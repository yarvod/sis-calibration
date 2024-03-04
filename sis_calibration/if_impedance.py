import logging
from datetime import datetime
from typing import Tuple

import numpy as np
from mpmath import besselj
from scipy.constants import hbar, e

from .respfn import RespFnFromIVData
from .utils import kron

logger = logging.getLogger(__name__)
debug = logger.debug


def G(
    resp: RespFnFromIVData,
    nu: float,
    nu0: float,
    V0: float,
    al: float,
    v_gap: float,
    i_gap: float,
    lim: int = 10,
    mrange: Tuple[int] = (-1, 0, 1)
):
    """
    :param resp: Response function from autonomus I-V curve
    :param float nu: LO rate
    :param float nu0: IF rate
    :param float V0: bias voltage
    :param float al: Pumping level
    :param float v_gap: Gap voltage
    :param float i_gap: Gap current
    :param int lim: Sum limit
    :param tuple of int mrange: matrix size
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
                        * kron(m - m1, n1 - n)
                        * (
                                (
                                        resp.idc(
                                            (V0 + n1 * hbar * om / e + hbar * omm(m1) / e)
                                            / v_gap
                                        )
                                        - resp.idc(
                                    (V0 + n1 * hbar * om / e) / v_gap
                                )
                                )
                                + (
                                        resp.idc((V0 + n * hbar * om / e) / v_gap)
                                        - resp.idc(
                                    (V0 + n * hbar * om / e - hbar * omm(m1) / e)
                                    / v_gap
                                )
                                )
                        )
                        * i_gap
                    )
            g[m + d][m1 + d] *= e / (2 * hbar * omm(m1))

    delta = datetime.now() - start_time
    debug(f"G calculation time: {delta}")
    return g


def B(resp: RespFnFromIVData,
    nu: float,
    nu0: float,
    V0: float,
    al: float,
    v_gap: float,
    i_gap: float,
    lim: int = 10,
    mrange: Tuple[int] = (-1, 0, 1)
):
    """
    :param resp: Response function from autonomus I-V curve
    :param float nu: LO rate
    :param float nu0: IF rate
    :param float V0: bias voltage
    :param float al: Pumping level
    :param float v_gap: Gap voltage
    :param float i_gap: Gap current
    :param int lim: Sum limit
    :param tuple of int mrange: matrix size
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
                        * kron(m - m1, n1 - n)
                        * (
                                (
                                        resp.ikk(
                                            (V0 + n1 * hbar * om / e + hbar * omm(m1) / e)
                                            / v_gap
                                        )
                                        - resp.ikk(
                                    (V0 + n1 * hbar * om / e) / v_gap
                                )
                                )
                                - (
                                        resp.ikk((V0 + n * hbar * om / e) / v_gap)
                                        - resp.ikk(
                                    (V0 + n * hbar * om / e - hbar * omm(m1) / e)
                                    / v_gap
                                )
                                )
                        )
                        * i_gap
                    )
            b[m + d][m1 + d] *= e / (2 * hbar * omm(m1))

    delta = datetime.now() - start_time
    debug(f"B calculation time: {delta}")
    return b


def Z(
    resp: RespFnFromIVData,
    nu: float,
    nu0: float,
    V0: float,
    al: float,
    v_gap: float,
    i_gap: float,
    ym: Tuple[float] = None,
    lim: int = 10,
    mrange: Tuple[int] = (-1, 0, 1)
):
    """
    :param resp: Response function from autonomus I-V curve
    :param float nu: LO rate
    :param float nu0: IF rate
    :param float V0: bias voltage
    :param float al: Pumping level
    :param float v_gap: Gap voltage
    :param float i_gap: Gap current
    :param tuple of float ym: Ym vector
    :param int lim: Sum limit
    :param tuple of int mrange: matrix size
    """
    start_time = datetime.now()
    if ym is not None:
        g = np.array(G(resp, nu, nu0, V0, al, v_gap, i_gap, lim, mrange))
        b = np.array(B(resp, nu, nu0, V0, al, v_gap, i_gap, lim, mrange))
        y = g + np.eye(3, 3) * ym + b * 1j
        res = np.linalg.inv(y)[1][1]

    else:
        g = np.array(G(resp, nu, nu0, V0, al, v_gap, i_gap, lim, (0,)))
        b = np.array(B(resp, nu, nu0, V0, al, v_gap, i_gap, lim, (0,)))
        y = g[0][0] + b[0][0] * 1j
        res = 1 / y

    delta = datetime.now() - start_time
    debug(f"Z calculation time: {delta}")
    return res
