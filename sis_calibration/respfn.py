import numpy as np
from scipy.signal import hilbert

from .utils import slope
from scipy.interpolate import InterpolatedUnivariateSpline as Interp


# Build initialization bias voltage
VRANGE = 35
VNPTS = 7001
VINIT = np.linspace(0, VRANGE, VNPTS)
VSTEP = float(VINIT[1] - VINIT[0])
VINIT.flags.writeable = False


# Generate response function --------------------------------------------------


class RespFn(object):
    def __init__(self, voltage, current, **params):

        params = _default_params(params)

        if params["verbose"]:
            print("Generating response function:")

        assert voltage[0] == 0.0, "First voltage value must be zero"
        assert voltage[-1] > 5, "Voltage must extend to at least 5"

        # Reflect about y-axis
        voltage = np.r_[-voltage[::-1][:-1], voltage]
        current = np.r_[-current[::-1][:-1], current]

        # Smear DC I-V curve (optional)
        if params["v_smear"] is not None:
            v_step = voltage[1] - voltage[0]
            current = (
                gauss_conv(current - voltage, sigma=params["v_smear"] / v_step)
                + voltage
            )
            if params["verbose"]:
                print(" - Voltage smear: {:.4f}".format(params["v_smear"]))

        # Calculate Kramers-Kronig (KK) transform
        current_kk = kk_trans(voltage, current, params["kk_n"])

        # Interpolate
        f_interp = _setup_interpolation(voltage, current, current_kk, **params)

        # Place interpolation objects into hidden attributes
        self._f_idc = f_interp[0]
        self._f_ikk = f_interp[1]
        self._f_didc = f_interp[2]
        self._f_dikk = f_interp[3]

        # Save DC I-V curve and KK transform as attributes
        self.voltage = voltage
        self.current = current
        self.voltage_kk = voltage
        self.current_kk = current_kk

    def __str__(self):  # pragma: no cover

        return "Response function object: RespFn"

    def __repr__(self):  # pragma: no cover

        return self.__str__()

    def __call__(self, vbias):

        return self.resp(vbias)

    def idc(self, vbias):
        """Interpolate the DC I-V curve.

        This is the imaginary component of the respones function, and it is
        used to calculate the quasiparticle tunneling currents in
        ``qmix.qtcurrent.qtcurrent``.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: DC tunneling current

        """

        return self._f_idc(vbias)

    def ikk(self, vbias):
        """Interpolate the Kramers-Kronig transform of the DC I-V curve at the
        given bias voltage.

        This is the real component of the response function, and it is
        used to calculate the quasiparticle tunneling currents in
        ``qmix.qtcurrent.qtcurrent``.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: KK transform of the DC I-V curve

        """

        return self._f_ikk(vbias)

    def didc(self, vbias):
        """Interpolate the derivative of the DC I-V curve at the given bias
        voltage.

        This is defined as ``d(idc) / d(vb)`` where ``idc`` is the DC tunneling
        current and ``vb`` is the bias voltage.

        Note:

            This method is not used directly by QMix, but it can be useful if
            you are calculating the tunneling currents using Tucker theory
            (see: Tucker and Feldman, 1985).

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: derivative of the DC tunneling current

        """

        return self._f_didc(vbias)

    def dikk(self, vbias):
        """Interpolate the derivative of the Kramers-Kronig transform.

        This is defined as ``d(ikk) / d(vb)`` where ``ikk`` is the Kramers-
        Kronig transform of the DC tunneling current and ``vb`` is the bias
        voltage.

        Note:

            This method is not used directly by QMix, but it can be useful if
            you are calculating the tunneling currents using Tucker theory
            (see: Tucker and Feldman, 1985).

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: derivative of the KK transform of the DC I-V curve

        """

        return self._f_dikk(vbias)

    def resp(self, vbias):
        """Interpolate the response function.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Response function (a complex value)

        """

        return self._f_ikk(vbias) + 1j * self._f_idc(vbias)

    def resp_conj(self, vbias):
        """Interpolate the complex conjugate of the response function.

        Note:

            This method is not used directly by QMix, but it can be useful if
            you are calculating the tunneling currents using Tucker theory
            (see: Tucker and Feldman, 1985).

            This method is included because it might be *slightly* faster than
            ``np.conj(resp(vb))`` where ``resp`` is an instance of this
            class.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Complex conjugate of the response function

        """

        return self._f_ikk(vbias) - 1j * self._f_idc(vbias)

    def resp_swap(self, vbias):
        """Interpolate the response function, with the real and imaginary
        components swapped.

        Note:

            This method is not used directly by QMix, but it can be useful if
            you are calculating the tunneling currents using Tucker theory
            (see: Tucker and Feldman, 1985).

            This method is included because it might be *slightly* faster than
            ``1j*np.conj(resp(vb))`` where ``resp`` is an instance of
            this class. This is the normal way that you would swap the real
            and imaginary components.

        Args:
            vbias (ndarray): Bias voltage (normalized)

        Returns:
            ndarray: Response function with the real and imaginary components
                swapped

        """

        return self._f_idc(vbias) + 1j * self._f_ikk(vbias)


# Generate from I-V data ------------------------------------------------------


class RespFnFromIVData(RespFn):
    """Generate the response function from I-V data.

    Unlike ``RespFn``, this class will resample the I-V data to optimize the
    interpolation.

    Note:

        This class expects normalized I-V data that extends from at least
        ``vb=0`` to ``vb=vlimit``, where ``vb`` is the bias voltage and
        ``vlimit`` is one of the keyword arguments.

    Args:
        voltage (ndarray): normalized DC bias voltage
        current (ndarray): normalized DC tunneling current

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        v_smear (float, default is None): smear DC I-V curve by convolving with
            a Gaussian dist. with this std. dev.
        kk_n (int, default is 50): padding for Hilbert transform
            (see ``qmix.mathfn.kktrans.kk_trans``)
        spline_order (int, default is 3): spline order for interpolations
        vlimit (float, default is 1.8): import all DC I-V data from ``vb=0`` to
            ``vb=vlimit``, where ``vb`` is the bias voltage normalized to the
            gap voltage.

    """

    def __init__(self, voltage, current, **kwargs):
        params = _default_params(kwargs, 81, 101)

        # Force slope=1 above vmax
        vmax = params.get("vlimit", 1.8)
        mask = (voltage > 0) & (voltage < vmax)
        current = current[mask]
        voltage = voltage[mask]
        b = current[-1] - voltage[-1]
        current = np.append(current, [50.0 + b])
        voltage = np.append(voltage, [50.0])

        # Re-sample I-V data
        current = np.interp(VINIT, voltage, current)
        voltage = np.copy(VINIT)

        RespFn.__init__(self, voltage, current, **params)

    def __str__(self):
        return "Response function object: RespFnFromIVData"


# Helper functions ------------------------------------------------------------


def _setup_interpolation(voltage, current, current_kk, **params):
    """Setup interpolation.

    This function will sample the response function, such that there are more
    points around curvier regions than linear regions, and then setup the
    interpolation. This is optimized to make interpolating the data as fast as
    possible.

    Args:
        voltage: bias voltage
        current: DC tunneling current
        current_kk: KK transform of the DC tunneling current
        **params: interpolation parameters (see keyword arguments)

    Keyword Args:
        verbose (bool, default is True): print info to terminal?
        max_npts_dc (int, default is 101): maximum number of points in DC I-V
            curve
        max_npts_kk (int, default is 151): maximum number of points in KK
            transform
        max_interp_error (float, default is 0.001): maximum interpolation error
            (in units of normalized current)
        check_error (bool, default is False): check interpolation error?
        spline_order (int, default is 3): spline order for interpolations

    Returns:
        tuple: the interpolation objects

    """

    # Interpolation parameters
    npts_dciv = params["max_npts_dc"]
    npts_kkiv = params["max_npts_kk"]
    interp_error = params["max_interp_error"]
    check_error = params["check_error"]
    verbose = params["verbose"]
    spline_order = params["spline_order"]

    if verbose:
        print(" - Interpolating:")

    # Sample data
    dc_idx = _sample_curve(voltage, current, npts_dciv, 0.25)
    kk_idx = _sample_curve(voltage, current, npts_kkiv, 1.0)

    # Interpolate (cubic spline)
    # Note: k=1 or 2 is much faster, but increases the numerical error.
    f_dc = Interp(voltage[dc_idx], current[dc_idx], k=spline_order)
    f_kk = Interp(voltage[kk_idx], current_kk[kk_idx], k=spline_order)

    # Splines for derivatives
    f_ddc = f_dc.derivative()
    f_dkk = f_kk.derivative()

    # Find max error
    v_check_range = VRANGE - 1
    # idx_start = (voltage + v_check_range).argmin()
    idx_check = (-v_check_range < voltage) & (voltage < v_check_range)
    err_v = voltage[idx_check]
    err_dc = current[idx_check] - f_dc(voltage[idx_check])
    err_kk = current_kk[idx_check] - f_kk(voltage[idx_check])

    # # Debug
    # plt.figure()
    # plt.plot(voltage, current, 'k')
    # plt.plot(voltage[dc_idx], f_dc(voltage[dc_idx]), 'ro--')
    # plt.figure()
    # plt.plot(voltage, current_kk, 'k')
    # plt.plot(voltage[kk_idx], f_kk(voltage[kk_idx]), 'ro--')
    # plt.show()

    # Print to terminal
    if verbose:
        msg1 = "\t- DC I-V curve:"
        msg2 = "\t\t- npts for DC I-V: {}"
        msg3 = "\t\t- avg. error: {:.4E}"
        msg4 = "\t\t- max. error: {:.4f} at v={:.2f}"
        msg5 = "\t- KK curve:"
        msg6 = "\t\t- npts for KK I-V: {}"
        msg7 = "\t\t- avg. error: {:.4E}"
        msg8 = "\t\t- max. error: {:.4f} at v={:.2f}"
        msg9 = ""

        print(msg1)
        print(msg2.format(len(dc_idx)))
        print(msg3.format(np.mean(np.abs(err_dc))))
        print(msg4.format(err_dc.max(), err_v[err_dc.argmax()]))
        print(msg5)
        print(msg6.format(len(kk_idx)))
        print(msg7.format(np.mean(np.abs(err_kk))))
        print(msg8.format(err_kk.max(), err_v[err_kk.argmax()]))
        print(msg9)

    # Check error
    if check_error:
        assert (
            err_dc.max() < interp_error
        ), "Interpolation error too high. Please increase max_npts_dc"
        assert (
            err_kk.max() < interp_error
        ), "Interpolation error too high. Please increase max_npts_kk"

    return f_dc, f_kk, f_ddc, f_dkk


def _sample_curve(voltage, current, max_npts, smear):
    """Sample curve. Sample more often when the curve is curvier.

    Args:
        voltage (ndarray): DC bias voltage
        current (ndarray): current (either DC tunneling or KK)
        max_npts (int): maximum number of sample points
        smear (float): smear current (only for sampling purposes)

    Returns:
        list: indices of sample points

    """

    # Second derivative
    dd_current = np.abs(slope(voltage, slope(voltage, current)))

    # Cumulative sum of second derivative
    v_step = voltage[1] - voltage[0]
    cumsum = np.cumsum(gauss_conv(dd_current, sigma=smear / v_step))

    # Build sampling array
    idx_list = [0]
    # Add indices based on curvy-ness
    cumsum_last = 0.0
    voltage_last = voltage[0]
    for idx, v in enumerate(voltage):
        condition1 = abs(v) < 0.05 or abs(v - 1) < 0.1 or abs(v + 1) < 0.1
        condition2 = v - voltage_last >= 1.0
        condition3 = (cumsum[idx] - cumsum_last) * max_npts / cumsum[-1] > 1
        condition4 = idx < 3 or idx > len(voltage) - 4
        if condition1 or condition2 or condition3 or condition4:
            if idx != idx_list[-1]:
                idx_list.append(idx)
            voltage_last = v
            cumsum_last = cumsum[idx]
    # Add 10 to start/end
    for i in range(0, int(1 / VSTEP), int(1 / VSTEP / 10)):
        idx_list.append(i)
    for i in range(
        len(voltage) - int(1 / VSTEP) - 1, len(voltage), int(1 / VSTEP / 10)
    ):
        idx_list.append(i)
    # Add 30 pts to middle
    ind_low = np.abs(voltage + 1.0).argmin()
    ind_high = np.abs(voltage - 1.0).argmin()
    npts = ind_high - ind_low
    for i in range(ind_low, ind_high, npts // 30):
        idx_list.append(i)

    idx_list = list(set(idx_list))
    idx_list.sort()

    return idx_list


def _default_params(
    kwargs,
    max_dc=101,
    max_kk=151,
    max_error=0.001,
    check_error=False,
    verbose=True,
    v_smear=None,
    kk_n=50,
    spline_order=3,
):
    """These are the default parameters that are used for generating response
    functions. These parameters match the keyword arguments of ``RespFn``, so
    see that docstring for more information."""

    # Grab default params from the keyword arguments for this function
    params = {
        "max_npts_dc": max_dc,
        "max_npts_kk": max_kk,
        "max_interp_error": max_error,
        "check_error": check_error,
        "verbose": verbose,
        "v_smear": v_smear,
        "kk_n": kk_n,
        "spline_order": spline_order,
    }

    # Update kwargs with the new parameters
    params.update(kwargs)

    return params


def gauss_conv(x, sigma=10, ext_x=3):
    """Smooth data using a Gaussian convolution.

    Args:
        x (ndarray): noisy data
        sigma (float): std. dev. of Gaussian curve, given as number of data
                       points
        ext_x (float): Gaussian curve will extend from ext_x * sigma in each
                       direction

    Returns:
        ndarray: filtered data

    """

    wind = _gauss(sigma, ext_x)
    wlen = np.len(wind)

    assert wlen <= np.alen(x), "Window size must be smaller than data size"
    assert sigma * ext_x >= 1, "Window size must be larger than 1. Increase ext_x."

    s = np.r_[x[wlen - 1 : 0 : -1], x, x[-2 : -wlen - 1 : -1]]
    y_out = np.convolve(wind / wind.sum(), s, mode="valid")
    y_out = y_out[wlen // 2 : -wlen // 2 + 1]

    return y_out


def _gauss(sigma, n_sigma=3):
    """Generate a discrete, normalized Gaussian centered on zero.

    Used for filtering data.

    Args:
        sigma (float): standard deviation
        n_sigma (float): extend x in each direction by ext_x * sigma

    Returns:
        ndarray: discrete Gaussian curve

    """

    x_range = n_sigma * sigma
    x = np.arange(-x_range, x_range + 1e-5, 1, dtype=float)

    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma) ** 2)

    return y


def kk_trans(v, i, n=50):
    """Calculate the Kramers-Kronig transform from DC I-V data.

    Note:

        Voltage spacing must be constant!

    Args:
        v (ndarray): normalized voltage (DC I-V curve)
        i (ndarray): normalized current (DC I-V curve)
        n (int): padding for Hilbert transform

    Returns:
        ndarray: kk transform

    """

    npts = v.shape[0]

    # Ensure v has (roughly) even spacing
    assert np.abs((v[1] - v[0]) - (v[1:] - v[:-1])).max() < 1e-5

    # Subtract v to make kk defined at v=infinity
    ikk = -(hilbert(i - v, N=npts * n)).imag
    ikk = ikk[:npts]

    return ikk
