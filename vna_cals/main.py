from tkinter import filedialog, StringVar, BooleanVar, IntVar
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame

from vna import get_data, save_data
from Mixer import Mixer
from scpi import Block
from NIBlock import NIContainer
import numpy as np


class Base:

    block = Block()
    niblock = NIContainer()
    # mixer = Mixer()

    # SetUp
    def browse_button(self):
        dir_name = filedialog.askdirectory()
        self.exp_path.set(dir_name)

    # VNA
    def show_data(self, save=False, plot=True):
        start = float(self.freq_start.get())
        stop = float(self.freq_stop.get())
        get_data(param=self.s_param.get(),
                 vna_ip=self.vna_ip.get(),
                 plot=plot,
                 plot_phase=self.plot_phase.get(),
                 freq_start=start, freq_stop=stop,
                 exp_path=self.exp_path.get(),
                 freq_num=int(self.point_num.get()) or 201,
                 avg=self.vna_avg.get())

        if save:
            pic_path = filedialog.asksaveasfilename(defaultextension=".csv")
            save_data(pic_path=pic_path, exp_path=self.exp_path.get())

    # Calibrations
    def attach_file(self, which: str):
        if which == 'open':
            self.open_path.set(filedialog.askopenfilename())
            if self.open_path and self.exp_path:
                self.open_path_rel.set(''.join(self.open_path.get().rsplit(self.exp_path.get())))
        elif which == 'short':
            self.short_path.set(filedialog.askopenfilename())
            if self.short_path and self.exp_path:
                self.short_path_rel.set(''.join(self.short_path.get().rsplit(self.exp_path.get())))
        elif which == 'load':
            self.load_path.set(filedialog.askopenfilename())
            if self.load_path and self.exp_path:
                self.load_path_rel.set(''.join(self.load_path.get().rsplit(self.exp_path.get())))

        elif which == 'IV':
            self.IV_curve_path.set(filedialog.askopenfilename())
            if self.IV_curve_path and self.exp_path:
                self.IV_curve_path_rel.set(''.join(self.IV_curve_path.get().rsplit(self.exp_path.get())))

    # I-V curve
    def meas_iv(self, plot=False, save=False):
        iv = self.block.measure_IV(
            v_from=float(self.volt_start.get()),
            v_to=float(self.volt_stop.get()), points=int(self.iv_point_num.get()))
        if plot:
            self.block.plot_iv(iv)
        if save:
            path = filedialog.asksaveasfilename(defaultextension=".csv")
            self.block.write_IV_csv(path=path, iv=iv)

    def meas_iv_new(self, plot=False, save=False):
        volt_range = np.linspace(
            float(self.volt_start.get()), float(self.volt_stop.get()), int(self.iv_point_num.get())
        )
        self.niblock.update_params
        self.niblock.measure_iv(volt_range)
        if plot:
            self.niblock.plot_iv()
        if save:
            path = filedialog.asksaveasfilename(defaultextension=".csv")
            self.niblock.write_IV_csv(path=path)

    def calc_offset(self):
        pass

    def measure_reflection(self, save=True):
        volt_range = np.linspace(
            float(self.volt_start.get()), float(self.volt_stop.get()), int(self.iv_point_num.get())
        )
        self.niblock.measure_reflection(
            volt_range=volt_range,
            f_from=float(self.freq_start.get()), f_to=float(self.freq_stop.get()), f_points=int(self.point_num.get()),
            s_par=self.s_param.get(), exp_path=self.exp_path.get(), avg=self.vna_avg.get(), vna_ip=self.vna_ip.get(),
        )
        self.niblock.plot_iv()
        if save:
            path_refl = filedialog.asksaveasfilename(defaultextension=".csv")
            self.niblock.write_refl_csv(path=path_refl)
            path_iv = filedialog.asksaveasfilename(defaultextension=".csv")
            self.niblock.write_IV_csv(path=path_iv)
        


class UI(ttk.Frame, Base):

    def __init__(self, isapp=True, name='ui'):
        ttk.Frame.__init__(self, name=name)
        Base.__init__(self)
        self.pack(expand=Y, fill=BOTH)
        self.vna_ip = '10.208.234.8'

        self.master.title('UI')
        self.isapp = isapp
        self._create_widgets()

    def _create_widgets(self):
        self._create_demo_panel()

    def _create_demo_panel(self):
        demoPanel = ttk.Frame(self, name='demo')
        demoPanel.pack(side=TOP, fill=BOTH, expand=Y)

        # create the notebook
        nb = ttk.Notebook(demoPanel, name='notebook', bootstyle="info")

        nb.enable_traversal()

        nb.pack(fill=BOTH, expand=Y, padx=2, pady=3)
        self._create_setup_tab(nb)
        self._create_vna_tab(nb)
        self._create_calibration_tab(nb)
        self._create_ivcurve_tab(nb)

    def _create_setup_tab(self, nb):
        frame = ttk.Frame(nb, name='setup')
        # widgets to be displayed on 'Description' tab

        self.exp_path = StringVar()
        self.vna_ip = StringVar(value='10.208.234.8')
        self.block_ip = StringVar(value='192.168.1.34')

        ttk.Label(frame, text='Experiment path:', font=('bold', '14'))\
            .grid(row=0, column=0, padx=5, pady=5, sticky='W')
        ttk.Label(frame, textvariable=self.exp_path)\
            .grid(row=0, column=1, padx=5, pady=5, sticky='W')
        ttk.Button(frame, text='Browse', command=self.browse_button)\
            .grid(row=0, column=2, padx=5, pady=5, sticky='W')
        ttk.Label(frame, text='VNA IP:')\
            .grid(row=1, column=0, padx=5, pady=5, sticky='W')
        ttk.Entry(frame, textvariable=self.vna_ip)\
            .grid(row=1, column=1, padx=5, pady=5, sticky='W')

        # position and set resize behaviour
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure((0, 1), weight=1, uniform=1)

        # add to notebook (underline = index for short-cut character)
        nb.add(frame, text='SetUp', underline=0, padding=2)

    def _create_vna_tab(self, nb):
        frame = ttk.Frame(nb, name='vna')

        ttk.Label(frame, text='S-Parameter:').grid(row=0, column=0, padx=5, pady=5)

        self.s_param = ttk.Combobox(frame)
        self.s_param['values'] = ('S11', 'S12', 'S21', 'S22')
        self.s_param.current(2)
        self.s_param.grid(row=0, column=1, padx=5, pady=5)

        self.point_num = StringVar()
        ttk.Label(frame, text='Point num:').grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.point_num).grid(row=1, column=1, padx=5, pady=5)

        self.freq_start = StringVar()
        ttk.Label(frame, text='start freq:').grid(row=2, column=0, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.freq_start).grid(row=2, column=1, padx=5, pady=5)

        self.freq_stop = StringVar()
        ttk.Label(frame, text='stop freq:').grid(row=3, column=0, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.freq_stop).grid(row=3, column=1, padx=5, pady=5)

        self.vna_avg = IntVar()
        self.vna_avg.set(1)
        ttk.Label(frame, text='avg:').grid(row=4, column=0, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.vna_avg).grid(row=4, column=1, padx=5, pady=5)

        ttk.Button(frame, text='Show', command=self.show_data).grid(row=5, column=0, padx=5, pady=5)
        ttk.Button(frame, text='Show&Save', command=lambda: self.show_data(True)).grid(row=5, column=1, padx=5, pady=5)
        self.plot_phase = BooleanVar()
        ttk.Checkbutton(frame, text='Plot phase', var=self.plot_phase).grid(row=5, column=2, padx=5, pady=5)

        # add to notebook (underline = index for short-cut character)
        nb.add(frame, text='VNA', underline=0, padding=2)

    def _create_calibration_tab(self, nb):
        frame = ttk.Frame(nb, name='calibrations')

        ttk.Label(frame, text='Open cal:') \
            .grid(row=0, column=0)

        self.open_path = StringVar()
        self.open_path_rel = StringVar()

        ttk.Label(frame, textvariable=self.open_path_rel) \
            .grid(row=0, column=1)
        ttk.Button(frame, text='Attach', command=lambda: self.attach_file('open')) \
            .grid(row=0, column=2)
        self.V_bias_open = StringVar()
        ttk.Entry(frame, textvariable=self.V_bias_open) \
            .grid(row=0, column=3)

        ttk.Label(frame, text='Short cal:') \
            .grid(row=1, column=0)
        self.short_path = StringVar()
        self.short_path_rel = StringVar()
        ttk.Label(frame, textvariable=self.short_path_rel) \
            .grid(row=1, column=1)
        ttk.Button(frame, text='Attach', command=lambda: self.attach_file('short')) \
            .grid(row=1, column=2)
        self.V_bias_short = StringVar()
        ttk.Entry(frame, textvariable=self.V_bias_short) \
            .grid(row=1, column=3)

        ttk.Label(frame, text='Load cal:') \
            .grid(row=2, column=0)
        self.load_path = StringVar()
        self.load_path_rel = StringVar()
        ttk.Label(frame, textvariable=self.load_path_rel) \
            .grid(row=2, column=1)
        ttk.Button(frame, text='Attach', command=lambda: self.attach_file('load')) \
            .grid(row=2, column=2)
        self.V_bias_load = StringVar()
        ttk.Entry(frame, textvariable=self.V_bias_load) \
            .grid(row=2, column=3)

        ttk.Label(frame, text='I-V curve:') \
            .grid(row=3, column=0)
        self.IV_curve_path = StringVar()
        self.IV_curve_path_rel = StringVar()
        ttk.Label(frame, textvariable=self.IV_curve_path_rel) \
            .grid(row=3, column=1)
        ttk.Button(frame, text='Attach', command=lambda: self.attach_file('IV')) \
            .grid(row=3, column=2)

        # Label(frame, text='Attach measure:') \
        #     .grid(row=5, column=1, ipadx=5, ipady=5, padx=2, pady=2)
        # measure_path = StringVar()
        # measure_path_rel = StringVar()
        # Label(frame, textvariable=measure_path_rel) \
        #     .grid(row=5, column=2, ipadx=5, ipady=5, padx=2, pady=2)
        # Button(frame, text='Attach', command=attach_measure_button) \
        #     .grid(row=5, column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
        #
        # Button(frame, text='Calibrate measure', command=lambda: calibrate_button('measure')) \
        #     .grid(row=5, column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
        # Button(frame, text='Plot calibrations', command=plot_calibrations_button) \
        #     .grid(row=5, column=5, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

        nb.add(frame, text='Calibration', underline=0, padding=2)

    def _create_ivcurve_tab(self, nb):
        frame = ttk.Frame(nb, name='i-v curve')

        iv_frame = ttk.Frame(frame)
        iv_frame.pack()

        self.iv_point_num = StringVar()
        ttk.Label(iv_frame, text='Point num:').grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(iv_frame, textvariable=self.iv_point_num).grid(row=0, column=1, padx=5, pady=5)

        self.volt_start = StringVar()
        ttk.Label(iv_frame, text='start volt:').grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(iv_frame, textvariable=self.volt_start).grid(row=0, column=3, padx=5, pady=5)

        self.volt_stop = StringVar()
        ttk.Label(iv_frame, text='stop volt:').grid(row=0, column=4, padx=5, pady=5)
        ttk.Entry(iv_frame, textvariable=self.volt_stop).grid(row=0, column=5, padx=5, pady=5)

        ttk.Button(iv_frame, text='Measure curve', command=lambda: self.meas_iv_new(save=True)) \
            .grid(row=1, column=0, padx=5, pady=5)

        ttk.Button(iv_frame, text='Measure refl', command=lambda: self.measure_reflection()) \
            .grid(row=1, column=1, padx=5, pady=5)

        self.use_offset = BooleanVar()
        ttk.Checkbutton(iv_frame, text='Use offset', var=self.use_offset).grid(row=2, column=0, padx=5, pady=5)

        ttk.Button(iv_frame, text='Calculate offset', command=lambda: self.calc_offset()) \
            .grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Button(iv_frame, text='set 0', command=lambda: self.niblock.set_zero()) \
            .grid(row=3, column=0, padx=5, pady=5)

        ttk.Button(iv_frame, text='update params', command=lambda: self.niblock.update_params()) \
            .grid(row=3, column=1, padx=5, pady=5)

        nb.add(frame, text='I-V curve', underline=0, padding=2)


if __name__ == '__main__':
    UI().mainloop()
