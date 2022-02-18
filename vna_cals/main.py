import csv
from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Checkbutton, Combobox
from vna import get_data, save_data, save_calibrated_data
from Mixer import Mixer, Calibration


# Experiment path
def browse_button():
    global exp_path
    dir_name = filedialog.askdirectory()
    exp_path.set(dir_name)


# VNA interaction
def get_picture_button():
    global plot_phase, exp_path
    num = int(iterations.get()) if iterations.get() else None
    spann = int(span.get()) if span.get() else None
    start = float(freq_start.get())
    stop = float(freq_stop.get())
    get_data(param=combo_param.get(),
             plot_phase=plot_phase.get(),
             freq_start=start, freq_stop=stop,
             exp_path=exp_path.get(),
                      freq_num=int(point_num.get()) or 201,
                               num=num, mov_aver=mov_aver.get(), span=spann, aver=aver.get())


def save_data_button():
    global exp_path
    pic_path = filedialog.asksaveasfilename(defaultextension=".csv")
    save_data(pic_path=pic_path, exp_path=exp_path.get())


def calibrate_button(meas):
    global exp_path, open_path, short_path, load_path
    cal_path = {
        'open': open_path.get(),
        'short': short_path.get(),
        'load': load_path.get(),
    }
    V_bias = {
        'open': float(V_bias_open.get()),
        'short': float(V_bias_short.get()),
        'load': float(V_bias_load.get())
    }
    IV_csv_path = IV_curve_path.get()
    meas_path = f'{exp_path.get()}/current/data.csv' if meas=='current' else f'{measure_path.get()}'

    mixer = Mixer(IV_csv_path=IV_csv_path, meas_path=meas_path, cal_path=cal_path, V_bias=V_bias,
                  point_num=int(point_num.get()) or 201)
    mixer.set_calibration()
    mixer.calibrate()
    mixer.calibration.plot(plot_phase=plot_phase.get())

    if meas == 'current':
        with open(f'{exp_path.get()}/current/calibrated_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['freq-Hz', 're', 'im'])
            for i in range(len(mixer.calibration.freq_list)):
                writer.writerow(
                    [
                        mixer.calibration.freq_list[i],
                        mixer.calibration.point_calibrated[i].real,
                        mixer.calibration.point_calibrated[i].imag
                    ]
                )


def save_calibrated_data_button():
    global exp_path
    calibrated_data_path = filedialog.asksaveasfilename(defaultextension=".csv")
    save_calibrated_data(data_path=calibrated_data_path, exp_path=exp_path.get())
    save_data_button()


# Calibrations
def attach_open_button():
    global open_path, open_path_rel, exp_path
    open_path.set(filedialog.askopenfilename())
    if open_path and exp_path:
        open_path_rel.set(''.join(open_path.get().rsplit(exp_path.get())))


def attach_short_button():
    global short_path, short_path_rel
    short_path.set(filedialog.askopenfilename())
    if short_path and exp_path:
        short_path_rel.set(''.join(short_path.get().rsplit(exp_path.get())))


def attach_load_button():
    global load_path, load_path_rel
    load_path.set(filedialog.askopenfilename())
    if load_path and exp_path:
        load_path_rel.set(''.join(load_path.get().rsplit(exp_path.get())))


def attach_IV_curve_button():
    global IV_curve_path, IV_curve_path_rel
    IV_curve_path.set(filedialog.askopenfilename())
    if IV_curve_path and exp_path:
        IV_curve_path_rel.set(''.join(IV_curve_path.get().rsplit(exp_path.get())))


def attach_measure_button():
    global measure_path, measure_path_rel
    measure_path.set(filedialog.askopenfilename())
    if measure_path and exp_path:
        measure_path_rel.set(''.join(measure_path.get().rsplit(exp_path.get())))


def apply_V_bias(cal, resist):
    global V_bias
    if cal == 'open':
        V_bias['open'] = resist
    elif cal == 'short':
        V_bias['short'] = resist
    elif cal == 'load':
        V_bias['load'] = resist
    print(V_bias)


def plot_calibrations_button():
    global exp_path, open_path, short_path, load_path
    cal_path = {
        'open': open_path.get(),
        'short': short_path.get(),
        'load': load_path.get(),
    }

    calibration = Calibration(cal_path=cal_path, point_num=int(point_num.get()) or 201)
    calibration.plot_cals()


# Global variables

# Main part
root = Tk()

frame_Experiment = Frame(relief=RAISED, borderwidth=1)
frame_VNA = Frame(relief=RAISED, borderwidth=1)
frame_VNA_settings = Frame(relief=RAISED, borderwidth=1)
frame_Calibrations = Frame(relief=RAISED, borderwidth=1)
# frame_Analysis = Frame(relief=RAISED, borderwidth=1)

root['bg'] = 'white'
root.title('vna-cals')
root.wm_attributes('-alpha', 1)
root.geometry('830x585')
root.resizable(width=True, height=False)

# Experiment path
Label(master=frame_Experiment, text='Experiment path:', font=('bold', '18')).grid(row=0, column=0, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
exp_path = StringVar()
Label(master=frame_Experiment, textvariable=exp_path).grid(row=0, column=1, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_Experiment, text='Browse', command=browse_button).grid(row=0,column=2, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

# VNA settings
Label(master=frame_VNA_settings, text='VNA Settings:', font=('bold', '18')).grid(row=0, column=0, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
combo_param = Combobox(master=frame_VNA_settings)
combo_param['values'] = ('S11', 'S12', 'S21', 'S22')
combo_param.current(2)
combo_param.grid(row=1, column=0, ipadx=5, ipady=5, padx=2, pady=2, sticky='sw')
Label(master=frame_VNA_settings, text='Point num:').grid(row=1, column=1, ipadx=5, ipady=5, padx=2, pady=2)
point_num = StringVar()
point_num_entry = Entry(master=frame_VNA_settings, textvariable=point_num).grid(row=1,column=2, ipadx=1, ipady=1, padx=2, pady=2, sticky='w')
plot_phase = BooleanVar()
plot_phase_chk = Checkbutton(master=frame_VNA_settings, text='Plot phase', var=plot_phase).grid(row=1, column=3, ipadx=5, ipady=5, padx=2, pady=2)
mov_aver = BooleanVar()
mov_aver_chk = Checkbutton(master=frame_VNA_settings, text='Moving average. span:', var=mov_aver).grid(row=2, column=0, ipadx=5, ipady=5, padx=2, pady=2)
span = StringVar()
span_entry = Entry(master=frame_VNA_settings, textvariable=span).grid(row=2,column=1, ipadx=1, ipady=1, padx=2, pady=2, sticky='w')
aver = BooleanVar()
aver_chk = Checkbutton(master=frame_VNA_settings, text='VNA average. iterations:', var=aver).grid(row=2, column=2, ipadx=5, ipady=5, padx=2, pady=2)
iterations = StringVar()
iterations_entry = Entry(master=frame_VNA_settings, textvariable=iterations).grid(row=2,column=3, ipadx=1, ipady=1, padx=2, pady=2, sticky='w')

Label(master=frame_VNA_settings, text='start freq:').grid(row=3, column=0, ipadx=5, ipady=5, padx=2, pady=2)
freq_start = StringVar()
freq_start_entry = Entry(master=frame_VNA_settings, textvariable=freq_start).grid(row=3,column=1, ipadx=1, ipady=1, padx=2, pady=2, sticky='w')
Label(master=frame_VNA_settings, text='stop freq:').grid(row=3, column=2, ipadx=5, ipady=5, padx=2, pady=2)
freq_stop = StringVar()
freq_stop_entry = Entry(master=frame_VNA_settings, textvariable=freq_stop).grid(row=3,column=3, ipadx=1, ipady=1, padx=2, pady=2, sticky='w')


# VNA interaction
Label(master=frame_VNA, text='VNA interaction:', font=('bold', '18')).grid(row=0, column=0, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Button(master=frame_VNA, text='Plot current', command=get_picture_button).grid(row=1, column=1, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Button(master=frame_VNA, text='Save current data', command=save_data_button).grid(row=1, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_VNA, text='Calibrate current', command=lambda: calibrate_button('current')).grid(row=1, column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Button(master=frame_VNA, text='Save calibrated data', command=save_calibrated_data_button).grid(row=1, column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

# Calibrations
Label(master=frame_Calibrations, text='Calibrations:', font=('bold', '18')).grid(row=0, column=1, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Label(master=frame_Calibrations, text='Attach open cal:').grid(row=1, column=1, ipadx=5, ipady=5, padx=2, pady=2)
open_path = StringVar()
open_path_rel = StringVar()
Label(master=frame_Calibrations, textvariable=open_path_rel).grid(row=1, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_Calibrations, text='Attach', command=attach_open_button).grid(row=1, column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
V_bias_open = StringVar()
V_bias_open_entry = Entry(master=frame_Calibrations, textvariable=V_bias_open).grid(row=1,column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=frame_Calibrations, text='Attach short cal:').grid(row=2, column=1, ipadx=5, ipady=5, padx=2, pady=2)
short_path = StringVar()
short_path_rel = StringVar()
Label(master=frame_Calibrations, textvariable=short_path_rel).grid(row=2, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_Calibrations, text='Attach', command=attach_short_button).grid(row=2,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
V_bias_short = StringVar()
V_bias_short_entry = Entry(master=frame_Calibrations, textvariable=V_bias_short).grid(row=2,column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=frame_Calibrations, text='Attach load cal:').grid(row=3, column=1, ipadx=5, ipady=5, padx=2, pady=2)
load_path = StringVar()
load_path_rel = StringVar()
Label(master=frame_Calibrations, textvariable=load_path_rel).grid(row=3, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_Calibrations, text='Attach', command=attach_load_button).grid(row=3,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
V_bias_load = StringVar()
V_bias_load_entry = Entry(master=frame_Calibrations, textvariable=V_bias_load).grid(row=3,column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=frame_Calibrations, text='Attach I-V curve:').grid(row=4, column=1, ipadx=5, ipady=5, padx=2, pady=2)
IV_curve_path = StringVar()
IV_curve_path_rel = StringVar()
Label(master=frame_Calibrations, textvariable=IV_curve_path_rel).grid(row=4, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_Calibrations, text='Attach', command=attach_IV_curve_button).grid(row=4,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=frame_Calibrations, text='Attach measure:').grid(row=5, column=1, ipadx=5, ipady=5, padx=2, pady=2)
measure_path = StringVar()
measure_path_rel = StringVar()
Label(master=frame_Calibrations, textvariable=measure_path_rel).grid(row=5, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(master=frame_Calibrations, text='Attach', command=attach_measure_button).grid(row=5,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Button(master=frame_Calibrations, text='Calibrate measure', command=lambda: calibrate_button('measure')).grid(row=5,column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Button(master=frame_Calibrations, text='Plot calibrations', command=plot_calibrations_button).grid(row=5,column=5, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

# Analysis
# Label(master=frame_Analysis, text='Analysis:', font=('bold', '18')).grid(row=0, column=0, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

# showing frames
frame_Experiment.place(height=50, relwidth=0.99, y=5, x=5)
frame_VNA_settings.place(height=170, relwidth=0.99, y=55, x=5)
frame_VNA.place(height=100, relwidth=0.99, y=225, x=5)
frame_Calibrations.place(height=260, relwidth=0.99, y=325, x=5)
# frame_Analysis.place(height=100, relwidth=0.99, y=525, x=5)
root.mainloop()
