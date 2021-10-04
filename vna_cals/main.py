from tkinter import *
from tkinter import messagebox, filedialog
from tkinter.ttk import Checkbutton
from vna import get_data, save_data
from calibrations import Calibrations

# Experiment path
def browse_button():
    global exp_path
    dir_name = filedialog.askdirectory()
    exp_path.set(dir_name)

# VNA interaction
def get_picture_button():
    global plot_phase, exp_path
    get_data(param='S21', plot_phase=plot_phase.get(), exp_path=exp_path.get())

def save_data_button():
    global exp_path
    pic_path = filedialog.asksaveasfilename(defaultextension=".csv")
    save_data(pic_path=pic_path, exp_path=exp_path.get())

def current_calibrate():
    global exp_path, open_path, short_path, load_path
    csv_path = {
        'open' : open_path,
        'short' : short_path,
        'load' : load_path,
        'point' : f'{exp_path}/current/data.csv'
    }
    calibrations = Calibrations()
    calibrations

    pass

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

def attach_measure_button():
    global measure_path, measure_path_rel
    measure_path.set(filedialog.askopenfilename())
    if measure_path and exp_path:
        measure_path_rel.set(''.join(measure_path.get().rsplit(exp_path.get())))

def apply_resist(cal, resist):
    print(cal, resist)

root = Tk()

root['bg'] = 'white'
root.title('vna-cals')
root.wm_attributes('-alpha', 1)
root.geometry('700x400')
root.resizable(width=True, height=True)

# Experiment path
Label(master=root, text='Experiment path:', font=('bold', '18')).grid(row=0, column=1, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
exp_path = StringVar()
Label(master=root, textvariable=exp_path).grid(row=0, column=2, ipadx=5, ipady=5, padx=2, pady=2, columnspan=2)
Button(text='Browse', command=browse_button).grid(row=0,column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

# VNA interaction
Label(master=root, text='VNA interaction:', font=('bold', '18')).grid(row=1, column=1, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
plot_phase = BooleanVar()
plot_phase_chk = Checkbutton(root, text='Plot phase', var=plot_phase)
plot_phase_chk.grid(row=2, column=1, ipadx=5, ipady=5, padx=2, pady=2)

Button(text='Get graph', command=get_picture_button).grid(row=2, column=2, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Button(text='Save data', command=save_data_button).grid(row=2, column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

# Calibrations
Label(master=root, text='Calibrations:', font=('bold', '18')).grid(row=3, column=1, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Label(master=root, text='Attach open cal:').grid(row=4, column=1, ipadx=5, ipady=5, padx=2, pady=2)
open_path = StringVar()
open_path_rel = StringVar()
Label(master=root, textvariable=open_path_rel).grid(row=4, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(text='Attach', command=attach_open_button).grid(row=4,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
resist_open = StringVar(root)
resist_open_entry = Entry(root, textvariable=resist_open).grid(row=4,column=4, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')
Button(text='Apply resist', command= lambda: apply_resist('open', float(resist_open.get()))).grid(row=4,column=6, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=root, text='Attach short cal:').grid(row=5, column=1, ipadx=5, ipady=5, padx=2, pady=2)
short_path = StringVar()
short_path_rel = StringVar()
Label(master=root, textvariable=short_path_rel).grid(row=5, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(text='Attach', command=attach_short_button).grid(row=5,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=root, text='Attach load cal:').grid(row=6, column=1, ipadx=5, ipady=5, padx=2, pady=2)
load_path = StringVar()
load_path_rel = StringVar()
Label(master=root, textvariable=load_path_rel).grid(row=6, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(text='Attach', command=attach_load_button).grid(row=6,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

Label(master=root, text='Attach measure:').grid(row=7, column=1, ipadx=5, ipady=5, padx=2, pady=2)
measure_path = StringVar()
measure_path_rel = StringVar()
Label(master=root, textvariable=measure_path_rel).grid(row=7, column=2, ipadx=5, ipady=5, padx=2, pady=2)
Button(text='Attach', command=attach_measure_button).grid(row=7,column=3, ipadx=5, ipady=5, padx=2, pady=2, sticky='w')

root.mainloop()