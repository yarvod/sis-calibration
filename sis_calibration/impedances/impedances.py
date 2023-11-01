import pandas as pd
from sis_calibration.utils import reim
import os


path = os.path.dirname(os.path.abspath(__file__))


class Impedances:

    PATH2DATA = path + "/data/"

    DESIGNS = dict(
        (("t1", "T1.csv"), ("t2", "T2.csv"), ("t3", "T3.csv"), ("t4", "T4.csv"))
    )

    def __init__(self, design: str) -> None:
        assert design in self.DESIGNS, f"Choose one of designs: {self.DESIGNS.keys()}"
        self.design = design

    def __str__(self):
        return f"Design: {self.design}"

    __repr__ = __str__

    @property
    def design_path(self):
        return self.PATH2DATA + self.DESIGNS[self.design]

    @property
    def raw_data(self):
        return pd.read_csv(self.design_path)

    @property
    def data(self):
        reim_data = reim(self.raw_data)
        data = pd.DataFrame(dict(freq=self.raw_data.freq, reim=reim_data))
        return data
