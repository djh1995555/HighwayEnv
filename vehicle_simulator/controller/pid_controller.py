#!/usr/bin/env python
import math
import numpy as np
import pandas as pd

class PIDController():
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute_control_cmd(self, lateral_error):
        return [-lateral_error * 0.5]