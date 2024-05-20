#!/usr/bin/env python
import math
import numpy as np
import pandas as pd

class PIDController():
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute_control_cmd(self, control_input, control_output):
        curvature = control_input['curvature']
        wheel_base = control_input["wheel_base"]

        # lateral_error = control_input['lat_err']
        # heading_err = control_input['heading_err']
        lateral_error = control_input['preview_lat_err']
        heading_err = control_input['preview_heading_err']
        feedforward = np.arctan(wheel_base * curvature)
        feedback = -(lateral_error * 0.7 + heading_err * 0.0)
        control_output['feedforward'] = feedforward
        control_output['feedback'] = feedback
        control_output['lat_cmd'] = feedforward + feedback
        return [feedback]