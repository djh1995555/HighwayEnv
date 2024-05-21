#!/usr/bin/env python
import os
import shutil
import yaml
import argparse
import datetime
import pandas as pd
import gymnasium as gym
from collections import OrderedDict
from controller.pid_controller import PIDController
from utils.report_generator import ReportGenerator
from stable_baselines3 import SAC

class VehicleSimulator():
    def __init__(self, args, epoch = 0, train_mode = False):
        self._simulator_config_filepath = os.path.join(dir_path,'config', args.config)
        with open(self._simulator_config_filepath, 'r') as f:
            self._simulator_config = yaml.load(f,Loader=yaml.Loader)
        env_name = self._simulator_config["env_name"]
        self._output_dir = os.path.join(dir_path,'output', env_name, '{}'.format(datetime.datetime.now()))
        self._model_dir = os.path.join(dir_path,'model', env_name, '{}'.format(datetime.datetime.now()))
        self._env = gym.make(env_name, render_mode="rgb_array")

        if(self._simulator_config['control_type'] == 'classic'):
            if(self._simulator_config['controller_type'] == 'pid'):
                self._controller = PIDController()
        elif(self._simulator_config['control_type'] == 'rl_based'):
            if(self._simulator_config['model_type'] == 'sac'):
                model_filepath = os.path.join(dir_path,self._simulator_config["sac_model_filepth"])
                self._controller = SAC.load(model_filepath)

        self._target_signals = []
        for sub_panel, list in self._simulator_config["plot_target_signal"].items():
            for signal_name in list:
                self._target_signals.append(signal_name)
        self._report_generator = ReportGenerator()
        self.reset()

    def reset(self):
        self._data_selected = OrderedDict()
        self._debug_info = dict()
        for signal_name in self._target_signals: 
            self._debug_info[signal_name] = []

    def run(self):
        done = truncated = False
        obs, info = self._env.reset()
        count = 0
        control_output = {}
        while not done and count < 320:
            # positive -> right
            if(self._simulator_config['control_type'] == 'classic'):
                if("lat_err" in info["debug_info"]):
                    action = self._controller.compute_control_cmd(info["debug_info"], control_output)
                else:
                    action = [0]
            elif(self._simulator_config['control_type'] == 'rl_based'):
                action, _states = self._controller.predict(obs, deterministic=True)
                control_output['lat_cmd'] = action[0]
            obs, reward, done, truncated, info = self._env.step(action)
            
            self.construct_debug_info(info["debug_info"],control_output)
            self._env.render()
            count += 1
        output_name = os.path.join(self._output_dir, "report.html")
        self.select_target_signal()
        self._report_generator.plot(self._data_selected, output_name)
        self.cpoy_config_file()

    def cpoy_config_file(self):
        tgt_config = os.path.join(self._output_dir,args.config)
        shutil.copyfile(self._simulator_config_filepath, tgt_config)

    def construct_debug_info(self, debug_info, control_output):
        for signal_name in self._target_signals: 
            if(signal_name in debug_info):
                self._debug_info[signal_name].append(debug_info[signal_name])
            elif(signal_name in control_output):
                self._debug_info[signal_name].append(control_output[signal_name])

    def select_target_signal(self):
        for sub_panel, list in self._simulator_config["plot_target_signal"].items():
            sub_dict = OrderedDict()
            for signal_name in list:
                time = [i for i in range(len(self._debug_info[signal_name]))]
                sub_dict[signal_name] = (time, self._debug_info[signal_name])
            self._data_selected[sub_panel] = sub_dict  

def main(args):
		vehicle_simulator = VehicleSimulator(args)
		vehicle_simulator.run()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sim Vehicle Dynamic Model')
    dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--config', default='simulator_config.yaml', type=str)
    parser.add_argument('--train-data-dir', default=os.path.join(dir_path,'data','train_data'))
    parser.add_argument('--output-dir', default=os.path.join(dir_path,'output','{}'.format(datetime.datetime.now())))
    parser.add_argument('--model-dir', default=os.path.join(dir_path,'model','{}'.format(datetime.datetime.now())))
    parser.add_argument('--target-signal-filepath', default=os.path.join(dir_path,'config/plot_signal.yaml'), type=str)
    args = parser.parse_args()
  
    main(args)