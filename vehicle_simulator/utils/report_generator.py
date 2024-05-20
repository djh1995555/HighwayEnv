import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportGenerator():
	def __init__(self):
		pass

	def plot(self, data_dict, output_name):
			num_subplots = len(data_dict)
			fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=list(data_dict.keys()), shared_xaxes = True)

			for i, (key, value) in enumerate(data_dict.items(), start=1):
					for sub_key, sub_value in value.items():
							fig.add_trace(go.Scatter(x=sub_value[0], y=sub_value[1], mode='lines', name=sub_key), row=i, col=1)
							fig.update_xaxes(showticklabels=True, row=i, col=1)
							
			fig.update_layout(height = 200 * num_subplots, 
										 		width = 1800, 
												title_x=0.5,
												title_text = "Debug Info of Vehicle Simulator")

			pos = output_name.rfind("/")
			output_dir = output_name[:pos]
			if(not os.path.exists(output_dir)):
				os.makedirs(output_dir)
			fig.write_html(output_name)
