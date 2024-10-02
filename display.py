import h5py
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.colors import LogNorm
from IPython.display import display
import numpy as np
import tifffile
import pprint
pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)  # Disable sorting of keys
pprint = pp.pprint

class Display:
    """
    Class to visualize datasets from an HDF5 file, with interactive controls for dataset selection, 
    scale adjustments (linear/log), and saving the displayed data as TIFF images.

    Parameters:
    ----------
    dataset : str, optional
        The default dataset key to be loaded from the HDF5 file. Defaults to 'alvium_nf_ff'.
    run_number : int, optional
        The run number used to construct the HDF5 file path. Defaults to 115.
    data_path : str, optional
        The base path where HDF5 files are stored. Defaults to "/sdf/data/lcls/ds/mec/mecl1038823/hdf5/smalldata/".
    **kwargs : dict, optional
        Additional keyword arguments to customize behavior:
        - figsize : tuple, optional
            The size of the figure.
        - scale : str, optional
            The initial color scale for the plot ('linear' or 'log'). Defaults to 'linear'.
        - vmin : float, optional
            The initial minimum value for the color scale.
        - vmax : float, optional
            The initial maximum value for the color scale.
        - xlim : tuple, optional
            Limits for the x-axis, e.g., (0, 100).
        - ylim : tuple, optional
            Limits for the y-axis, e.g., (0, 100).
        - save_tiffs : bool, optional
            If True, saves the current dataset display as a TIFF file. Defaults to False.
    """
    def __init__(self, 
                 dataset='alvium_nf_ff', 
                 run_number=115, 
                 data_path="/sdf/data/lcls/ds/mec/mecl1038823/hdf5/smalldata/",
                 **kwargs
                ):
        # Initialize dataset path and data
        self.data_path = data_path
        self.run_number = run_number
        self.run_file = f'{self.data_path:s}mecl1038823_Run{self.run_number:04d}.h5'
        self.run_data = h5py.File(self.run_file, 'r')
        
        self.kwargs = kwargs
        
        # Collect all the available dataset keys
        self.dataset_keys = [key for key in self.run_data.keys()]

        # Load initial dataset
        self.dataset_key = dataset
        self.data = self.load_data(self.dataset_key)

        # Set up widgets and plot
        self.init_widgets()
        self.init_plot()
        self.display()

    def load_data(self, key):
        try:
            data = self.run_data[key]['full_area'][()][0, :, :]
        except:
            data = None
        return data

    def init_widgets(self):
        # Dataset selector
        self.dataset_selector = widgets.Dropdown(
            options=self.dataset_keys,
            value=self.dataset_key,
            description='Dataset:'
        )

        # Scale selector
        self.scale_selector = widgets.Dropdown(
            options=['linear', 'log'],
            value=self.kwargs.get('scale', 'linear'),
            description='Scale:'
        )

        # Define consistent layout for sliders
        slider_layout = widgets.Layout(width='400px')

        def v_value(direction="min",vmin=-1e99,vmax=1e99):
            # Extract xlim and ylim from kwargs, if present
            xlim = self.kwargs.get('xlim', (0, self.data.shape[1]))  # Default to the full x-range
            ylim = self.kwargs.get('ylim', (0, self.data.shape[0]))  # Default to the full y-range

            # Ensure the xlim and ylim are within bounds
            x_min, x_max = max(0, xlim[0]), min(self.data.shape[1], xlim[1])
            y_min, y_max = max(0, ylim[0]), min(self.data.shape[0], ylim[1])

            # Calculate the min value within the specified xlim and ylim range
            data_subset = self.data[y_min:y_max, x_min:x_max]
            if direction == "min":
                v_value = np.max([self.kwargs.get('vmin', data_subset.min()),0.0])
            if direction == "max":
                v_value = np.min([self.kwargs.get('vmax', data_subset.max()),vmax])
            return v_value
        
        # Linear sliders for adjusting color scale range (vmin and vmax)
        self.vmin_slider_linear = widgets.FloatSlider(
            value=v_value("min"),
            min=self.data.min(),
            max=self.data.max(),
            step=0.1,
            description='Vmin:',
            layout=slider_layout
        )

        self.vmax_slider_linear = widgets.FloatSlider(
            value=v_value("max"),
            min=self.data.min(),
            max=self.data.max(),
            step=0.1,
            description='Vmax:',
            layout=slider_layout
        )

        # Log sliders for adjusting color scale range (vmin and vmax)
        self.vmin_slider_log = widgets.FloatLogSlider(
            value=self.kwargs.get('vmin', v_value("min",vmin=1e-6)),
            base=10,
            min=np.log10(np.max([self.data.min(), 1e-6])),
            max=np.log10(self.data.max()),
            step=0.1,
            description='Vmin:',
            readout_format='.2e',
            layout=slider_layout
        )

        self.vmax_slider_log = widgets.FloatLogSlider(
            value=v_value("max"),
            base=10,
            min=np.log10(np.max([self.data.min(), 1e-6])),
            max=np.log10(self.data.max()),
            step=0.1,
            description='Vmax:',
            readout_format='.2e',
            layout=slider_layout
        )

        # Combine sliders
        self.linear_sliders = widgets.VBox([self.vmin_slider_linear, self.vmax_slider_linear])
        self.log_sliders = widgets.VBox([self.vmin_slider_log, self.vmax_slider_log])
        self.slider_box = widgets.VBox([self.linear_sliders])  # Show linear sliders initially
        
        self.save_tiffs = widgets.Checkbox(
            description = "save as tiff",
            value = self.kwargs.get("save_tiffs",False)
        )

        self.output_widget = widgets.Output(
            layout=slider_layout
        )  # Initialize an output widget for print statements
        if "shotsheet" in self.kwargs:
            with self.output_widget:
                run_number = self.run_number
                try:
                    pprint(self.kwargs["shotsheet"].get_all(filter={"run_number":run_number})[run_number])
                except:
                    print("Warning! No metadata for this run!")
    
        self.output_py_widget = widgets.Output(
            layout=widgets.Layout(width='800px')
        )  # Initialize an output widget for print statements
        if "shotsheet" in self.kwargs:
            with self.output_py_widget:
                run_number = self.run_number
                print("Data in workbook 'Python':'")
                try:
                    pprint(self.kwargs["shotsheet"].get_all(filter={"run_number":run_number},python=True)[run_number])
                except:
                    print("Warning! No metadata for this run!")

        # Add observers
        self.dataset_selector.observe(self.on_dataset_change, names='value')
        self.scale_selector.observe(self.on_scale_change, names='value')
        self.vmin_slider_linear.observe(self.on_vmin_change, names='value')
        self.vmax_slider_linear.observe(self.on_vmax_change, names='value')
        self.vmin_slider_log.observe(self.on_vmin_change, names='value')
        self.vmax_slider_log.observe(self.on_vmax_change, names='value')
        self.save_tiffs.observe(self.save_tiff, names='value')

    def init_plot(self):
        # Turn off interactive mode to prevent automatic display
        plt.ioff()
        # Create a figure and axis for the plot
        if "figsize" in self.kwargs:
            self.fig, self.ax = plt.subplots(figsize=self.kwargs["figsize"])
        else:
            self.fig, self.ax = plt.subplots()
        
        self.fig.canvas.header_visible = False

        self.colorbar = None  # Initialize a variable for the colorbar
        self.update_plot()

    def update_plot(self):
        # Update the plot based on user inputs
        scale = self.scale_selector.value
        vmin = self.vmin_slider_log.value if scale == 'log' else self.vmin_slider_linear.value
        vmax = self.vmax_slider_log.value if scale == 'log' else self.vmax_slider_linear.value

        self.ax.clear()  # Clear the plot

        # Define the normalization based on the selected scale
        if scale == 'log':
            im = self.ax.imshow(self.data, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax), aspect=self.kwargs.get("aspect","equal"))
        else:
            im = self.ax.imshow(self.data, cmap='inferno', vmin=vmin, vmax=vmax, aspect=self.kwargs.get("aspect","equal"))

            
        if "xlim" in self.kwargs: self.ax.set_xlim(self.kwargs["xlim"])
        if "ylim" in self.kwargs: self.ax.set_ylim(self.kwargs["ylim"])
        
        self.ax.set_title(f'run: {self.run_number} | Dataset: {self.dataset_selector.value}')

        # Remove the old colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()

        # Add a new colorbar
        self.colorbar = plt.colorbar(im, ax=self.ax)
        self.fig.canvas.draw_idle()
        self.save_tiff()

    def update_sliders(self):
        # Update sliders based on the selected scale
        scale = self.scale_selector.value
        data = self.data
        M = data.max()
        m = np.max([data.min(), 1e-6])  # Min value clamped to 1e-6 to avoid log issues

        if scale == 'log':
            # Configure log sliders
            self.vmin_slider_log.min = np.log10(m)
            self.vmin_slider_log.max = np.log10(M)
            self.vmax_slider_log.min = np.log10(m)
            self.vmax_slider_log.max = np.log10(M)
            self.vmin_slider_log.value = np.log10(m)
            self.vmax_slider_log.value = np.log10(M)

            # Display log sliders
            self.slider_box.children = [self.log_sliders]
        else:
            # Configure linear sliders
            self.vmin_slider_linear.min = m
            self.vmin_slider_linear.max = M
            self.vmax_slider_linear.min = m
            self.vmax_slider_linear.max = M
            self.vmin_slider_linear.value = m
            self.vmax_slider_linear.value = M

            # Display linear sliders
            self.slider_box.children = [self.linear_sliders]

    def save_tiff(self, change=0):
        save = self.save_tiffs.value
        if save:
            filename = f'tiffs/{self.run_number:04d}_{self.dataset_selector.value}.tiff'
            from pathlib import Path
            Path("tiffs").mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(filename, self.data.astype(np.float32))
        
    def on_scale_change(self, change):
        self.update_sliders()
        self.update_plot()

    def on_vmin_change(self, change):
        self.update_plot()

    def on_vmax_change(self, change):
        self.update_plot()

    def on_dataset_change(self, change):
        self.data = self.load_data(change['new'])
        self.update_sliders()
        self.update_plot()

    def display(self):
        # Arrange the plot and widgets side by side
        controls = widgets.VBox([self.dataset_selector, self.scale_selector, self.slider_box,self.save_tiffs, self.output_widget])
        row1 = widgets.HBox([self.fig.canvas, controls])  # Place widgets next to the plot
        display_area = widgets.VBox([row1, self.output_py_widget])

        # Display everything
        display(display_area)