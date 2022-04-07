"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel
# from magicgui import magic_factory, magicgui
from scipy.ndimage import gaussian_filter

import localize_psf.rois as roi_fns
from localize_psf import fit
import localize_psf.fit_psf as psf
from localize_psf import localize

class FullSlider(QWidget):
    """
    Custom Slider widget with its label and value displayed.
    """

    def __init__(self, range=(0, 1), step=0.01, label='', layout=QHBoxLayout, *args, **kwargs):
        super(FullSlider, self).__init__(*args, **kwargs)

        self.step = step

        layout = layout()

        self.label = QLabel(label)
        layout.addWidget(self.label)

        if isinstance(layout, QHBoxLayout):
            self.sld = QSlider(Qt.Horizontal)
        else:
            self.sld = QSlider(Qt.Vertical)
        # wrangle range and steps as QtSlider handles only integers
        mini = int(range[0] / step)
        maxi = int(range[1] / step)
        # self.sld.setRange(*range)
        self.sld.setRange(mini, maxi)
        self.sld.setPageStep(1)  # minimum possible
        self.sld.valueChanged.connect(self._convert_value)
        # the real converted value we want
        self.value = self.sld.value() * self.step
        layout.addWidget(self.sld)

        self.readout = QLabel(str(self.value))
        layout.addWidget(self.readout)

        self.setLayout(layout)
        # make available the connect method
        self.valueChanged = self.sld.valueChanged

    def _convert_value(self):
        self.value = self.sld.value() * self.step
        self.readout.setText("{:.2f}".format(self.value))


class KernelQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # DoG blob detection widgets
        self.sld_sigma_xy_small = FullSlider(range=(0.1, 20), step=0.1, label="sigma xy small")
        self.sld_sigma_xy_small.valueChanged.connect(self._on_slide)
        self.sld_sigma_xy_large = FullSlider(range=(0.1, 20), step=0.1, label="sigma xy large")
        self.sld_sigma_xy_large.valueChanged.connect(self._on_slide)
        self.sld_sigma_z_small = FullSlider(range=(0.1, 20), step=0.1, label="sigma z small")
        self.sld_sigma_z_small.valueChanged.connect(self._on_slide)
        self.sld_sigma_z_large = FullSlider(range=(0.1, 20), step=0.1, label="sigma z large")
        self.sld_sigma_z_large.valueChanged.connect(self._on_slide)

        self.sld_dog_thresh = FullSlider(range=(0.1, 20), step=0.1, label="DoG threshold")
        self.sld_dog_thresh.valueChanged.connect(self._on_slide)

        self.but_dog = QPushButton()
        self.but_dog.setText('Apply DoG')
        self.but_dog.clicked.connect(self._compute_dog)

        # gaussian fitting widgets
        self.sld_gauss = FullSlider(label="gauss fit")
        self.but_fit = QPushButton()
        self.but_fit.setText('Fit spots')
        # self.but_fit.clicked.connect(self._fit_spots)

        # spot filtering widgets
        self.but_filter = QPushButton()
        self.but_filter.setText('Filter spots')
        # self.but_filter.clicked.connect(self._filter_spots)


        # general layout of the widget
        outerLayout = QVBoxLayout()
        # layout for DoG blob detection
        dogLayout = QVBoxLayout()
        dogLayout.addWidget(self.sld_sigma_xy_small)
        dogLayout.addWidget(self.sld_sigma_xy_large)
        dogLayout.addWidget(self.sld_sigma_z_small)
        dogLayout.addWidget(self.sld_sigma_z_large)
        dogLayout.addWidget(self.sld_dog_thresh)
        dogLayout.addWidget(self.but_dog)
        # layout for fitting gaussian spots
        fitLayout = QVBoxLayout()
        fitLayout.addWidget(self.but_fit)
        # layout for filtering gaussian spots
        filterLayout = QHBoxLayout()
        filterLayout.addWidget(self.but_filter)

        outerLayout.addLayout(dogLayout)
        outerLayout.addLayout(fitLayout)
        outerLayout.addLayout(filterLayout)

        self.setLayout(outerLayout)

    def _on_slide(self):
        print("sigma is {:.2f}".format(self.sld_sigma_xy_small.value))

    def _compute_dog(self):
        
        if len(self.viewer.layers) == 0:
            print("Open an image first")
        else:
            filter_sigma_small = (self.sld_sigma_z_small.value, self.sld_sigma_xy_small.value, self.sld_sigma_xy_small.value)
            filter_sigma_large = (self.sld_sigma_z_large.value, self.sld_sigma_xy_large.value, self.sld_sigma_xy_large.value)
            pixel_sizes = (1, 1, 1)
            sigma_cutoff = 2
            kernel_small = localize.get_filter_kernel(filter_sigma_small, pixel_sizes, sigma_cutoff)
            kernel_large = localize.get_filter_kernel(filter_sigma_large, pixel_sizes, sigma_cutoff)

            img = self.viewer.layers[0].data
            img_high_pass = localize.filter_convolve(img, kernel_small, use_gpu=False)
            img_low_pass = localize.filter_convolve(img, kernel_large, use_gpu=False)
            img_filtered = img_high_pass - img_low_pass
            # im_gauss = gaussian_filter(self.viewer.layers[0].data, sigma=self.sld_sigma_xy_small.value)
            if 'filtered' not in self.viewer.layers:
                self.viewer.add_image(img_filtered, name='filtered')
            else:
                self.viewer.layers['filtered'].data = img_filtered

# class ExampleQWidget(QWidget):
#     # your QWidget.__init__ can optionally request the napari viewer instance
#     # in one of two ways:
#     # 1. use a parameter called `napari_viewer`, as done here
#     # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
#     def __init__(self, napari_viewer):
#         super().__init__()
#         self.viewer = napari_viewer

#         btn = QPushButton("Click me!")
#         btn.clicked.connect(self._on_click)

#         self.setLayout(QHBoxLayout())
#         self.layout().addWidget(btn)

#     def _on_click(self):
#         print("napari has", len(self.viewer.layers), "layers")


# @magic_factory
# def example_magic_widget(img_layer: "napari.layers.Image"):
#     print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
# def example_function_widget(img_layer: "napari.layers.Image"):
#     print(f"you have selected {img_layer}")

# sigma_options = {
#     'label': 'sigma of gaussian kernel:', 
#     'widget_type':'Slider',
#     'min': 0.1, 
#     'max' : 20.0, 
#     'step': 0.1,
#     }
# param_options = {'sigma':sigma_options})
# def make_kernel(sigma: float = 1):
#     print(f"sigma is {sigma}")

# kernel_magicgui_widget = magicgui(make_kernel, sigma=sigma_options, auto_call=False)