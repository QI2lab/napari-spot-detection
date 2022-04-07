"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QSlider, QLabel, QLineEdit
# from magicgui import magic_factory, magicgui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import napari
import scipy.signal
import scipy.ndimage as ndi
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

    def set_value(self, value):
        # first set the slider at the correct position
        self.sld.setValue(int(value / self.step))
        # then convert the slider position to have the value
        # we don't directly convert in order to account for rounding errors in the silder
        self._convert_value()


class KernelQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # expected spot size
        self.lab_spot_size_xy = QLabel('Expected spot size xy (px)')
        self.txt_spot_size_xy = QLineEdit()
        self.txt_spot_size_xy.setText('5')
        self.lab_spot_size_z = QLabel('Expected spot size z (px)')
        self.txt_spot_size_z = QLineEdit()
        self.txt_spot_size_z.setText('5')
        self.lab_sigma_ratio = QLabel('Expected spot size z (px)')
        self.txt_sigma_ratio = QLineEdit()
        self.txt_sigma_ratio.setText('1.6')
        self.but_auto_sigmas = QPushButton()
        self.but_auto_sigmas.setText('Auto sigmas')
        self.but_auto_sigmas.clicked.connect(self._make_sigmas)

        # DoG blob detection widgets
        self.sld_sigma_xy_small = FullSlider(range=(0.1, 20), step=0.1, label="sigma xy small")
        self.sld_sigma_xy_small.valueChanged.connect(self._on_slide)
        self.sld_sigma_xy_large = FullSlider(range=(0.1, 20), step=0.1, label="sigma xy large")
        self.sld_sigma_xy_large.valueChanged.connect(self._on_slide)
        self.sld_sigma_z_small = FullSlider(range=(0.1, 20), step=0.1, label="sigma z small")
        self.sld_sigma_z_small.valueChanged.connect(self._on_slide)
        self.sld_sigma_z_large = FullSlider(range=(0.1, 20), step=0.1, label="sigma z large")
        self.sld_sigma_z_large.valueChanged.connect(self._on_slide)

        self.sld_blob_thresh = FullSlider(range=(0.1, 20), step=0.1, label="Blob threshold")
        self.sld_blob_thresh.valueChanged.connect(self._on_slide)

        self.but_dog = QPushButton()
        self.but_dog.setText('Apply DoG')
        self.but_dog.clicked.connect(self._compute_dog)

        self.but_find_peaks = QPushButton()
        self.but_find_peaks.setText('Find peaks')
        self.but_find_peaks.clicked.connect(self._find_peaks)

        # gaussian fitting widgets
        self.but_fit = QPushButton()
        self.but_fit.setText('Fit spots')
        # self.but_fit.clicked.connect(self._fit_spots)

        # spot filtering widgets
        self.but_filter = QPushButton()
        self.but_filter.setText('Filter spots')
        # self.but_filter.clicked.connect(self._filter_spots)


        # general layout of the widget
        outerLayout = QVBoxLayout()
        # layout for spot size parametrization
        spotsizeLayout = QVBoxLayout()
        spotsizeLayout_xy = QHBoxLayout()
        spotsizeLayout_xy.addWidget(self.lab_spot_size_xy)
        spotsizeLayout_xy.addWidget(self.txt_spot_size_xy)
        spotsizeLayout_z = QHBoxLayout()
        spotsizeLayout_z.addWidget(self.lab_spot_size_z)
        spotsizeLayout_z.addWidget(self.txt_spot_size_z)
        spotsizeLayout_sigmas = QHBoxLayout()
        spotsizeLayout_sigmas.addWidget(self.lab_sigma_ratio)
        spotsizeLayout_sigmas.addWidget(self.txt_sigma_ratio)
        spotsizeLayout_sigmas.addWidget(self.but_auto_sigmas)
        spotsizeLayout.addLayout(spotsizeLayout_xy)
        spotsizeLayout.addLayout(spotsizeLayout_z)
        spotsizeLayout.addLayout(spotsizeLayout_sigmas)

        # layout for DoG filtering
        dogLayout = QVBoxLayout()
        dogLayout.addWidget(self.sld_sigma_xy_small)
        dogLayout.addWidget(self.sld_sigma_xy_large)
        dogLayout.addWidget(self.sld_sigma_z_small)
        dogLayout.addWidget(self.sld_sigma_z_large)
        dogLayout.addWidget(self.but_dog)
        dogLayout.addWidget(self.sld_blob_thresh)
        dogLayout.addWidget(self.but_find_peaks)
        # layout for fitting gaussian spots
        fitLayout = QVBoxLayout()
        fitLayout.addWidget(self.but_fit)
        # layout for filtering gaussian spots
        filterLayout = QHBoxLayout()
        filterLayout.addWidget(self.but_filter)

        outerLayout.addLayout(spotsizeLayout)
        outerLayout.addLayout(dogLayout)
        outerLayout.addLayout(fitLayout)
        outerLayout.addLayout(filterLayout)

        self.setLayout(outerLayout)

    def _on_slide(self):
        # print("sigma is {:.2f}".format(self.sld_sigma_xy_small.value))
        pass

    def _make_sigmas(self):
        """
        Compute min and max of sigmas x, y and z with traditionnal settings.
        """

        sx = float(self.txt_spot_size_xy.text())
        sz = float(self.txt_spot_size_z.text())
        # FWHM = 2.355 x sigma
        sigma_xy = sx / 2.355
        sigma_z = sz / 2.355
        # to reproduce LoG with Dog we need sigma_big = 1.6 * sigma_small
        sigma_ratio = float(self.txt_sigma_ratio.text())
        # sigma_ratio = 2
        sigma_xy_small = sigma_xy / sigma_ratio**(1/2)
        sigma_xy_large = sigma_xy * sigma_ratio**(1/2)
        sigma_z_small = sigma_z / sigma_ratio**(1/2)
        sigma_z_large = sigma_z * sigma_ratio**(1/2)
        self.sld_sigma_xy_small.set_value(sigma_xy_small)
        self.sld_sigma_xy_large.set_value(sigma_xy_large)
        self.sld_sigma_z_small.set_value(sigma_z_small)
        self.sld_sigma_z_large.set_value(sigma_z_large)


    def _compute_dog(self):
        """
        Apply a Differential of Gaussian filter on the first image available in Napari.
        """
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
    
    def _find_peaks(self):
        """
        Threshold the image resulting from the DoG filter and detect peaks.
        """
        if 'filtered' not in self.viewer.layers:
            print("Run a DoG filter on an image first")
        else:
            dog_thresh = self.sld_blob_thresh.value
            img_filtered = self.viewer.layers['filtered'].data
            img_filtered[img_filtered < dog_thresh] = 0

            sx = sy = float(self.txt_spot_size_xy.text())
            sz = float(self.txt_spot_size_z.text())
            min_separations = np.array([sz, sy, sx]).astype(int)

            footprint = localize.get_max_filter_footprint(min_separations=min_separations, drs=(1,1,1))
            # array of size nz, ny, nx of True

            maxis = ndi.maximum_filter(img_filtered, footprint=np.ones(min_separations))
            self.centers_guess_inds, amps = localize.find_peak_candidates(img_filtered, footprint, threshold=dog_thresh)
            if 'local maxis' not in self.viewer.layers:
                self.viewer.add_points(self.centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
            else:
                self.viewer.layers['local maxis'].data = self.centers_guess_inds



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