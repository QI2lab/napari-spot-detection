"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QPushButton, QSlider, QLabel, QLineEdit, QCheckBox, QFileDialog
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
# from magicgui import magic_factory, magicgui
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json
import warnings
import napari
import scipy.signal
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter

import localize_psf.rois as roi_fns
from localize_psf import fit
import localize_psf.fit_psf as psf
from localize_psf import localize

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from napari_spot_detection import _image_processing as ip


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

    def setValue(self, value):
        # first set the slider at the correct position
        self.sld.setValue(int(value / self.step))
        # then convert the slider position to have the value
        # we don't directly convert in order to account for rounding errors in the silder
        self._convert_value()


class SpotDetection(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # automatic adaptation of parameters when steps complete, False when loading parameters
        self.auto_params = True 

        # expected spot size
        self.lab_spot_size_xy = QLabel('Expected spot size xy (px)')
        self.txt_spot_size_xy = QLineEdit()
        self.txt_spot_size_xy.setText('5')
        self.lab_spot_size_z = QLabel('Expected spot size z (px)')
        self.txt_spot_size_z = QLineEdit()
        self.txt_spot_size_z.setText('5')
        self.lab_sigma_ratio = QLabel('DoG sigma ratio big / small')
        self.txt_sigma_ratio = QLineEdit()
        self.txt_sigma_ratio.setText('1.6')
        self.but_auto_sigmas = QPushButton()
        self.but_auto_sigmas.setText('Auto sigmas')
        self.but_auto_sigmas.clicked.connect(self._make_sigmas)

        # DoG blob detection widgets
        self.sld_sigma_xy_small = FullSlider(range=(0.1, 20), step=0.1, label="sigma xy small")
        self.sld_sigma_xy_large = FullSlider(range=(0.1, 20), step=0.1, label="sigma xy large")
        self.sld_sigma_z_small = FullSlider(range=(0.1, 20), step=0.1, label="sigma z small")
        self.sld_sigma_z_large = FullSlider(range=(0.1, 20), step=0.1, label="sigma z large")

        # self.sld_blob_thresh = FullSlider(range=(0.1, 20), step=0.1, label="Blob threshold")
        self.lab_blob_thresh = QLabel('Blob threshold')
        self.sld_blob_thresh = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.sld_blob_thresh.setRange(0, 50)
        self.sld_blob_thresh.setValue(10)
        # self.sld_blob_thresh.setBarIsRigid(False) not implemented for QLabeledDoubleSlider :'(

        self.but_dog = QPushButton()
        self.but_dog.setText('Apply DoG')
        self.but_dog.clicked.connect(self._compute_dog)

        self.but_find_peaks = QPushButton()
        self.but_find_peaks.setText('Find peaks')
        self.but_find_peaks.clicked.connect(self._find_peaks)

        self.lab_merge_peaks = QLabel('Merge peaks distances (xy, z)')
        self.txt_merge_peaks_xy = QLineEdit()
        self.txt_merge_peaks_xy.setText('5')
        self.txt_merge_peaks_z = QLineEdit()
        self.txt_merge_peaks_z.setText('5')
        self.but_merge_peaks = QPushButton()
        self.but_merge_peaks.setText('Merge peaks')
        self.but_merge_peaks.clicked.connect(self._merge_peaks)

        # gaussian fitting widgets
        self.lab_roi_sizes = QLabel('Fit ROI sizes (xy, z)')
        self.txt_roi_size_xy = QLineEdit()
        self.txt_roi_size_xy.setText('10')
        self.txt_roi_size_z = QLineEdit()
        self.txt_roi_size_z.setText('10')
        
        self.lab_min_roi_sizes = QLabel('Minimum ROI sizes (xy, z)')
        self.txt_min_roi_size_xy = QLineEdit()
        self.txt_min_roi_size_xy.setText('5')
        self.txt_min_roi_size_z = QLineEdit()
        self.txt_min_roi_size_z.setText('5')

        self.but_auto_roi = QPushButton()
        self.but_auto_roi.setText('Auto ROI sizes')
        self.but_auto_roi.clicked.connect(self._make_roi_sizes)

        self.but_fit = QPushButton()
        self.but_fit.setText('Fit spots')
        self.but_fit.clicked.connect(self._fit_spots)

        self.but_plot_fitted = QPushButton()
        self.but_plot_fitted.setText('Plot fitted parameters')
        self.but_plot_fitted.clicked.connect(self._plot_fitted_params)
        self.but_plot_fitted_2D = QPushButton()
        self.but_plot_fitted_2D.setText('Plot 2D distributions')
        self.but_plot_fitted_2D.clicked.connect(self._plot_fitted_params_2D)

        # spot filtering widgets
        self.lab_filter_amplitude_range = QLabel('Range amplitude')
        self.sld_filter_amplitude_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.sld_filter_amplitude_range.setRange(1, 4)
        self.sld_filter_amplitude_range.setValue([2, 3])
        self.sld_filter_amplitude_range.setBarIsRigid(False)
        self.chk_filter_amplitude_min = QCheckBox()
        self.chk_filter_amplitude_max = QCheckBox()
        self.lab_filter_sigma_xy_range = QLabel('Range sigma x/y')
        self.sld_filter_sigma_xy_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.sld_filter_sigma_xy_range.setRange(1, 4)
        self.sld_filter_sigma_xy_range.setValue([2, 3])
        self.sld_filter_sigma_xy_range.setBarIsRigid(False)
        self.chk_filter_sigma_xy_min = QCheckBox()
        self.chk_filter_sigma_xy_max = QCheckBox()
        self.lab_filter_sigma_z_range = QLabel('Range sigma z')
        self.sld_filter_sigma_z_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.sld_filter_sigma_z_range.setRange(1, 4)
        self.sld_filter_sigma_z_range.setValue([2, 3])
        self.sld_filter_sigma_z_range.setBarIsRigid(False)
        self.chk_filter_sigma_z_min = QCheckBox()
        self.chk_filter_sigma_z_max = QCheckBox()
        self.lab_filter_sigma_ratio_range = QLabel('Range sigma ratio z/xy')
        self.sld_filter_sigma_ratio_range = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.sld_filter_sigma_ratio_range.setRange(1, 4)
        self.sld_filter_sigma_ratio_range.setValue([2, 3])
        self.sld_filter_sigma_ratio_range.setBarIsRigid(False)
        self.chk_filter_sigma_ratio_min = QCheckBox()
        self.chk_filter_sigma_ratio_max = QCheckBox()
        self.lab_filter_chi_squared = QLabel('min chi squared')
        self.sld_filter_chi_squared = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.sld_filter_chi_squared.setRange(1, 3)
        self.sld_filter_chi_squared.setValue(2)
        self.chk_filter_chi_squared = QCheckBox()
        self.lab_filter_dist_center = QLabel('distance to center')
        self.sld_filter_dist_center = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.sld_filter_dist_center.setRange(1, 3)
        self.sld_filter_dist_center.setValue(2)
        self.chk_filter_dist_center = QCheckBox()
        self.but_filter = QPushButton()
        self.but_filter.setText('Filter spots')
        self.but_filter.clicked.connect(self._filter_spots)

        self.but_save_spots = QPushButton()
        self.but_save_spots.setText('Save spots')
        self.but_save_spots.clicked.connect(self._save_spots)
        self.but_load_spots = QPushButton()
        self.but_load_spots.setText('Load spots')
        self.but_load_spots.clicked.connect(self._load_spots)
        self.but_save_parameters = QPushButton()
        self.but_save_parameters.setText('Save detection parameters')
        self.but_save_parameters.clicked.connect(self._save_parameters)
        self.but_load_parameters = QPushButton()
        self.but_load_parameters.setText('Load detection parameters')
        self.but_load_parameters.clicked.connect(self._load_parameters)


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
        blobThreshLayout = QHBoxLayout()
        blobThreshLayout.addWidget(self.lab_blob_thresh)
        blobThreshLayout.addWidget(self.sld_blob_thresh)
        dogLayout.addLayout(blobThreshLayout)
        dogLayout.addWidget(self.but_find_peaks)
        mergePeaksLayout = QHBoxLayout()
        mergePeaksLayout.addWidget(self.lab_merge_peaks)
        mergePeaksLayout.addWidget(self.txt_merge_peaks_xy)
        mergePeaksLayout.addWidget(self.txt_merge_peaks_z)
        dogLayout.addLayout(mergePeaksLayout)
        dogLayout.addWidget(self.but_merge_peaks)
        # layout for fitting gaussian spots
        fitLayout = QVBoxLayout()
        roisizesLayout = QHBoxLayout()
        roisizesLayout.addWidget(self.lab_roi_sizes)
        roisizesLayout.addWidget(self.txt_roi_size_xy)
        roisizesLayout.addWidget(self.txt_roi_size_z)
        minroisizesLayout = QHBoxLayout()
        minroisizesLayout.addWidget(self.lab_min_roi_sizes)
        minroisizesLayout.addWidget(self.txt_min_roi_size_xy)
        minroisizesLayout.addWidget(self.txt_min_roi_size_z)
        fitLayout.addLayout(roisizesLayout)
        fitLayout.addLayout(minroisizesLayout)
        fitLayout.addWidget(self.but_auto_roi)
        fitLayout.addWidget(self.but_fit)
        plotFittedLayout = QHBoxLayout()
        plotFittedLayout.addWidget(self.but_plot_fitted)
        plotFittedLayout.addWidget(self.but_plot_fitted_2D)
        fitLayout.addLayout(plotFittedLayout)

        # layout for filtering gaussian spots
        filterLayout = QGridLayout()
        # amplitudes
        filterLayout.addWidget(self.lab_filter_amplitude_range, 0, 0)
        filterLayout.addWidget(self.sld_filter_amplitude_range, 0, 1)
        chk_layout = QHBoxLayout()
        chk_layout.addWidget(self.chk_filter_amplitude_min)
        chk_layout.addWidget(self.chk_filter_amplitude_max)
        filterLayout.addLayout(chk_layout, 0, 2)
        # sigma xy
        filterLayout.addWidget(self.lab_filter_sigma_xy_range, 1, 0)
        filterLayout.addWidget(self.sld_filter_sigma_xy_range, 1, 1)
        chk_layout = QHBoxLayout()
        chk_layout.addWidget(self.chk_filter_sigma_xy_min)
        chk_layout.addWidget(self.chk_filter_sigma_xy_max)
        filterLayout.addLayout(chk_layout, 1, 2)
        # sigma z
        filterLayout.addWidget(self.lab_filter_sigma_z_range, 2, 0)
        filterLayout.addWidget(self.sld_filter_sigma_z_range, 2, 1)
        chk_layout = QHBoxLayout()
        chk_layout.addWidget(self.chk_filter_sigma_z_min)
        chk_layout.addWidget(self.chk_filter_sigma_z_max)
        filterLayout.addLayout(chk_layout, 2, 2)
        # sigma ratio z/xy
        filterLayout.addWidget(self.lab_filter_sigma_ratio_range, 3, 0)
        filterLayout.addWidget(self.sld_filter_sigma_ratio_range, 3, 1)
        chk_layout = QHBoxLayout()
        chk_layout.addWidget(self.chk_filter_sigma_ratio_min)
        chk_layout.addWidget(self.chk_filter_sigma_ratio_max)
        filterLayout.addLayout(chk_layout, 3, 2)
        # chi squared
        filterLayout.addWidget(self.lab_filter_chi_squared, 4, 0)
        filterLayout.addWidget(self.sld_filter_chi_squared, 4, 1)
        filterLayout.addWidget(self.chk_filter_chi_squared, 4, 2)
        # distance to center
        filterLayout.addWidget(self.lab_filter_dist_center, 5, 0)
        filterLayout.addWidget(self.sld_filter_dist_center, 5, 1)
        filterLayout.addWidget(self.chk_filter_dist_center, 5, 2)
        filterLayout.addWidget(self.but_filter, 6, 1)

        # layout for saving and loading spots data and detection parameters
        saveloadLayout = QGridLayout()
        saveloadLayout.addWidget(self.but_save_spots, 0, 0)
        saveloadLayout.addWidget(self.but_load_spots, 0, 1)
        saveloadLayout.addWidget(self.but_save_parameters, 1, 0)
        saveloadLayout.addWidget(self.but_load_parameters, 1, 1)

        outerLayout.addLayout(spotsizeLayout)
        outerLayout.addLayout(dogLayout)
        outerLayout.addLayout(fitLayout)
        outerLayout.addLayout(filterLayout)
        outerLayout.addLayout(saveloadLayout)

        self.setLayout(outerLayout)


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
        sigma_xy_small = sigma_xy / sigma_ratio**(1/2)
        sigma_xy_large = sigma_xy * sigma_ratio**(1/2)
        sigma_z_small = sigma_z / sigma_ratio**(1/2)
        sigma_z_large = sigma_z * sigma_ratio**(1/2)
        self.sld_sigma_xy_small.setValue(sigma_xy_small)
        self.sld_sigma_xy_large.setValue(sigma_xy_large)
        self.sld_sigma_z_small.setValue(sigma_z_small)
        self.sld_sigma_z_large.setValue(sigma_z_large)
        self.sigma_xy = sigma_xy
        self.sigma_z = sigma_z


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

            # analyze specific image if selected, or first available one if no selection
            if len(self.viewer.layers.selection) == 0:
                img = self.viewer.layers[0].data
            else:
                # selection is a set, we need some wrangle to get the first element
                first_selected_layer = next(iter(self.viewer.layers.selection))
                img = first_selected_layer.data
            
            img_high_pass = localize.filter_convolve(img, kernel_small, use_gpu=False)
            img_low_pass = localize.filter_convolve(img, kernel_large, use_gpu=False)
            img_filtered = img_high_pass - img_low_pass
            if 'filtered' not in self.viewer.layers:
                self.viewer.add_image(img_filtered, name='filtered')
            else:
                self.viewer.layers['filtered'].data = img_filtered
            # basic auto thresholding
            if self.auto_params:
                blob_thresh = np.percentile(img_filtered.ravel(), 95)
                self.sld_blob_thresh.setRange(np.percentile(img_filtered.ravel(), 10), 
                                              img_filtered.max())
                self.sld_blob_thresh.setValue(blob_thresh)
    

    def _find_peaks(self):
        """
        Threshold the image resulting from the DoG filter and detect peaks.
        """
        if 'filtered' not in self.viewer.layers:
            print("Run a DoG filter on an image first")
        else:
            blob_thresh = self.sld_blob_thresh.value()
            img_filtered = self.viewer.layers['filtered'].data
            img_filtered[img_filtered < blob_thresh] = 0

            sx = sy = float(self.txt_spot_size_xy.text())
            sz = float(self.txt_spot_size_z.text())
            min_separations = np.array([sz, sy, sx]).astype(int)

            footprint = localize.get_max_filter_footprint(min_separations=min_separations, drs=(1,1,1))
            # array of size nz, ny, nx of True

            maxis = ndi.maximum_filter(img_filtered, footprint=np.ones(min_separations))
            self.centers_guess_inds, self.amps = localize.find_peak_candidates(img_filtered, footprint, threshold=blob_thresh, use_gpu_filter=False)
            if 'local maxis' not in self.viewer.layers:
                self.viewer.add_points(self.centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
            else:
                self.viewer.layers['local maxis'].data = self.centers_guess_inds
            if self.auto_params:
                self._make_roi_sizes()
            self.peaks_layer_name = 'local maxis'
            # used variables for gaussian fit if peaks are not merged
            self.use_centers_inds = self.centers_guess_inds
            self.use_amps = self.amps
            self.peaks_merged = False


    def _merge_peaks(self):
        """
        Merge peaks that are close to each other.
        """

        if 'local maxis' not in self.viewer.layers:
            print("Find peaks first")
        else:
            coords = self.centers_guess_inds
            max_xy = float(self.txt_merge_peaks_xy.text())
            max_z = float(self.txt_merge_peaks_z.text())

            # network based version
            # weight_img = self.viewer.layers['filtered'].data # or use raw data: self.viewer.layers[0].data
            # self.merged_centers_inds = ip.filter_nearby_peaks(coords, max_z, max_xy, weight_img)
            # # need ravel_multi_index to get pixel values of weight_img at several 3D coordinates
            # amplitudes_id = np.ravel_multi_index(self.merged_centers_inds.astype(int).transpose(), weight_img.shape)
            # self.amps = weight_img.ravel()[amplitudes_id]

            # Peter's version
            self.merged_centers_inds, inds_comb = localize.filter_nearby_peaks(coords, max_xy, max_z, weights=self.amps, mode="average")
            self.merged_amps = self.amps[inds_comb]
            nb_points = len(self.merged_centers_inds)
            print(f"Found {nb_points} points separated by dxy > {max_xy} and dz > {max_z}")

            if 'merged maxis' not in self.viewer.layers:
                self.viewer.add_points(self.merged_centers_inds, name='merged maxis', 
                                       blending='additive', size=3, face_color='g')
            else:
                self.viewer.layers['merged maxis'].data = self.merged_centers_inds
            # used variables for gaussian fit now that peaks are merged
            self.peaks_layer_name = 'merged maxis'
            self.use_centers_inds = self.merged_centers_inds
            self.use_amps = self.merged_amps
            self.peaks_merged = True


    def _make_roi_sizes(self):
        """
        Compute the x/y and z sizes of ROIs to fit gaussians to spots.
        """

        sx = sy = float(self.txt_spot_size_xy.text())
        sz = float(self.txt_spot_size_z.text())
        fit_roi_sizes = (2.355 * np.array([sz, sy, sx])).astype(int)
        min_fit_roi_sizes = fit_roi_sizes * 0.5
        self.fit_roi_sizes = fit_roi_sizes
        self.min_fit_roi_sizes = min_fit_roi_sizes

        self.txt_roi_size_xy.setText(str(fit_roi_sizes[-1]))
        self.txt_roi_size_z.setText(str(fit_roi_sizes[0]))
        self.txt_min_roi_size_xy.setText(str(min_fit_roi_sizes[-1]))
        self.txt_min_roi_size_z.setText(str(min_fit_roi_sizes[0]))
    

    def get_roi_coordinates(self, centers, sizes, max_coords_val, min_sizes, return_sizes=True):
        """
        Make pairs of (z, y, x) coordinates defining an ROI.
        
        Parameters
        ----------
        centers : ndarray, dtype int
            Centers of future ROIs, a Nx3 array.
        sizes : array or list
            Size of ROIs in each dimensions.
        max_coords_val : array or list
            Maximum value of coordinates in each dimension,
            typically the original image shape - 1.
        min_sizes : array or list
            Minimum size of ROIs in each dimension.
        
        Returns
        -------
        roi_coords : ndarray
            Pairs of point coordinates, a 2xNx3 array.
        roi_coords : ndarray
            Shape of each ROI, Nx3 array.
        """
        
        # make raw coordinates
        min_coords = centers - sizes / 2
        max_coords = centers + sizes / 2
        coords = np.stack([min_coords, max_coords]).astype(int)
        # clean min and max values of coordinates
        coords[coords < 0] = 0
        for i in range(3):
            coords[1, coords[1, :, i] > max_coords_val[i], i] = max_coords_val[i]
        # delete small ROIs
        roi_sizes = coords[1, :, :] - coords[0, :, :]
        select = ~np.any([roi_sizes[:, i] <= min_sizes[i] for i in range(3)], axis=0)
        coords = coords[:, select, :]
        # swap axes for latter convenience
        roi_coords = np.swapaxes(coords, 0, 1)
        
        if return_sizes:
            roi_sizes = roi_sizes[select, :]
            return roi_coords, roi_sizes
        else:
            return roi_coords

    
    def extract_ROI(self, img, coords):
        """
        Extract a portion of an image given by the coordinates of 2 points.
        
        Parameters
        ----------
        img : ndarray, dimension 3
            The i;age from which the ROI is extracted.
        coords : ndarry, shape (2, 3)
            The 2 coordinates of the 3 dimensional points at the corner of the ROI.
        
        Returns
        -------
        roi : ndarray
            A region of interest of the original image.
        """
        
        z0, y0, x0 = coords[0]
        z1, y1, x1 = coords[1]
        roi = img[z0:z1, y0:y1, x0:x1]
        return roi

    
    def _fit_spots(self):
        """
        Perform a gaussian fitting on each ROI.
        """

        roi_size_xy = int(float(self.txt_roi_size_xy.text()))
        roi_size_z = int(float(self.txt_roi_size_z.text()))
        min_roi_size_xy = int(float(self.txt_min_roi_size_xy.text()))
        min_roi_size_z = int(float(self.txt_min_roi_size_z.text()))
        img = self.viewer.layers[0].data

        fit_roi_sizes = np.array([roi_size_z, roi_size_xy, roi_size_xy])
        min_fit_roi_sizes = np.array([min_roi_size_z, min_roi_size_xy, min_roi_size_xy])

        roi_coords, roi_sizes = self.get_roi_coordinates(
            centers = self.use_centers_inds,  # first detected peaks or merged peaks
            sizes = fit_roi_sizes, 
            max_coords_val = np.array(img.shape) - 1, 
            min_sizes = min_fit_roi_sizes,
        )
        nb_rois = roi_coords.shape[0]
        # centers_guess is different from centers_guess_inds because each ROI 
        # can be smaller on the borders of the image
        centers_guess = (roi_sizes / 2)


        # actually fitting
        all_res = []
        chi_squared = []
        # all_init_params = []
        for i in range(nb_rois):
            # extract ROI
            roi = self.extract_ROI(img, roi_coords[i])
            # fit gaussian in ROI
            init_params = np.array([
                self.use_amps[i], 
                centers_guess[i, 2],
                centers_guess[i, 1],
                centers_guess[i, 0],
                self.sigma_xy, 
                self.sigma_z, 
                roi.min(),
            ])
            fit_results = localize.fit_gauss_roi(
                roi, 
                (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
                init_params,
                fixed_params=np.full_like(init_params, False),
            )
            chi_squared.append(fit_results['chi_squared'])
            all_res.append(fit_results['fit_params'])

        # process all the results
        all_res = np.array(all_res)
        self.amplitudes = all_res[:, 0]
        centers = all_res[:, 3:0:-1]
        self.sigmas_xy = all_res[:, 4]
        self.sigmas_z = all_res[:, 5]
        self.offsets = all_res[:, 6]
        self.chi_squared = np.array(chi_squared)
        # distances from initial guess
        self.dist_center = np.sqrt(np.sum((centers - centers_guess)**2, axis=1))
        # add origin coordinates of each ROI
        self.centers = centers + roi_coords[:, 0, :]
        # composed variables for filtering
        self.diff_amplitudes = self.amplitudes - self.offsets
        self.sigma_ratios = self.sigmas_z / self.sigmas_xy

        # update range of filters
        p_mini = 1
        p_maxi = 99
        if self.auto_params:
            self.sld_filter_amplitude_range.setRange(np.percentile(self.amplitudes, p_mini), np.percentile(self.amplitudes, p_maxi))
            self.sld_filter_sigma_xy_range.setRange(np.percentile(self.sigmas_xy, p_mini), np.percentile(self.sigmas_xy, p_maxi))
            self.sld_filter_sigma_z_range.setRange(np.percentile(self.sigmas_z, p_mini), np.percentile(self.sigmas_z, p_maxi))
            self.sld_filter_sigma_ratio_range.setRange(np.percentile(self.sigma_ratios, p_mini), np.percentile(self.sigma_ratios, p_maxi))
            self.sld_filter_chi_squared.setRange(np.percentile(self.chi_squared, p_mini), np.percentile(self.chi_squared, p_maxi))
            self.sld_filter_dist_center.setRange(np.percentile(self.dist_center, p_mini), np.percentile(self.dist_center, p_maxi))
    
        if 'fitted spots' not in self.viewer.layers:
            self.viewer.add_points(self.centers, name='fitted spots', blending='additive', size=3, face_color='g')
        else:
            self.viewer.layers['fitted spots'].data = self.centers
    

    def _plot_fitted_params(self):
        """
        Generate distribution plots of fitted parameters to help selecting
        appropriate threshold values for spot filtering.
        """

        p_mini = 1
        p_maxi = 99

        plt.figure()
        plt.hist(self.amplitudes, bins='auto', range=self.sld_filter_amplitude_range.value())
        plt.title("Distribution of amplitude values")

        plt.figure()
        plt.hist(self.sigmas_xy, bins='auto', range=self.sld_filter_sigma_xy_range.value())
        plt.title("Distribution of sigmas_xy values")

        plt.figure()
        plt.hist(self.sigmas_z, bins='auto', range=self.sld_filter_sigma_z_range.value())
        plt.title("Distribution of sigmas_z values")

        plt.figure()
        plt.hist(self.sigma_ratios, bins='auto', range=self.sld_filter_sigma_ratio_range.value())
        plt.title("Distribution of sigma_ratios values")

        plt.figure()
        plt.hist(self.chi_squared, bins='auto', range=(np.percentile(self.chi_squared, p_mini), 
                                                       np.percentile(self.chi_squared, p_maxi)))
        plt.title("Distribution of chi_squared values")

        plt.figure()
        plt.hist(self.dist_center, bins='auto', range=(np.percentile(self.dist_center, p_mini), 
                                                       np.percentile(self.dist_center, p_maxi)))
        plt.title("Distribution of dist_center values")

        plt.show()
    

    def _plot_fitted_params_2D(self):
        """
        Generate 2D distribution plots of fitted parameters to help selecting
        appropriate threshold values for spot filtering.
        """

        p_mini = 1
        p_maxi = 99

        distrib = {
            'amplitudes': {'data': self.amplitudes, 
                           'range': self.sld_filter_amplitude_range.value()},
            'sigmas_xy': {'data': self.sigmas_xy, 
                          'range': self.sld_filter_sigma_xy_range.value()},
            'sigmas_z': {'data': self.sigmas_z, 
                         'range': self.sld_filter_sigma_z_range.value()},
            'sigma_ratios': {'data': self.sigma_ratios, 
                             'range': self.sld_filter_sigma_ratio_range.value()},
            'chi_squared': {'data': self.chi_squared, 
                            'range': (np.percentile(self.chi_squared, p_mini), 
                                      np.percentile(self.chi_squared, p_maxi))},
            'dist_center': {'data': self.dist_center, 
                            'range': (np.percentile(self.dist_center, p_mini), 
                                      np.percentile(self.dist_center, p_maxi))},
        }
        
        var_labels = list(distrib.keys())
        var_combi = itertools.combinations(var_labels, 2)
        for var_x, var_y in var_combi:
            x_mini, x_maxi = distrib[var_x]['range']
            y_mini, y_maxi = distrib[var_y]['range']
            x_select = np.logical_and(distrib[var_x]['data'] >= x_mini, distrib[var_x]['data'] <= x_maxi)
            y_select = np.logical_and(distrib[var_y]['data'] >= y_mini, distrib[var_y]['data'] <= y_maxi)
            select = np.logical_and(x_select, y_select)
            x_data = distrib[var_x]['data'][select]
            y_data = distrib[var_y]['data'][select]

            plt.figure()
            plt.scatter(x_data, y_data, s=10, marker='.', c='b', alpha=0.5)
            plt.title(f"Distributions of {var_x} and {var_y}")
            plt.xlabel(var_x)
            plt.ylabel(var_y)
            plt.show()


    def _filter_spots(self):
        """
        Filter out spots based on gaussian fit results.
        """

        # list of boolean filters for all spots thresholds
        selectors = []
        
        if self.chk_filter_amplitude_min.isChecked():
            selectors.append(self.amplitudes >= self.sld_filter_amplitude_range.value()[0])
        if self.chk_filter_amplitude_max.isChecked():
            selectors.append(self.amplitudes <= self.sld_filter_amplitude_range.value()[1])
        if self.chk_filter_sigma_xy_min.isChecked():
            selectors.append(self.sigmas_xy >= self.sld_filter_sigma_xy_range.value()[0])
        if self.chk_filter_sigma_xy_max.isChecked():
            selectors.append(self.sigmas_xy <= self.sld_filter_sigma_xy_range.value()[1])
        if self.chk_filter_sigma_z_min.isChecked():
            selectors.append(self.sigmas_z >= self.sld_filter_sigma_z_range.value()[0])
        if self.chk_filter_sigma_z_max.isChecked():
            selectors.append(self.sigmas_z <= self.sld_filter_sigma_z_range.value()[1])
        if self.chk_filter_sigma_ratio_min.isChecked():
            selectors.append(self.sigma_ratios >= self.sld_filter_sigma_ratio_range.value()[0])
        if self.chk_filter_sigma_ratio_max.isChecked():
            selectors.append(self.sigma_ratios <= self.sld_filter_sigma_ratio_range.value()[1])
        if self.chk_filter_chi_squared.isChecked():
            selectors.append(self.chi_squared >= self.sld_filter_chi_squared.value())
        if self.chk_filter_dist_center.isChecked():
            selectors.append(self.dist_center <= self.sld_filter_dist_center.value())

        if len(selectors) == 0:
            print("Check at list one box to activate filters")
        else:
            self.spot_select = np.logical_and.reduce(selectors)
        
            # self.viewer.layers['filtered spots'].data = self.centers[self.spot_select] doesn't work
            if 'filtered spots' in self.viewer.layers:
                del self.viewer.layers['filtered spots']
            self.viewer.add_points(self.centers[self.spot_select], name='filtered spots', blending='additive', size=3, face_color='b')
            

    def _save_spots(self):

            # save the results
            if not hasattr(self, 'spot_select'):
                self.spot_select = np.full(len(self.centers), np.nan)
            if self.centers.shape[1] == 3:
                df_spots = pd.DataFrame({
                    'amplitudes': self.amplitudes,
                    'z': self.centers[:,0],
                    'y': self.centers[:,1],
                    'x': self.centers[:,2],
                    'sigmas_xy': self.sigmas_xy,
                    'sigmas_z': self.sigmas_z,
                    'offsets': self.offsets,
                    'chi_squareds': self.chi_squared,
                    'dist_center': self.dist_center,
                    'spot_select': self.spot_select,
                })
            else:
                df_spots = pd.DataFrame({
                    'amplitudes': self.amplitudes,
                    'y': self.centers[:,0],
                    'x': self.centers[:,1],
                    'sigmas_xy': self.sigmas_xy,
                    'offsets': self.offsets,
                    'chi_squareds': self.chi_squared,
                    'dist_center': self.dist_center,
                    'spot_select': self.spot_select,
                })

            path_save = QFileDialog.getSaveFileName(self, 'Export spots data')[0]
            if path_save != '':
                if not path_save.endswith('.csv'):
                    path_save = path_save + '.csv'
                df_spots.to_csv(path_save, index=False)


    def _load_spots(self):

        path_load = QFileDialog.getOpenFileName(self,"Load spots data","","CSV Files (*.csv);;All Files (*)")[0]
        if path_load != '':
            df_spots = pd.read_csv(path_load)
            self.amplitudes = df_spots['amplitudes']
            self.sigmas_xy = df_spots['sigmas_xy']
            self.offsets = df_spots['offsets']
            self.chi_squareds = df_spots['chi_squareds']
            self.dist_center = df_spots['dist_center']
            self.spot_select = df_spots['spot_select']
            if 'z' in df_spots.columns:
                self.centers = np.zeros((len(df_spots['x']), 3))
                self.centers[:, 0] = df_spots['z']
                self.centers[:, 1] = df_spots['y']
                self.centers[:, 2] = df_spots['x']
                self.sigmas_z = df_spots['sigmas_z']
                self.sigma_ratios = self.sigmas_z / self.sigmas_xy
            else:
                self.centers = np.zeros((len(df_spots['x']), 2))
                self.centers[:, 0] = df_spots['y']
                self.centers[:, 1] = df_spots['x']
            
            if 'fitted spots' in self.viewer.layers:
                del self.viewer.layers['fitted spots']
            self.viewer.add_points(self.centers, name='fitted spots', blending='additive', size=3, face_color='g')
            # display filtered spots if there was a filtering
            if ~np.all(self.spot_select.isna()):
                if 'filtered spots' in self.viewer.layers:
                    del self.viewer.layers['filtered spots']
                self.viewer.add_points(self.centers[self.spot_select], name='filtered spots', blending='additive', size=3, face_color='b')
            

    def _save_parameters(self):
        detection_parameters = {
            'txt_spot_size_z': float(self.txt_spot_size_z.text()),
            'txt_spot_size_xy': float(self.txt_spot_size_xy.text()),
            'txt_sigma_ratio': float(self.txt_sigma_ratio.text()),
            'sigma_z': self.sigma_z,
            'sigma_xy': self.sigma_xy,
            'sld_sigma_z_small': self.sld_sigma_z_small.value,
            'sld_sigma_xy_small': self.sld_sigma_xy_small.value,
            'sld_sigma_z_large': self.sld_sigma_z_large.value,
            'sld_sigma_xy_large': self.sld_sigma_xy_large.value,
            'sld_blob_thresh': self.sld_blob_thresh.value(),
            'peaks_merged': self.peaks_merged,
            'txt_merge_peaks_xy': self.txt_merge_peaks_xy.text(),
            'txt_merge_peaks_z': self.txt_merge_peaks_z.text(),
            'txt_roi_size_z': float(self.txt_roi_size_z.text()),
            'txt_roi_size_xy': float(self.txt_roi_size_xy.text()),
            'txt_min_roi_size_z': float(self.txt_min_roi_size_z.text()),
            'txt_min_roi_size_xy': float(self.txt_min_roi_size_xy.text()),
            'chk_filter_amplitude_min': self.chk_filter_amplitude_min.isChecked(),
            'chk_filter_amplitude_max': self.chk_filter_amplitude_max.isChecked(),
            'sld_filter_amplitude_range': self.sld_filter_amplitude_range.value(),
            'chk_filter_sigma_xy_min': self.chk_filter_sigma_xy_min.isChecked(),
            'chk_filter_sigma_xy_max': self.chk_filter_sigma_xy_max.isChecked(),
            'sld_filter_sigma_xy_range': self.sld_filter_sigma_xy_range.value(),
            'chk_filter_sigma_z_min': self.chk_filter_sigma_z_min.isChecked(),
            'chk_filter_sigma_z_max': self.chk_filter_sigma_z_max.isChecked(),
            'sld_filter_sigma_z_range': self.sld_filter_sigma_z_range.value(),
            'chk_filter_sigma_ratio_min': self.chk_filter_sigma_ratio_min.isChecked(),
            'chk_filter_sigma_ratio_max': self.chk_filter_sigma_ratio_max.isChecked(),
            'sld_filter_sigma_ratio_range': self.sld_filter_sigma_ratio_range.value(),
            'chk_filter_chi_squared': self.chk_filter_chi_squared.isChecked(),
            'sld_filter_chi_squared': self.sld_filter_chi_squared.value(),
            'chk_filter_dist_center': self.chk_filter_dist_center.isChecked(),
            'sld_filter_dist_center': self.sld_filter_dist_center.value(),
        }

        path_save = QFileDialog.getSaveFileName(self, 'Export detection parameters')[0]
        if path_save != '':
            if not path_save.endswith('.json'):
                path_save = path_save + '.json'
            with open(path_save, "w") as write_file:
                json.dump(detection_parameters, write_file, indent=4)


    def _load_parameters(self):

        path_load = QFileDialog.getOpenFileName(self,"Load spots data","","JSON Files (*.json);;All Files (*)")[0]
        if path_load != '':
            # deactivate automatic parameters update during pipeline execution
            self.auto_params = False
            with open(path_load, "r") as read_file:
                detection_parameters = json.load(read_file)
            
            # parse all keys and modify the widgets' values
            self.txt_spot_size_z.setText(str(detection_parameters['txt_spot_size_z']))
            self.txt_spot_size_xy.setText(str(detection_parameters['txt_spot_size_xy']))
            self.txt_sigma_ratio.setText(str(detection_parameters['txt_sigma_ratio']))
            self.sld_sigma_z_small.setValue(detection_parameters['sld_sigma_z_small'])
            self.sld_sigma_xy_small.setValue(detection_parameters['sld_sigma_xy_small'])
            self.sld_sigma_z_large.setValue(detection_parameters['sld_sigma_z_large'])
            self.sld_sigma_xy_large.setValue(detection_parameters['sld_sigma_xy_large'])
            self.sld_blob_thresh.setValue(detection_parameters['sld_blob_thresh'])
            self.peaks_merged = detection_parameters['peaks_merged']
            self.txt_merge_peaks_xy.setText(str(detection_parameters['txt_merge_peaks_xy']))
            self.txt_merge_peaks_z.setText(str(detection_parameters['txt_merge_peaks_z']))
            self.txt_roi_size_z.setText(str(detection_parameters['txt_roi_size_z']))
            self.txt_roi_size_xy.setText(str(detection_parameters['txt_roi_size_xy']))
            self.txt_min_roi_size_z.setText(str(detection_parameters['txt_min_roi_size_z']))
            self.txt_min_roi_size_xy.setText(str(detection_parameters['txt_min_roi_size_xy']))
            self.chk_filter_amplitude_min.setChecked(detection_parameters['chk_filter_amplitude_min'])
            self.chk_filter_amplitude_max.setChecked(detection_parameters['chk_filter_amplitude_max'])
            self.sld_filter_amplitude_range.setValue(detection_parameters['sld_filter_amplitude_range'])
            self.chk_filter_sigma_xy_min.setChecked(detection_parameters['chk_filter_sigma_xy_min'])
            self.chk_filter_sigma_xy_max.setChecked(detection_parameters['chk_filter_sigma_xy_max'])
            self.sld_filter_sigma_xy_range.setValue(detection_parameters['sld_filter_sigma_xy_range'])
            self.chk_filter_sigma_z_min.setChecked(detection_parameters['chk_filter_sigma_z_min'])
            self.chk_filter_sigma_z_max.setChecked(detection_parameters['chk_filter_sigma_z_max'])
            self.sld_filter_sigma_z_range.setValue(detection_parameters['sld_filter_sigma_z_range'])
            self.chk_filter_sigma_ratio_min.setChecked(detection_parameters['chk_filter_sigma_ratio_min'])
            self.chk_filter_sigma_ratio_max.setChecked(detection_parameters['chk_filter_sigma_ratio_max'])
            self.sld_filter_sigma_ratio_range.setValue(detection_parameters['sld_filter_sigma_ratio_range'])
            self.chk_filter_chi_squared.setChecked(detection_parameters['chk_filter_chi_squared'])
            self.sld_filter_chi_squared.setValue(detection_parameters['sld_filter_chi_squared'])
            self.chk_filter_dist_center.setChecked(detection_parameters['chk_filter_dist_center'])
            self.sld_filter_dist_center.setValue(detection_parameters['sld_filter_dist_center'])
            if 'sigma_z' in detection_parameters and 'sigma_xy' in detection_parameters:
                self.sigma_z = detection_parameters['sigma_z']
                self.sigma_xy = detection_parameters['sigma_xy']
            else:
                warnings.warn("sigma_xy or sigma_z was not found in loaded parameters, they will be recomputed, "\
                              "which may overwrite some parameters if they differed from autocomputed values", 
                               RuntimeWarning)
                self._make_sigmas()



if __name__ == "__main__":
    viewer = napari.Viewer()
    napari.run()

# TODO:
#   - all functions compatible with 2D data
#   - add tilted image parameters for OPM data
#   - add raw estimation of spots parameters by drawing box around one spot and estimating (fitting gaussian?) parameters
#   - add panel to select data from big images, random tile, specific channel, etc...
#   - add parallel computation (Tiler + Dask) fr big images
#   - add manual annotation and automatic training for filtering parameters
#   - add variable range for all sliders