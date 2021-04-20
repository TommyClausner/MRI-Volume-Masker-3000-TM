#!/usr/bin/env python3
"""Used to draw brain masks on MRI image data (nii.gz).

Usage:
    python vol2mask.py path/to/volume.nii.gz

Author:
    Tommy Clausner (2020) - tommy.clausner@gmail.com
"""

import argparse
import json
import os
import sys
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.segment.mask import median_otsu
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from scipy import ndimage


class GUI:
    """Graphical user interface (GUI) for Tommy's Volume Masker 3000 TM.
    """
    c_max = 0.7
    mask_alpha = 0.2
    mask_alpha_backup = None
    show_mask = True

    status_text = ''
    status_font = {'verticalalignment': 'bottom',
                   'horizontalalignment': 'left', 'size': 8}

    info_text = ''
    info_text_font = {'horizontalalignment': 'center', 'size': 16}

    cid = None

    ax_lims = None

    xys = []
    lasso = []
    ind = []

    def __init__(self):
        """Constructor method
        """

        self.status_font['color'] = config["info text color"]
        self.info_text_font['color'] = config["popup text color"]
        self.mask_cmap = config["mask colormap"]

        # make main window
        self.fig = plt.figure(facecolor=(0.22, 0.22, 0.22))
        self.fig.canvas.manager.set_window_title(
            "Tommy's MRI Volume Masker 3000 TM")
        self.figtitle = self.fig.text(0.025, 0.025, '',
                                      fontdict={'size': 8, 'color': 'w'})

        # divide into parts
        gs = self.fig.add_gridspec(2, 10)

        self.selected = controller.get_view_mask(controller.slice)

        # main drawing window
        self.main_ax = self.fig.add_subplot(gs[:, :7])
        self.main_ax.axis('off')
        self.main_img = self.main_ax.imshow(
            controller.get_view_data(controller.slice),
            config["data colormap"])
        self.mask_main_img = self.main_ax.imshow(
            controller.get_view_mask(controller.slice),
            self.mask_cmap)
        # upper right window
        self.upper_ax = self.fig.add_subplot(gs[:1, 7:])
        self.upper_ax.axis('off')
        self.upper_img = self.upper_ax.imshow(np.empty((2, 2)),
                                              config["data colormap"])
        self.mask_upper_img = self.upper_ax.imshow(np.empty((2, 2)),
                                                   self.mask_cmap)

        # lower right window
        self.lower_ax = self.fig.add_subplot(gs[1:, 7:])
        self.lower_ax.axis('off')
        self.lower_img = self.lower_ax.imshow(np.empty((2, 2)),
                                              config["data colormap"])
        self.mask_lower_img = self.lower_ax.imshow(np.empty((2, 2)),
                                                   self.mask_cmap)

        # info text object
        self.status = self.main_ax.set_title(self.status_text,
                                             fontdict=self.status_font,
                                             loc='left')

        # popup status change text
        self.popup_info = self.main_ax.text(0.5, 0.5, self.info_text,
                                            fontdict=self.info_text_font,
                                            transform=self.main_ax.transAxes)

        self.main_ax.autoscale(False)
        # customize figure toolbar
        try:
            for rm_tools in ['home', 'back', 'forward', 'subplots', 'save',
                             'help']:
                self.fig.canvas.manager.toolmanager.remove_tool(rm_tools)
        except AttributeError:
            warnings.warn("No zoom and pan tools available. "
                          "Try setting different backend in config",
                          UserWarning)
        self.main_ax.callbacks.connect('xlim_changed', self.on_xlims_change)
        self.main_ax.callbacks.connect('ylim_changed', self.on_ylims_change)

        self.filter = {
            'name': ['no filter', 'sobel', 'gauss'],
            'filter': [None,
                       ndimage.sobel,
                       ndimage.gaussian_filter],
            'args': [None,
                     [],
                     [2 * data.header.get_zooms()[0]]]
        }

        self.filter.update({'counter': [ind
                                        for ind, n in
                                        enumerate(self.filter['name'])
                                        if n == config['start filter']][0]})

        self.draw_mode = config['start draw mode']

        # disable cursor data
        self.main_ax.format_coord = lambda x, y: ''
        self.upper_ax.format_coord = lambda x, y: ''
        self.lower_ax.format_coord = lambda x, y: ''

        self.mask_main_img.format_cursor_data = lambda x: ''
        self.mask_upper_img.format_cursor_data = lambda x: ''
        self.mask_lower_img.format_cursor_data = lambda x: ''

    def update_img_extent(self, img):
        """Updates image objects by replacing their extent value to fit data in
        img.

        :param img: The new volume data (3D).
        :type img: ndarray
        """
        extent = [0,
                  img.shape[2],
                  img.shape[1],
                  0]
        self.main_img.set_extent(extent)

        self.mask_main_img.set_extent(extent)

        self.upper_img.set_extent([-0.5, img.shape[0] + 0.5,
                                   img.shape[2] + 0.5, -0.5])
        self.mask_upper_img.set_extent([-0.5, img.shape[0] + 0.5,
                                        img.shape[2] + 0.5, -0.5])

        self.lower_img.set_extent([-0.5, img.shape[0] + 0.5,
                                   img.shape[1] + 0.5, -0.5])
        self.mask_lower_img.set_extent([-0.5, img.shape[0] + 0.5,
                                        img.shape[1] + 0.5, -0.5])

    def update_axes_limits(self, ax_lims=None):
        """Updates axes limits to fit data.

        :param ax_lims: Resets view.
        :type ax_lims: None|list, optional
        """
        if ax_lims is None:
            self.main_ax.set_xlim(
                [0, controller.get_view_data(controller.slice).shape[1]])
            self.main_ax.set_ylim(
                [controller.get_view_data(controller.slice).shape[0], 0])

    def update_plots(self, new_data=None, new_mask=None, first_dim_ind=None):
        """Updates all plots.

        :param new_data: New volume data (3D) to update the plots.
        :type new_data: None|ndarray, optional
        :param new_mask: New volume mask (3D) to update the plots.
        :type new_mask: None|ndarray, optional
        :param first_dim_ind: Slice index to show. If None (default) the middle
            slice will be selected.
        :type first_dim_ind: None|int, optional
        :return:
        """
        if new_data is None:
            new_data = controller.get_view_data()

        if new_mask is None:
            new_mask = controller.get_view_mask()

        if first_dim_ind is None:
            first_dim_ind = controller.slice

        # function to actually set the data
        def _set_data(img, data, clim=None, alpha=None):
            img.set_data(data)
            if alpha is not None:
                img.set_alpha(alpha)
            if clim is not None:
                img.set_clim(clim)
            return img

        # apply filters and update axes limits
        img_filter = self.filter['filter'][
            self.filter['counter'] % len(self.filter['name'])]
        filter_args = self.filter['args'][
            self.filter['counter'] % len(self.filter['name'])]

        if img_filter is not None:
            new_data = img_filter(new_data, *filter_args)

        # central block with mask
        self.main_img = _set_data(self.main_img,
                                  new_data[first_dim_ind, :, :],
                                  (0, self.c_max))

        self.mask_main_img = _set_data(self.mask_main_img,
                                       self.selected.reshape(
                                           new_data[first_dim_ind, :, :].shape
                                       ),
                                       (0, 1),
                                       self.mask_alpha)
        self.update_axes_limits(self.ax_lims)
        self.update_img_extent(new_data)
        # upper right corner with mask
        self.upper_img = _set_data(self.upper_img,
                                   new_data[:,
                                   int(new_data.shape[1] / 2), :].T,
                                   (0, self.c_max))

        tmp_alpha = self.mask_alpha * np.ones_like(
            new_data[:, int(new_data.shape[1] / 2), :].T)
        tmp_alpha[:, first_dim_ind] = 1
        self.mask_upper_img = _set_data(self.mask_upper_img,
                                        new_mask[:, int(
                                            new_data.shape[1] / 2), :].T,
                                        (0, 1),
                                        tmp_alpha)

        # lower right corner with mask
        self.lower_img = _set_data(self.lower_img,
                                   new_data[:, :, int(
                                       new_data.shape[2] / 2)].T,
                                   (0, self.c_max))

        tmp_alpha = self.mask_alpha * np.ones_like(
            new_data[:, :, int(new_data.shape[2] / 2)].T)
        tmp_alpha[:, first_dim_ind] = 1
        self.mask_lower_img = _set_data(self.mask_lower_img,
                                        new_mask[:, :, int(
                                            new_data.shape[2] / 2)].T,
                                        (0, 1),
                                        tmp_alpha)

        # Update figure title text
        max_path = 69  # maximum last 69 characters of path is displayed
        len_path = len(data.volume_path)
        sel_chars = len_path - max_path
        if sel_chars < 0:
            sel_chars = 0
        self.status_text = 'file: '
        self.status_text += '' if len_path < max_path else '...'
        self.status_text += data.volume_path[sel_chars:]
        self.status_text += '\n\n'
        self.status_text += 'slice: {} | draw mode: {} | filter: {} | h help'
        self.status_text = self.status_text.format(
            controller.slice,
            self.draw_mode,
            self.filter['name'][self.filter['counter'] %
                                len(self.filter['name'])])
        self.status.set_text(self.status_text)
        self.fig.canvas.draw_idle()

    def update_popup_text(self, info, delay=0.):
        """Updates popup text on status change.

        :param info: What message to show.
        :type info: str
        :param delay: How long to show it (in seconds).
        :type delay: float, optional
        """
        self.popup_info.set_text(info)
        self.fig.canvas.draw_idle()
        if delay > 0:
            plt.pause(delay)
        self.popup_info.set_text('')
        self.fig.canvas.draw_idle()

    def on_xlims_change(self, _):
        """Updates axes limits on change (e.g. zoom). Wrapper function of
        :func:`vol2mask.GUI.on_ylims_change.`
        """
        self.on_ylims_change(self)

    def on_ylims_change(self, _):
        """Updates axes limits on change (e.g. zoom).
        """
        self.trigger_tool('zoom', if_up=True)
        self.ax_lims = [self.main_ax.get_xlim(), self.main_ax.get_ylim()]

    def trigger_tool(self, tool, if_up=True):
        """Sets state of tool.

        :param tool: Name of the tool to trigger.
        :type tool: str
        :param if_up: Whether to only trigger tool when is up (disable tool).
        :type if_up: bool
        """
        if self.fig.canvas.manager.toolmanager.get_tool(tool) is None:
            return
        if if_up:
            if self.fig.canvas.manager.toolmanager.get_tool(tool)._toggled:
                self.fig.canvas.manager.toolmanager.trigger_tool(tool)
        else:
            self.fig.canvas.manager.toolmanager.trigger_tool(tool)

    def binary_mask(self, binary_mask):
        cmap = config["mask colormap"] if not binary_mask else 'binary_r'
        self.mask_cmap = cmap
        self.mask_main_img.set_cmap(self.mask_cmap)
        self.mask_upper_img.set_cmap(self.mask_cmap)
        self.mask_lower_img.set_cmap(self.mask_cmap)
        if self.mask_alpha < 1:
            self.mask_alpha_backup = self.mask_alpha + 0
        self.mask_alpha = self.mask_alpha_backup if not binary_mask else 1

    def onselect(self, verts):
        """Callback function for lasso.
        """
        path = Path(verts)
        self.xy_compute()
        self.ind = path.contains_points(self.xys, radius=-1)
        self.selected.flat[self.ind] = 1 if self.draw_mode == 'add' else 0
        self.update_plots()

    def disconnect(self):
        """Remove lasso.
        """
        self.lasso.disconnect_events()
        self.update_plots()

    def xy_compute(self):
        """Transform image data into index data (for selection).
        """
        xv, yv = np.meshgrid(
            np.arange(controller.get_view_data(controller.slice).shape[1]),
            np.arange(controller.get_view_data(controller.slice).shape[0]))
        self.xys = np.vstack((xv.flatten(), yv.flatten())).T

    def reset_selection(self):
        """Reset lasso selection
        """
        self.xy_compute()
        self.ind = []
        self.selected = controller.get_view_mask(controller.slice).flatten()
        self.lasso = LassoSelector(self.main_ax, onselect=self.onselect)

    def connect(self, button_handler):
        """Connect :class:`vol2mask.GUI` to  :class:`vol2mask.Controller`
        """
        self.cid = self.fig.canvas.mpl_connect("key_release_event",
                                              button_handler)
        self.reset_selection()
        self.update_plots()


class Data:
    """Data used for Tommy's Volume Masker 3000 TM.

    :param volume_path: Full path to nifti file.
    :type volume_path: str
    :param make_mask: If make_mask='auto' (default), a brain mask is obtained
        by analyzing the image data. If 'none', no brain mask will be created.
        If path to nifti file, this file will be used as a mask.
    :type make_mask: str
    """
    mask = []
    volume = []
    affine = []
    header = []
    save_path = ''
    volume_path = ''

    def __init__(self, volume_path, make_mask='auto'):
        """Constructor method
        """
        self.load_data(volume_path)

        self.load_mask(make_mask)

    def load_data(self, path):
        self.volume_path = path
        # load volume and normalize
        print('reading nifti file...')
        path = os.path.dirname(self.volume_path)
        filename = os.path.basename(self.volume_path)
        img = nib.load(self.volume_path)
        print('Done.')
        self.volume = img.get_fdata()
        self.volume -= self.volume.min()
        self.volume /= self.volume.max()

        # in case of 4D volumes only extract spatial dims
        if self.volume.ndim > 3:
            self.volume = self.volume[:, :, :, 0]

        self.affine = img.affine.copy()
        self.header = img.header.copy()

        self.save_path = os.path.join(path, '_'.join(['m', filename]))

    def load_mask(self, make_mask):
        if make_mask is None:
            # empty mask
            self.mask = np.zeros_like(self.volume)
        else:
            # use dipy to isolate brain as an initial mask
            if make_mask.lower() == 'auto':
                print('creating brain mask...')
                _, self.mask = median_otsu(self.volume, median_radius=1,
                                           numpass=2)
                print('done.')
            elif make_mask.lower() == 'none':
                # empty mask
                self.mask = np.zeros_like(self.volume)
            else:
                print('reading mask file...')

                # same initial axes swap as for volume
                self.mask = nib.load(make_mask).get_fdata()
                print('done.')

    def export_mask(self):
        """Write mask to file.
        """

        # make nifti using information from volume and store
        print('saving...')
        out = nib.Nifti1Image(self.mask.astype(bool),
                              affine=self.affine, header=self.header)
        nib.save(out, self.save_path)


class Controller:
    """Controller for Tommy's Volume Masker 3000 TM.
    """
    binary_mask = False
    axes_swaps = [(2, 0)]

    def __init__(self):
        """Constructor method
        """
        self.slice = None if not config['start slice'] else config[
            'start slice']

        if self.slice is None:
            self.slice = int(data.volume.shape[2] / 2)

    def _swapaxes(self, raw_data, reversed_swap=False):
        if reversed_swap:
            for so in self.axes_swaps[::-1]:
                raw_data = raw_data.swapaxes(*so)
        else:
            for so in self.axes_swaps:
                raw_data = raw_data.swapaxes(*so)
        return raw_data

    def get_view_data(self, first_dim_ind=None):
        """Return 3D volume data.

        :param first_dim_ind: Which slice of the first dimension.
        :type: None|int, optional
        :return: 3D volume data.
        :rtype: ndarray
        """
        if first_dim_ind is not None:
            return self._swapaxes(data.volume)[first_dim_ind, :, :]
        return self._swapaxes(data.volume)

    def get_view_mask(self, first_dim_ind=None):
        """Return 3D mask data.

        :param first_dim_ind: Which slice of the first dimension.
        :type: None|int, optional
        :return: 3D mask data.
        :rtype: ndarray
        """
        if first_dim_ind is not None:
            return self._swapaxes(data.mask)[first_dim_ind, :, :]
        return self._swapaxes(data.mask)

    def button_handler(self, event):
        """Handles button presses
        """
        update = True
        reset = False

        # execute button specific function or just pass (update = False)
        if event.key == "h":

            print('\n################################################\n'
                  'Button mapping Tommy\'s MRI Volume Masker 3000 TM\n'
                  '################################################\n')
            for k, v in config['keyboard'].items():
                print(': '.join([k, v]))
            gui.update_popup_text('see console output for help...', 1)

        elif event.key == config['keyboard']['slice up']:

            self.slice += 1
            if self.slice >= self.get_view_data().shape[0]:
                self.slice = self.get_view_data().shape[0] - 1
            reset = True

        elif event.key == config['keyboard']['slice down']:

            self.slice -= 1
            if self.slice < 0:
                self.slice = 0
            reset = True

        elif event.key == config['keyboard']['switch view plane']:

            self.axes_swaps.append((0, 2))
            self.axes_swaps.append((1, 2))
            self.slice = int(self.get_view_data().shape[0] / 2)
            gui.ax_lims = None
            reset = True

        elif event.key == config['keyboard']['switch draw mode']:

            gui.draw_mode = 'remove' if gui.draw_mode == 'add' else 'add'
            gui.update_popup_text(gui.draw_mode, 0.25)

        elif event.key == config['keyboard']['export mask']:

            data.export_mask()
            gui.update_popup_text('Data successfully exported', 0.25)

        elif event.key == config['keyboard']['quit']:

            gui.update_popup_text('Later...', 0.25)
            plt.close(gui.fig)
            sys.exit()

        elif event.key == config['keyboard']['increase mask alpha']:

            gui.mask_alpha -= 0.05
            if gui.mask_alpha < 0:
                gui.mask_alpha = 0

        elif event.key == config['keyboard']['decrease mask alpha']:

            gui.mask_alpha += 0.05
            if gui.mask_alpha > 1:
                gui.mask_alpha = 1

        elif event.key == config['keyboard']['increase brightness']:

            gui.c_max -= 0.05
            if gui.c_max < 0:
                gui.c_max = 0

        elif event.key == config['keyboard']['decrease brightness']:

            gui.c_max += 0.05
            if gui.c_max > 1:
                gui.c_max = 1

        elif event.key == config['keyboard']['switch filter']:

            gui.filter['counter'] += 1
            gui.update_popup_text(
                gui.filter['name'][
                    gui.filter['counter'] % len(gui.filter['name'])], 0.25)

        elif event.key == config['keyboard']['binary mask view']:

            self.binary_mask = not self.binary_mask
            gui.binary_mask(self.binary_mask)

        elif event.key == config['keyboard']['enable / disable mask']:

            gui.show_mask = not gui.show_mask
            gui.mask_main_img.set_visible(gui.show_mask)
            gui.mask_upper_img.set_visible(gui.show_mask)
            gui.mask_lower_img.set_visible(gui.show_mask)

        elif event.key == config['keyboard']['set slice']:

            mask = self.get_view_mask()
            mask_slice = gui.selected.reshape(mask[self.slice, :, :].shape)
            mask[self.slice] = mask_slice
            data.mask = self._swapaxes(mask, reversed_swap=True)
            gui.disconnect()
            gui.update_popup_text('Slice set', 0.25)
            reset = True

        elif event.key == config['keyboard']['load file']:

            root = Tk()
            root.withdraw()
            fname = askopenfilename(
                title="Select (f)MRI image data",
                initialdir=os.path.dirname(os.path.abspath(data.volume_path)))
            root.destroy()
            if len(fname) > 0:
                data.load_data(fname)
                data.load_mask('auto')
                self.slice = None if not config['start slice'] else config[
                    'start slice']

                if self.slice is None:
                    self.slice = int(data.volume.shape[2] / 2)
                self.axes_swaps = [(2, 0)]
                reset = True
                gui.ax_lims = None

        elif event.key == config['keyboard']['load mask']:

            root = Tk()
            root.withdraw()
            fname = askopenfilename(
                title="Select (f)MRI image mask",
                initialdir=os.path.dirname(os.path.abspath(data.volume_path)))
            root.destroy()
            if len(fname) > 0:
                data.load_mask(fname)
                gui.selected = self.get_view_mask(self.slice)
                reset = True

        elif event.key == config['keyboard']['reset zoom']:

            gui.ax_lims = None

        else:

            update = False

        if reset:
            # reset selection if not set
            gui.reset_selection()

        # disable zoom and pan when updating window
        if update:

            # in case backend does not support tools
            try:
                gui.trigger_tool('zoom', if_up=True)
                gui.trigger_tool('pan', if_up=True)
            except Exception as e:
                print(e)
                warnings.warn("No zoom and pan tools available. "
                              "Try setting different backend in config",
                              UserWarning)
            gui.update_plots()


def main():
    """ Execute program
    """

    # variable definition on program level
    global data, gui, controller, config

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'config.json')

    # if config not found, create from default config (config_template.json)
    try:
        with open(config_path, 'r') as jf:
            config = json.load(jf)
        print('using existing config')
    except FileNotFoundError:
        with open(config_path.replace('config.json',
                                      'config_template.json'), 'r') as jf:
            config = json.load(jf)
        with open(config_path, 'w') as jf:
            json.dump(config, jf)
        print('using template config')

    if config['backend']:
        mpl.use(config['backend'])

    # setup pan and zoom tool
    plt.rcParams['toolbar'] = 'toolmanager'

    for k in plt.rcParams.keys():
        if 'keymap' in k:
            for v in plt.rcParams[k]:
                plt.rcParams[k].remove(v)

    plt.rcParams['keymap.zoom'].append(config['keyboard']['toggle zoom'])
    plt.rcParams['keymap.pan'].append(config['keyboard']['toggle pan'])

    # ensure maximum vertical real estate
    plt.rcParams['axes.ymargin'] = 0

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='nifti file',
                        default='dialog')
    parser.add_argument('-m', '--mask', type=str, help='initial mask file',
                        default='auto')
    args = parser.parse_args()

    if args.file == 'dialog':
        root = Tk()
        root.withdraw()
        args.file = askopenfilename(
            title="Select (f)MRI image data",
            initialdir=os.getcwd())
        root.destroy()

    # start program
    data = Data(args.file, args.mask)
    controller = Controller()
    gui = GUI()
    gui.connect(controller.button_handler)

    plt.show()


if __name__ == '__main__':
    # variable definition on module level
    global data, gui, controller, config
    main()
