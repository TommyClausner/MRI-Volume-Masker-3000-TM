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
from tkinter import Tk     # from tkinter import Tk for Python 3.x
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
    mask_cmap = 'RdYlGn'

    status_text = ''
    status_font = {'color': 'white', 'verticalalignment': 'center',
                   'horizontalalignment': 'left', 'size': 8}

    info_text = ''
    info_text_font = {'color': 'white', 'horizontalalignment': 'center',
                      'size': 16}

    cid = None

    ax_lims = None

    def __init__(self):
        """Constructor method
        """

        # make main window
        self.fig = plt.figure(facecolor=(0.22, 0.22, 0.22))
        self.fig.canvas.manager.set_window_title(
            "Tommy's MRI Volume Masker 3000 TM")
        self.figtitle = self.fig.text(0.025, 0.025, '',
                                      fontdict={'size': 8, 'color': 'w'})

        # divide into parts
        gs = self.fig.add_gridspec(2, 10)

        # main drawing window
        self.main_ax = self.fig.add_subplot(gs[:, :7])
        self.main_ax.axis('off')
        self.main_img = self.main_ax.imshow(data.get_data(data.slice), "gray")
        self.mask_main_img = self.main_ax.imshow(data.get_mask(data.slice),
                                                 'RdYlGn')
        # upper right window
        self.upper_ax = self.fig.add_subplot(gs[:1, 7:])
        self.upper_ax.axis('off')
        self.upper_img = self.upper_ax.imshow(np.empty((2, 2)), "gray")
        self.mask_upper_img = self.upper_ax.imshow(np.empty((2, 2)), 'RdYlGn')

        # lower right window
        self.lower_ax = self.fig.add_subplot(gs[1:, 7:])
        self.lower_ax.axis('off')
        self.lower_img = self.lower_ax.imshow(np.empty((2, 2)), "gray")
        self.mask_lower_img = self.lower_ax.imshow(np.empty((2, 2)), 'RdYlGn')

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
            self.main_ax.set_xlim([0, data.get_data(data.slice).shape[1]])
            self.main_ax.set_ylim([data.get_data(data.slice).shape[0], 0])

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
            new_data = data.get_data()

        if new_mask is None:
            new_mask = data.get_mask()

        if first_dim_ind is None:
            first_dim_ind = data.slice

        # function to actually set the data
        def _set_data(img, data, clim=None, alpha=None):
            img.set_data(data)
            if alpha is not None:
                img.set_alpha(alpha)
            if clim is not None:
                img.set_clim(clim)
            return img

        # apply filters and update axes limits
        img_filter = controller.filter['filter'][
            controller.filter['counter'] % len(controller.filter['name'])]
        filter_args = controller.filter['args'][
            controller.filter['counter'] % len(controller.filter['name'])]

        if img_filter is not None:
            new_data = img_filter(new_data, *filter_args)

        # central block with mask
        self.main_img = _set_data(self.main_img,
                                  new_data[first_dim_ind, :, :],
                                  (0, self.c_max))

        self.mask_main_img = _set_data(self.mask_main_img,
                                       controller.selected.reshape(
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
        self.status_text = 'slice: {} | draw mode: {} | filter: {} | h help'
        self.status_text = self.status_text.format(
            data.slice,
            controller.draw_mode,
            controller.filter['name'][controller.filter['counter'] %
                                      len(controller.filter['name'])])
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
        self.fig.canvas.draw()
        if delay > 0:
            plt.pause(delay)
        self.popup_info.set_text('')
        self.fig.canvas.draw()

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
        self.mask_cmap = 'RdYlGn' if not binary_mask else 'binary_r'
        self.mask_main_img.set_cmap(self.mask_cmap)
        self.mask_upper_img.set_cmap(self.mask_cmap)
        self.mask_lower_img.set_cmap(self.mask_cmap)
        if self.mask_alpha < 1:
            self.mask_alpha_backup = self.mask_alpha + 0
        self.mask_alpha = self.mask_alpha_backup if not binary_mask else 1


class Data:
    """Data used for Tommy's Volume Masker 3000 TM.

    :param volume_path: Full path to nifti file.
    :type volume_path: str
    :param make_mask: If make_mask='auto' (default), a brain mask is obtained
        by analyzing the image data. If 'none', no brain mask will be created.
        If path to nifti file, this file will be used as a mask.
    :type make_mask: str
    """
    axes_swaps = []

    def __init__(self, volume_path, make_mask='auto'):
        """Constructor method
        """
        self.volume_path = volume_path
        # load volume and normalize
        print('reading nifti file...')
        path = os.path.dirname(self.volume_path)
        filename = os.path.basename(self.volume_path)
        img = nib.load(self.volume_path)
        print('Done.')
        self.volume = img.get_fdata()
        self.volume -= self.volume.min()
        self.volume /= self.volume.max()

        self.slice = None if not config['start slice'] else config['start slice']

        # in case of 4D volumes only extract spatial dims
        if self.volume.ndim > 3:
            self.volume = self.volume[:, :, :, 0]

        # default view plane for the use case for which this was programmed
        self.volume = self.volume.swapaxes(2, 0)

        if self.slice is None:
            self.slice = int(self.volume.shape[0] / 2)

        # keep track of how often axes have been swapped (reversed for export)
        self.axes_swaps.append((2, 0))

        self.affine = img.affine.copy()
        self.header = img.header.copy()

        self.save_path = os.path.join(path, '_'.join(['m', filename]))

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
                # use stored mask (provided as input argument using -m)
                print('reading mask file...')

                # same initial axes swap as for volume
                self.mask = nib.load(make_mask).get_fdata().swapaxes(2, 0)
                print('done.')

    def get_data(self, first_dim_ind=None):
        """Return 3D volume data.

        :param first_dim_ind: Which slice of the first dimension.
        :type: None|int, optional
        :return: 3D volume data.
        :rtype: ndarray
        """
        if first_dim_ind is not None:
            return self.volume[first_dim_ind, :, :]
        return self.volume

    def get_mask(self, first_dim_ind=None):
        """Return 3D mask data.

        :param first_dim_ind: Which slice of the first dimension.
        :type: None|int, optional
        :return: 3D mask data.
        :rtype: ndarray
        """
        if first_dim_ind is not None:
            return self.mask[first_dim_ind, :, :]
        return self.mask

    def set_data(self, data, first_dim_ind=None):
        """Set 3D volume data.

        :param data: 3D input data.
        :type: ndarray
        :param first_dim_ind: Which slice of the first dimension.
        :type: None|int, optional
        """
        if first_dim_ind is not None:
            self.volume[first_dim_ind, :, :] = data
        else:
            self.volume = data

    def set_mask(self, mask, first_dim_ind=None):
        """Set 3D mask data.

        :param data: 3D input mask.
        :type: ndarray
        :param first_dim_ind: Which slice of the first dimension.
        :type: None|int, optional
        """
        if first_dim_ind is not None:
            self.mask[first_dim_ind, :, :] = mask
        else:
            self.mask = mask

    def switch_view_plane(self):
        """Switch view plane. Swaps data axes.
        """
        self.set_data(self.get_data().swapaxes(0, 2))
        self.set_mask(self.get_mask().swapaxes(0, 2))
        self.set_data(self.get_data().swapaxes(1, 2))
        self.set_mask(self.get_mask().swapaxes(1, 2))

        self.axes_swaps.append((0, 2))
        self.axes_swaps.append((1, 2))

    def export_mask(self):
        """Write mask to file.
        """

        # make nifti using information from volume and store
        print('saving...')
        out = nib.Nifti1Image(self._reversed_mask_swap().astype(bool),
                              affine=self.affine, header=self.header)
        nib.save(out, self.save_path)

    def _reversed_mask_swap(self):
        """Like :func:`vol2mask.Data.get_mask`, but in original orientation.

        :return: 3D mask data in original orientation.
        :rtype: ndarray
        """
        data_mask = self.mask + 0
        for so in self.axes_swaps[::-1]:
            data_mask = data_mask.swapaxes(*so)
        return data_mask

    def _reversed_data_swap(self):
        """Like :func:`vol2mask.Data.get_data`, but in original orientation.

        :return: 3D volume data in original orientation.
        :rtype: ndarray
        """
        data_volume = self.volume + 0
        for so in self.axes_swaps[::-1]:
            data_volume = data_volume.swapaxes(*so)
        return data_volume


class Controller:
    """Controller for Tommy's Volume Masker 3000 TM.
    """
    xys = []
    lasso = []
    ind = []
    binary_mask = False

    def __init__(self):
        """Constructor method
        """

        # link gui and data
        self.canvas = gui.main_ax.figure.canvas
        self.Npts = len(self.xys)
        self.selected = data.get_mask()[data.slice, :, :]

        # set defaults
        self.draw_mode = config['start draw mode']

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
        self.reset_selection()

    def onselect(self, verts):
        """Callback function for lasso.
        """
        path = Path(verts)
        self.xy_compute()
        self.ind = path.contains_points(self.xys, radius=-1)
        self.selected.flat[self.ind] = 1 if self.draw_mode == 'add' else 0
        gui.update_plots()

    def disconnect(self):
        """Remove lasso.
        """
        self.lasso.disconnect_events()
        gui.update_plots()

    def xy_compute(self):
        """Transform image data into index data (for selection).
        """
        xv, yv = np.meshgrid(np.arange(data.get_data(data.slice).shape[1]),
                             np.arange(data.get_data(data.slice).shape[0]))
        self.xys = np.vstack((xv.flatten(), yv.flatten())).T

    def reset_selection(self):
        """Reset lasso selection
        """
        self.xy_compute()
        self.ind = []
        self.selected = data.get_mask()[data.slice, :, :].flatten()
        self.lasso = LassoSelector(gui.main_ax, onselect=self.onselect)

    def _btnfct_set_slice(self):
        """Callback for set slice button
        """
        data.set_mask(
            controller.selected.reshape(data.get_mask(data.slice).shape),
            first_dim_ind=data.slice)
        controller.disconnect()
        gui.update_popup_text('Slice set', 0.25)

    def _btnfct_next_slice(self):
        """Callback for next slice button
        """
        data.slice += 1
        if data.slice >= data.volume.shape[0]:
            data.slice = data.volume.shape[0] - 1

    def _btnfct_prev_slice(self):
        """Callback for previous slice button
        """
        data.slice -= 1
        if data.slice < 0:
            data.slice = 0

    def _btnfct_switch_plane(self):
        """Callback for switch view plane button
        """
        data.switch_view_plane()
        data.slice = int(data.get_data().shape[0] / 2)
        gui.ax_lims = None

    def _btnfct_draw_mode(self):
        """Callback for switch draw mode button
        """
        self.draw_mode = 'remove' if self.draw_mode == 'add' else 'add'
        gui.update_popup_text(self.draw_mode, 0.25)

    def _btnfct_export(self):
        """Callback for export button
        """
        data.export_mask()
        gui.update_popup_text('Data successfully exported', 0.25)

    def _btnfct_quit(self):
        """Callback for quit button
        """
        gui.update_popup_text('Later...', 0.25)
        plt.close(gui.fig)
        sys.exit()

    def _btnfct_alpha_plus(self):
        """Callback for increasing mask alpha button
        """
        gui.mask_alpha -= 0.05
        if gui.mask_alpha < 0:
            gui.mask_alpha = 0

    def _btnfct_alpha_minus(self):
        """Callback for decreasing mask alpha button
        """
        gui.mask_alpha += 0.05
        if gui.mask_alpha > 1:
            gui.mask_alpha = 1

    def _btnfct_brightness_inc(self):
        """Callback for increasing image brightness button
        """
        gui.c_max -= 0.05
        if gui.c_max < 0:
            gui.c_max = 0

    def _btnfct_brightness_dec(self):
        """Callback for decreasing image brightness button
        """
        gui.c_max += 0.05
        if gui.c_max > 1:
            gui.c_max = 1

    def _btnfct_switch_filter(self):
        """Callback for switching filter button
        """
        self.filter['counter'] += 1
        gui.update_popup_text(
            self.filter['name'][
                self.filter['counter'] % len(self.filter['name'])], 0.25)

    def _btnfct_help(self):
        """Callback for help button
        """
        print('\n################################################\n'
              'Button mapping Tommy\'s MRI Volume Masker 3000 TM\n'
              '################################################\n')
        for k, v in config['keyboard'].items():
            print(': '.join([k, v]))
        gui.update_popup_text('see console output for help...', 1)

    def _btnfct_show_mask(self):
        """Callback for show / hide mask button
        """
        gui.show_mask = not gui.show_mask
        gui.mask_main_img.set_visible(gui.show_mask)
        gui.mask_upper_img.set_visible(gui.show_mask)
        gui.mask_lower_img.set_visible(gui.show_mask)

    def _btnfct_binary_mask(self):
        self.binary_mask = not self.binary_mask
        gui.binary_mask(self.binary_mask)

    def _btnfct_new(self):
        """Callback for new file selection dialog
        """
        if mpl.get_backend() == 'TkAgg':
            global data
            root = Tk()
            root.withdraw()
            fname = askopenfilename(
                title="Select (f)MRI image data",
                initialdir=os.path.dirname(os.path.abspath(data.volume_path)))
            root.destroy()
            if len(fname) > 0:
                data = Data(fname)
                return True, None
        else:
            gui.update_popup_text('Can\'t launch file selection dialog.\n'
                                  'Change matplotlib backend to TkAgg', 2)
        return False, gui.ax_lims

    def button_handler(self, event):
        """Handles button presses
        """
        update = True
        reset = False

        # execute button specific function or just pass (update = False)
        if event.key == "h":
            self._btnfct_help()
        elif event.key == config['keyboard']['slice up']:
            self._btnfct_next_slice()
            reset = True
        elif event.key == config['keyboard']['slice down']:
            self._btnfct_prev_slice()
            reset = True
        elif event.key == config['keyboard']['switch view plane']:
            self._btnfct_switch_plane()
            reset = True
        elif event.key == config['keyboard']['switch draw mode']:
            self._btnfct_draw_mode()
        elif event.key == config['keyboard']['export mask']:
            self._btnfct_export()
        elif event.key == config['keyboard']['quit']:
            self._btnfct_quit()
        elif event.key == config['keyboard']['increase mask alpha']:
            self._btnfct_alpha_plus()
        elif event.key == config['keyboard']['decrease mask alpha']:
            self._btnfct_alpha_minus()
        elif event.key == config['keyboard']['increase brightness']:
            self._btnfct_brightness_inc()
        elif event.key == config['keyboard']['decrease brightness']:
            self._btnfct_brightness_dec()
        elif event.key == config['keyboard']['switch filter']:
            self._btnfct_switch_filter()
        elif event.key == config['keyboard']['binary mask view']:
            self._btnfct_binary_mask()
        elif event.key == config['keyboard']['enable / disable mask']:
            self._btnfct_show_mask()
        elif event.key == config['keyboard']['set slice']:
            self._btnfct_set_slice()
            reset = True
        elif event.key == config['keyboard']['new file']:
            reset, gui.ax_lims = self._btnfct_new()
        elif event.key == config['keyboard']['reset zoom']:
            gui.ax_lims = None
        else:
            update = False

        if reset:
            # reset selection if not set
            self.reset_selection()

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

    def connect_gui(self):
        """Connect :class:`vol2mask.GUI` to  :class:`vol2mask.Controller`
        """
        gui.cid = gui.fig.canvas.mpl_connect("key_press_event",
                                             self.button_handler)
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
        if mpl.get_backend() == 'TkAgg':
            root = Tk()
            root.withdraw()
            args.file = askopenfilename(
                title="Select (f)MRI image data",
                initialdir=os.getcwd())
            root.destroy()
        else:
            raise Exception("Either provide input file using -f or change "
                            "matplotlib backend to TkAgg to enable file "
                            "selection dialogs (can be set in config.json).")

    # start program
    data = Data(args.file, args.mask)
    gui = GUI()
    controller = Controller()
    controller.connect_gui()
    plt.show()


if __name__ == '__main__':
    # variable definition on module level
    global data, gui, controller, config
    main()
