import argparse
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.segment.mask import median_otsu
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from scipy import ndimage

plt.rcParams['toolbar'] = 'toolmanager'
for k in plt.rcParams.keys():
    if 'keymap' in k:
        for v in plt.rcParams[k]:
            plt.rcParams[k].remove(v)

plt.rcParams['keymap.zoom'].append('z')

plt.rcParams['axes.ymargin'] = 0
control_keys = 'navigation: up: next slice | '
control_keys += 'down: previous slice  | '
control_keys += 'v: switch view plane | '
control_keys += 'f: switch filter\n'
control_keys += 'control: enter: accept current slice | '
control_keys += 'e: export mask | '
control_keys += 'q: quit\n'
control_keys += 'drawing: d: draw mode | '
control_keys += '±: change mask alpha | '
control_keys += 'left / right: change brightness'


class GUI:
    c_max = 0.7
    mask_alpha = 0.2

    status_text = ''
    status_font = {'color': 'white', 'verticalalignment': 'center',
                   'horizontalalignment': 'left'}

    info_text = ''
    info_text_font = {'color': 'white', 'horizontalalignment': 'center',
                      'size': 16}

    cid = None

    def __init__(self):
        # make main window
        self.fig = plt.figure(facecolor=(0.22, 0.22, 0.22))
        self.fig.canvas.manager.set_window_title(
            "Tommy's MRI Volume Masker 3000 TM")
        self.figtitle = self.fig.text(0.025, 0.025, '',
                                      fontdict={'size': 8, 'color': 'w'})

        # divide into parts
        gs = self.fig.add_gridspec(20, 10)

        # main drawing window
        self.title_ax = self.fig.add_subplot(gs[0, :7])
        self.title_ax.axis('off')
        self.title_ax.margins(0)

        # main drawing window
        self.main_ax = self.fig.add_subplot(gs[1:, :7])
        self.main_ax.axis('off')
        self.main_img = self.main_ax.imshow(data.get_data(data.slice), "gray")
        self.mask_main_img = self.main_ax.imshow(data.get_mask(data.slice),
                                                 'RdYlGn')
        # upper right window
        self.upper_ax = self.fig.add_subplot(gs[:10, 7:])
        self.upper_ax.axis('off')
        self.upper_img = self.upper_ax.imshow(np.empty((2, 2)), "gray")
        self.mask_upper_img = self.upper_ax.imshow(np.empty((2, 2)), 'RdYlGn')

        # lower right window
        self.lower_ax = self.fig.add_subplot(gs[10:, 7:])
        self.lower_ax.axis('off')
        self.lower_img = self.lower_ax.imshow(np.empty((2, 2)), "gray")
        self.mask_lower_img = self.lower_ax.imshow(np.empty((2, 2)), 'RdYlGn')

        # info text object
        self.status = self.title_ax.set_title(self.status_text,
                                              fontdict=self.status_font,
                                              loc='left')

        # popup status change text
        self.popup_info = self.main_ax.text(0.5, 0.5, self.info_text,
                                            fontdict=self.info_text_font,
                                            transform=self.main_ax.transAxes)

        self.main_ax.autoscale(False)
        self.main_ax.callbacks.connect('xlim_changed', self._on_xlims_change)
        self.main_ax.callbacks.connect('ylim_changed', self._on_ylims_change)
        self.x_sel = [0, data.get_data(data.slice).shape[1]]
        self.y_sel = [data.get_data(data.slice).shape[0], 0]
        # customize figure toolbar
        try:
            for rm_tools in ['home', 'back', 'forward', 'subplots', 'save',
                             'help', 'pan']:
                self.fig.canvas.manager.toolmanager.remove_tool(rm_tools)
        except AttributeError:
            print('no tools')

    def update_img_extent(self, img):
        """
        Updates image objects by replacing their extent value to fit data in
        img.

        :param ndarray img:
            The new volume data (3D).
        """
        extent = [self.x_sel[0],
                  self.x_sel[1],
                  self.y_sel[0],
                  self.y_sel[1]]
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

    def _on_xlims_change(self, ax):
        """
        Detects change in xlim and sets values in self (to keep zoom / pan)
        """
        self.x_sel = [int(val) for val in ax.get_xlim()]

    def _on_ylims_change(self, ax):
        """
        Detects change in ylim and sets values in self (to keep zoom / pan)
        """
        self.y_sel = [int(val) for val in ax.get_ylim()]

    def update_axes_limits(self):
        """
        Updates axes limits to fit data.

        :param None|ndarray new_data:
            The new volume data (3D). If None (default) data will be obtained
            using data.get_data().
        """
        self.main_ax.set_xlim(self.x_sel)
        self.main_ax.set_ylim(self.y_sel)

    def update_plots(self, new_data=None, new_mask=None, first_dim_ind=None):
        """
        Updates all plots.

        :param None|ndarray new_data:
            New volume data (3D) to update the plots.
        :param None|ndarray new_mask:
            New volume mask (3D) to update the plots.
        :param None|int first_dim_ind:
            Slice index to show. If None (default) the middle slice will be
            selected.
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
        filter = controller.filter['filter'][
            controller.filter['counter'] % len(controller.filter['name'])]
        filter_args = controller.filter['args'][
            controller.filter['counter'] % len(controller.filter['name'])]

        if filter is not None:
            new_data = filter(new_data, *filter_args)

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
        self.update_axes_limits()
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

        self.update_status_text()
        self.fig.canvas.draw_idle()

    def update_status_text(self):
        """
        Updates info text below figure
        """
        self.status_text = 'slice: {} | draw mode: {} | filter: {}\n'
        self.status_text = self.status_text.format(
            data.slice,
            controller.draw_mode,
            controller.filter['name'][controller.filter['counter'] %
                                      len(controller.filter['name'])])
        self.status_text += control_keys

        self.status.set_text(self.status_text)

    def update_info_text(self, info, delay=0.):
        """
        Updates popup text on status change.

        :param str info:
            What to show.
        :param float delay:
            How long to show it.
        """
        self.popup_info.set_text(info)
        self.fig.canvas.draw()
        if delay > 0:
            plt.pause(delay)
        self.popup_info.set_text('')
        self.fig.canvas.draw()


class Data:
    swap_operations = []
    slice = None

    def __init__(self, volume_path, make_mask='auto'):
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

        # in case of 4D volumes only extract spatial dims
        if self.volume.ndim > 3:
            self.volume = self.volume[:, :, :, 0]

        # default view plane for the use case for which this was programmed
        self.volume = self.volume.swapaxes(2, 0)

        if self.slice is None:
            self.slice = int(self.volume.shape[0] / 2)

        self.swap_operations.append((2, 0))
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
        if first_dim_ind is not None:
            return self.volume[first_dim_ind, :, :]
        return self.volume

    def get_mask(self, first_dim_ind=None):
        if first_dim_ind is not None:
            return self.mask[first_dim_ind, :, :]
        return self.mask

    def set_data(self, data, first_dim_ind=None):
        if first_dim_ind is not None:
            self.volume[first_dim_ind, :, :] = data
        else:
            self.volume = data

    def set_mask(self, mask, first_dim_ind=None):
        if first_dim_ind is not None:
            self.mask[first_dim_ind, :, :] = mask
        else:
            self.mask = mask

    def switch(self):
        self.set_data(self.get_data().swapaxes(0, 2))
        self.set_mask(self.get_mask().swapaxes(0, 2))
        self.set_data(self.get_data().swapaxes(1, 2))
        self.set_mask(self.get_mask().swapaxes(1, 2))

        self.swap_operations.append((0, 2))
        self.swap_operations.append((1, 2))

    def export(self):
        """
                Write mask to file.
                """

        # make nifti using information from volume and store
        print('saving...')
        out = nib.Nifti1Image(self._reversed_mask_swap().astype(bool),
                              affine=self.affine, header=self.header)
        nib.save(out, self.save_path)

    def _reversed_mask_swap(self):
        data_mask = self.mask + 0
        for so in self.swap_operations[::-1]:
            data_mask = data_mask.swapaxes(*so)
        return data_mask

    def _reversed_data_swap(self):
        data_volume = self.volume + 0
        for so in self.swap_operations[::-1]:
            data_volume = data_volume.swapaxes(*so)
        return data_volume


class Controller:
    xys = []
    lasso = []
    ind = []
    draw_mode = 'add'

    def __init__(self):
        self.canvas = gui.main_ax.figure.canvas
        self.reset()
        self.Npts = len(self.xys)
        self.selected = data.get_mask()[data.slice, :, :]

        self.filter = {
            'counter': 0,
            'name': ['no filter', 'sobel', 'gauss'],
            'filter': [None,
                       ndimage.sobel,
                       ndimage.gaussian_filter],
            'args': [None,
                     [],
                     [2 * data.header.get_zooms()[0]]]
        }

    def onselect(self, verts):
        path = Path(verts)
        self.xy_compute()
        self.ind = path.contains_points(self.xys)
        self.selected.flat[self.ind] = 1 if self.draw_mode == 'add' else 0
        gui.update_plots()

    def disconnect(self):
        self.lasso.disconnect_events()
        gui.update_plots()

    def xy_compute(self):
        xv, yv = np.meshgrid(np.arange(data.get_data(data.slice).shape[1]),
                             np.arange(data.get_data(data.slice).shape[0]))
        self.xys = np.vstack((xv.flatten(), yv.flatten())).T

    def reset(self):
        self.xy_compute()
        self.lasso = LassoSelector(gui.main_ax, onselect=self.onselect)
        self.ind = []
        self.selected = data.get_mask()[data.slice, :, :].flatten()

    def button_handler(self, event):
        update = True
        if event.key == "enter":
            data.set_mask(
                controller.selected.reshape(data.get_mask(data.slice).shape),
                first_dim_ind=data.slice)

            controller.disconnect()
            gui.update_info_text('Slice set', 0.25)
        elif event.key == "up":

            data.slice += 1
            if data.slice >= data.volume.shape[0]:
                data.slice = data.volume.shape[0] - 1
        # Select previous slice for displaying.
        elif event.key == "down":
            data.slice -= 1
            if data.slice < 0:
                data.slice = 0
        # Switch view plane.
        elif event.key == "v":
            data.switch()
            data.slice = int(data.get_data().shape[0] / 2)
        # switch between drawing and removing
        elif event.key == "d":
            self.draw_mode = 'remove' if self.draw_mode == 'add' else 'add'
            gui.update_info_text(self.draw_mode, 0.25)
        # Export data (write to file).
        elif event.key == "e":
            data.export()
            gui.update_info_text('Data successfully exported', 0.25)
        # Exit program.
        elif event.key == "q":
            gui.update_info_text('Later...', 0.25)
            plt.close(gui.fig)
            sys.exit()
        # Reduce alpha of mask overlay.
        elif event.key == "+":
            gui.mask_alpha -= 0.05
            if gui.mask_alpha < 0:
                gui.mask_alpha = 0

        # Increase alpha of mask overlay.
        elif event.key == "-":
            gui.mask_alpha += 0.05
            if gui.mask_alpha > 1:
                gui.mask_alpha = 1
        # Lower upper clim of plots.
        elif event.key == "right":
            gui.c_max -= 0.05
            if gui.c_max < 0:
                gui.c_max = 0

        # Raise upper clim of plots.
        elif event.key == "left":
            gui.c_max += 0.05
            if gui.c_max > 1:
                gui.c_max = 1
        # switch between filters
        elif event.key == "f":
            self.filter['counter'] += 1
            gui.update_info_text(
                self.filter['name'][
                    self.filter['counter'] % len(self.filter['name'])], 0.25)
        elif event.key == "escape":
            gui.x_sel = [0, data.get_data(data.slice).shape[1]]
            gui.y_sel = [data.get_data(data.slice).shape[0], 0]
        else:
            update = False

        if update:
            self.reset()
            gui.update_plots()

    def connect(self):
        gui.cid = gui.fig.canvas.mpl_connect("key_press_event",
                                             self.button_handler)
        gui.update_plots()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='nifti file')
    parser.add_argument('-m', '--mask', type=str, help='initial mask file',
                        default='auto')
    args = parser.parse_args()

    global data, gui, controller
    data = Data(args.file, args.mask)
    gui = GUI()
    controller = Controller()
    controller.connect()
    plt.show()


if __name__ == '__main__':
    global data, gui, controller
    main()
