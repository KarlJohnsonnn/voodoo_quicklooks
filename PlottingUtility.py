import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr


_FIG_SIZE = (17, 6.7)
_FONT_SIZE = 12
_FONT_WEIGHT = 'normal'
dBZ_lim = [0, 1]
vel_lim = [0, 256]


# Some adjustments to the axis labels, ticks and fonts
def load_xy_style(ax, xlabel='Time [UTC]', ylabel='Height [m]'):
    """
    Method that alters the apperance of labels on the x and y axis in place.

    Note:
        If xlabel == 'Time [UTC]', the x axis set to major 
        ticks every 3 hours and minor ticks every 30 minutes.

    Args:
        ax (matplotlib.axis) :: axis that gets adjusted
        **xlabel (string) :: name of the x axis label
        **ylabel (string) :: name of the y axis label

    """

    ax.set_xlabel(xlabel, fontweight='normal', fontsize=_FONT_SIZE)
    ax.set_ylabel(ylabel, fontweight='normal', fontsize=_FONT_SIZE)
    if xlabel == 'Time [UTC]':
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 30]))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    ax.tick_params(axis='both', which='major', top=True, right=True, labelsize=_FONT_SIZE, width=3, length=4)
    ax.tick_params(axis='both', which='minor', top=True, right=True, width=2, length=3)


def load_cbar_style(cbar, cbar_label=''):
    """
    Method that alters the apperance of labels on the color bar axis in place.

    Args:
        ax (matplotlib.axis) :: axis that gets adjusted
        **cbar_label (string) :: name of the cbar axis label, Defaults to empty string.

    """
    cbar.ax.set_ylabel(cbar_label, fontweight='semibold', fontsize=_FONT_SIZE)
    cbar.ax.tick_params(axis='both', which='major', labelsize=_FONT_SIZE, width=2, length=4)


def plot_range_spectrogram(ZSpec, dt, **font_settings):
    unix_0 = np.float64(h.dt_to_ts(dt))

    Z = ZSpec.sel(ts = slice(unix_0, unix_0+30.0), rg = slice(font_settings['range_interval'][0], font_settings['range_interval'][1]))
    z_var = Z.values.copy()
    
    #z_var = np.ma.masked_less_equal(Z.values, 0.0)
    z_var = np.squeeze(z_var)

    x_var = np.linspace(0, 6*256, num=6*256)
    z_var = z_var.reshape((-1, 6*256), order='F')
    fig, ax = plt.subplots(1, figsize=_FIG_SIZE)

    surf = ax.pcolormesh(x_var, Z.rg, z_var[:, :], vmin=dBZ_lim[0], vmax=dBZ_lim[1], cmap='jet')
    #ax.set_xlim(vel_lim)
    ax.set_xlabel(xlabel= r"Doppler velocity bins [-]")

    ax.set_ylim(font_settings['range_interval'])
    ax.set_ylabel("range [m]", **{})
    ax.grid()

    # Add a color bar which maps values to colors.

    cbar = fig.colorbar(surf, fraction=0.2, shrink=1., pad=0.01, orientation='vertical',)
    cbar.ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=4)
    cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.minorticks_on()
    cbar.ax.set_ylabel("signal normalized", **{})
    plt.tight_layout()
    
def plot_time_spectrogram(ZSpec, rg, **font_settings):
    rg0 = np.float64(rg)

    Z = ZSpec.sel(rg = slice(rg0, rg0+40.0), ts = slice(ZSpec.ts[0], ZSpec.ts[-1]))
    z_var = Z.values.copy()
    
    #z_var = np.ma.masked_less_equal(Z.values, 0.0)
    z_var = np.squeeze(z_var)
    print(z_var.shape)

    y_var = np.linspace(0, 6*256, num=6*256)
    z_var = z_var.reshape((-1, 6*256), order='F')
    fig, ax = plt.subplots(1, figsize=_FIG_SIZE)

    surf = ax.pcolormesh([h.ts_to_dt(ts) for ts in Z.ts], y_var, z_var[:, :].T, vmin=dBZ_lim[0], vmax=dBZ_lim[1], cmap='jet')
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    ax.set_xlabel(xlabel= r"time [UTC]")

    ax.grid()

    # Add a color bar which maps values to colors.

    cbar = fig.colorbar(surf, fraction=0.2, shrink=1., pad=0.01, orientation='vertical',)
    cbar.ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=4)
    cbar.ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.minorticks_on()
    cbar.ax.set_ylabel("signal normalized", **{})
    plt.tight_layout()    
    

def plot_single_spectrogram(ZSpec, ts, rg, **font_settings):
  
    Z = ZSpec.sel(ts = ts, rg = rg)
    
    z_var = np.squeeze(Z.values)
    x_var = np.linspace(0, 256, num=256)
    
    if 'fig' in font_settings and 'ax' in font_settings:
        fig, ax = font_settings['fig'], font_settings['ax']
    else:
        fig, ax = plt.subplots(1, figsize=(7, 7))            

    for i in range(6):
        surf = ax.plot(x_var, z_var[:, i], linestyle='-', alpha=0.9)

    ax.set_xlabel(xlabel= r"Doppler velocity bins [-]")

    ax.set_ylim(dBZ_lim)
    ax.set_ylabel("signal normalized", **{})
    ax.grid(which='both')

    
    return fig, ax
    