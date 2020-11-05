import pandas
import datetime
import numpy as np
import sys
import time

from itertools import combinations, groupby
from scipy.io import loadmat
from numba import jit

sys.path.append('../larda/')
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import logging

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('MacOSX')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

"""
____ _    ____ ___  ____ _       _ _  _ ____ ____ ____ _  _ ____ ___ _ ____ _  _    ___  ____ ____ _ _  _ _ ___ _ ____ _  _ 
| __ |    |  | |__] |__| |       | |\ | |___ |/  | |__/ |\/| |__|  |  | |  | |\ |    |  \ |___ |___ | |\ | |  |  | |  | |\ | 
|__] |___ |__| |__] |  | |___    | | \| |    |__| |  \ |  | |  |  |  | |__| | \|    |__/ |___ |    | | \| |  |  | |__| | \| 
                                                                                                                            
"""
# definition of lidar box/linfcn thresholds in [log(sr-1 m-1)] and (CDR)
lidar_thresh_dict = {
    'Shupe2007': {'bsc': -4.5, 'dpl': 0.1, 'cdr': -6.532125137753436e-01},
    'deBoer2009': {'bsc': -4.3, 'dpl': 0.03, 'cdr': -1.208620483882601e+00},
    'Luke2010': {'bsc': -4.3, 'dpl': 0.1, 'cdr': -7.422612476299660e-01},
    'linear-1': {'slope': 12.0, 'intersection': -5.5},
    # 'other': {'bsc': -5.2, 'dpl': 0.15, 'cdr': -0.01},
    # 'Cloudnet': {'bsc': -4.7, 'dpl': 1.0, 'cdr': 20},
    # 'Willi': {'bsc': -5.2, 'dpl': 0.1, 'cdr': -6.532125137753436e-01}
}
lidar_thresh_dict_other = {
    'linear-2': {'slope': 2.0, 'intersection': -5.5},
}

# specific variable information, name from .mat file, unit name, variable limits
variable_dict = {
    # 'Ze_cc_ts': ['dBZ', [-60, 20]],
    # 'Vd_cc_ts': ['m s-1', [-4, 2]],
    # 'width_cc_ts': ['m s-1', [0, 2]],
    'dpol_NN_ts': ['1', [0, 0.3]],
    'CDR_NN_ts': ['1', [-3.0, 0.5]],
    'bsc_NN_ts': ['log(sr-1 m-1)', [-5.5, -2.5]]
}

# cloudnet classes that contain different kinds of liquid cloud droplets
liq_mask_flags = {
    'wrm': [1, 3],  # cloud droplets only, drizzle/rain + cloud droplets
    'scl': [5, 7],  # ice + supercooled liquid, melting ice + cloud liquid droplets
    'both': [1, 3, 5, 7],  # wrm + scl
    'all': np.linspace(1, 10, 10, dtype=np.int)  # all cloudnet classes
}

# define a list for with isotherms will be plotted
isotherm_list = [-38, -25, -10, 0]  # list must be increasing

plot_specs = {
    'fig_size': [12, 5],
    'font_size': 12,
    'label_size': 12,
    'font_weight': 'normal'

}

"""
____ _  _ _  _ ____ ___ _ ____ _  _    ___  ____ ____ _ _  _ _ ___ _ ____ _  _ 
|___ |  | |\ | |     |  | |  | |\ |    |  \ |___ |___ | |\ | |  |  | |  | |\ | 
|    |__| | \| |___  |  | |__| | \|    |__/ |___ |    | | \| |  |  | |__| | \| 
                                                                               
"""


def load_dot_mat_file(path, string):
    t0 = time.time()
    data = loadmat(path)
    logger.info(f'Loading {string} data took {datetime.timedelta(seconds=int(time.time() - t0))} [hour:min:sec]')
    return data


def cases_from_csv(data_loc, **kwargs):
    """This function extracts information from an excel sheet. It can be used for different scenarios.
    The first row of the excel sheet contains the headline, defined as follows:

    """

    df = pandas.read_csv(data_loc)
    case_list = []
    # exclude header from data
    for icase in range(df.shape[0]):
        begin_dt = datetime.datetime.strptime(str(df['begin_dt'][icase]), '%Y%m%d%H%M%S')
        end_dt = datetime.datetime.strptime(str(df['end_dt'][icase]), '%Y%m%d%H%M%S')

        if begin_dt < end_dt:
            case_list.append({
                'begin_dt': begin_dt,
                'end_dt': end_dt,
                'plot_range': [float(df['h0'][icase]), float(df['hend'][icase])],
                'cloudtype': df['cloudtype'][icase],
                'notes': df['notes'][icase],
                'extra': df['extra'][icase]
            })

    return case_list


def datenum2datetime(dn):
    """Converting Matlab datenum to Python datetime format.
    """
    return datetime.datetime.fromordinal(int(dn) - 366) + datetime.timedelta(days=dn % 1)


def since2001_to_dt(s):
    """seconds since 2001-01-01 to datetime"""
    # return (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    return datetime.datetime(2001, 1, 1) + datetime.timedelta(seconds=s)


def datetime2datenum(dt):
    """Converting Python datetime to Matlab datenum to format.
    """
    mdn = dt + datetime.timedelta(days=366)
    frac = (dt - datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def toC(datalist):
    return datalist[0]['var'], datalist[0]['mask']


def time_height_slicer(container, ts_bound, rg_bound):
    """Routine for slicing along time and range dimension for case studies.

    Args:
        container (dict) : complete .mat file data
        ts_bound (list) : list of indices for slicing along time axsis
        rg_bound (list) : list of indices for slicing along rane axsis

    Return:
        container (dict) : sliced container
    """

    assert ts_bound[0] != ts_bound[1], 'slicing failed, time boundaries wrong'
    assert rg_bound[0] != rg_bound[1], 'slicing failed, range boundaries wrong'

    var1D_time_list, var1D_range_list, var2D_list = [], [], []
    if 'ts_class_time' in container.keys():  # set variable lists for cloudnet classification dict
        var1D_time_list = ['cb_first_ts', 'ct_last_ts', 'ts_class_time']
        var1D_range_list = ['h_class']
        var2D_list = ['detect_status_ts', 'target_class_ts']
    elif 'ts_cat_time' in container.keys():  # set variable lists for cloudnet categorization dict
        var1D_time_list = ['lwp_ts', 'rainrate_ts', 'ts_cat_time']
        var1D_range_list = ['h_cat']
        var2D_list = ['T_mod_ts', 'Vd_cc_ts', 'Ze_cc_ts', 'att_bscatt_ts', 'ldr_cc_ts', 'p_mod_ts', 'rh_mod_ts', 'theta_mod_ts',
                      'wdir_mod_ts', 'width_cc_ts', 'winddirshear_mod_ts', 'windspdshear_mod_ts', 'wspd_mod_ts']
    elif 'ts_polly_time' in container.keys():  # set variable lists for PollyXT dict
        var1D_time_list = ['ts_polly_time']
        var1D_range_list = ['h_class']
        var2D_list = ['bsc_polly_ts', 'bsc_polly_woAI_ts', 'dpol_polly_ts', 'dpol_polly_woAI_ts']
    elif 'ts_NN_time' in container.keys():  # set variable lists for predicted Ed Luke ANN prediction dict
        var1D_time_list = ['ts_NN_time']
        var1D_range_list = ['h_class']
        var2D_list = ['CDR_NN_ts', 'bsc_NN_ts', 'dpol_NN_ts']
    elif 'ts_sp_time' in container.keys():  # set variable lists for Mira Doppler radar dict
        var1D_time_list = ['ts_sp_time']
        var1D_range_list = ['h_class']
        var2D_list = ['Vd_sp_ts', 'Ze_cc_ts', 'kurt_smoo_ts', 'ldr_cor_sp_ts', 'le_peak1_sp_ts', 'left_edge_sp_ts', 'ls_peak1_smoo_ts',
                      'peak1_loc_sp_ts', 're_peak1_sp_ts', 'right_edge_sp_ts', 'rs_peak1_smoo_ts', 'skew_smoo_ts', 'width_sp_ts']
    elif 'ts_rs_time' in container.keys():  # set variable lists for radiosonde dict
        var1D_time_list = ['ts_rs_time']
        var1D_range_list = ['h_rs']
        var2D_list = ['rh_rs_ts', 'T_rs_ts', 'TD_rs_ts']

    assert len(var1D_range_list) > 0, 'slicing failed, unknown container'

    for ivar in var1D_time_list:
        container[ivar] = container[ivar].reshape(container[ivar].size)
        container[ivar] = container[ivar][ts_bound[0]:ts_bound[1]]

    for ivar in var1D_range_list:
        container[ivar] = container[ivar].reshape(container[ivar].size)
        container[ivar] = container[ivar][rg_bound[0]:rg_bound[1]]

    for ivar in var2D_list:
        container[ivar] = container[ivar][rg_bound[0]:rg_bound[1], ts_bound[0]:ts_bound[1]]

    return container


def get_1st_cloud_base_idx(cb_first_ts, range_list):
    """Extract the indices of the first cloud base.

    Args:
        cb_first_ts (list) : list or np.array of fist cloud base occurrence, fill_value=nan
        range_list (list) : range bins

    Return:
        idx_1st_cloud_base (np.array) : sliced container
    """
    idx_1st_cloud_base = []
    for i_cb_ts in cb_first_ts:
        if np.isnan(i_cb_ts):
            idx_1st_cloud_base.append(np.nan)
        else:
            idx_1st_cloud_base.append(h.argnearest(range_list, i_cb_ts))

    return np.array(idx_1st_cloud_base)


def get_temp_lines(temperatures, rg_list, isoTemp):
    """Extracts the index, temperaure and range for a specific isotherm.

    Args:
        - temperature (np.array): 2D array containing temperature values, dimensions: time-range
        - rg_list (list): list of range values
        - isoTemp (integer): temperature in °C

    Return:
        - (dict): containing index, temperature and range of isotherm
    """
    idx_list, temp_list, range_list = [], [], []
    for iT in range(temperatures.shape[1]):
        idx, val = next(((idx, temp) for idx, temp in enumerate(temperatures[:, 0]) if temp < isoTemp), (None, np.nan))
        idx_list.append(idx)
        temp_list.append(val)
        range_list.append(rg_list[idx])

    return {'idx': np.array(idx_list), 'temp': np.array(temp_list), 'rg': np.array(range_list)}


def get_combined_liquid_mask(liq_pred_mask, cloudnet_liq_mask):
    """Return an 2D integer array, specifying the location in time and height, where liquid cloud droplets orruce.
        0   ...     no signal
        1   ...     Cloudnet and the NN detected liquid
        2   ...     liquid detected by the NN only
        3   ...     liquid detected by the Cloudnet only

        Args:
            liq_pred_mask (array, bool) : 2D array in time and height where True values represent liquid pixel from lidar prediction
            cloudnet_liq_mask (array, bool) : 2D array in time and height where True values represent liquid pixel from Cloudnet
    """
    combi_mask_liq = np.zeros(liq_pred_mask.shape, dtype=np.int)
    combi_mask_liq[liq_pred_mask * cloudnet_liq_mask] = 1  # pixel classified as liquid in both
    combi_mask_liq[liq_pred_mask * ~cloudnet_liq_mask] = 2  # pixel classified as liquid by NN only
    combi_mask_liq[~liq_pred_mask * cloudnet_liq_mask] = 3  # pixel classified as liquid by Cloudnet only

    return combi_mask_liq

def get_combined_scatter_mask(hsrl_pred, nFoOBins):
    """Return an 2D integer array, specifying the location in time and height, where liquid cloud droplets orruce.
        0   ...     no signal
        1   ...     Cloudnet and the NN detected liquid
        2   ...     liquid detected by the NN only
        3   ...     liquid detected by the Cloudnet only

        Args:
            liq_pred_mask (array, bool) : 2D array in time and height where True values represent liquid pixel from lidar prediction
            cloudnet_liq_mask (array, bool) : 2D array in time and height where True values represent liquid pixel from Cloudnet
    """
    combi_mask = np.zeros(hsrl_pred['bsc_NN_ts'].shape)

    dpl_xaxis = np.linspace(0, 0.3, nFoOBins).astype(np.float32)
    bsc1_yaxis = lidar_thresh_dict['linear-1']['slope'] * dpl_xaxis + lidar_thresh_dict['linear-1']['intersection']
    bsc2_yaxis = lidar_thresh_dict_other['linear-2']['slope'] * dpl_xaxis + lidar_thresh_dict_other['linear-2']['intersection']

    for iBin in range(nFoOBins):
        low_depol, high_depol  = dpl_xaxis[iBin] > hsrl_pred['dpol_NN_ts'], dpl_xaxis[iBin] < hsrl_pred['dpol_NN_ts']

        low_bsc1, high_bsc1  = bsc1_yaxis[iBin] > hsrl_pred['bsc_NN_ts'], bsc1_yaxis[iBin] < hsrl_pred['bsc_NN_ts']
        low_bsc2, high_bsc2  = bsc2_yaxis[iBin] > hsrl_pred['bsc_NN_ts'], bsc2_yaxis[iBin] < hsrl_pred['bsc_NN_ts']
        combi_mask[(low_bsc1 * high_bsc2 * high_depol)] = 2
        combi_mask[low_bsc2 * high_depol] = 3
        combi_mask[high_bsc1 * low_depol] = 1


    return combi_mask



@jit()
def mask_below_temperature(mask_copy, isotherm):
    # remove liquid pixel below certain temperatures
    if isotherm['idx'].any() is not None:
        for iT in range(mask_copy.shape[1]):
            if isotherm['idx'][iT] is not None:
                mask_copy[isotherm['idx'][iT]:, iT] = False
    return mask_copy


def sum_liquid_layer_thickness_per_category(cloudnet_liq_mask, ann_liquid_mask, combi_mask_liq, rg_res=30.0):
    """Calculating the liquid layer thickness of the total vertical column"""
    sum_ts = dict()
    sum_ts.update({itemp: np.count_nonzero(ival, axis=0) * rg_res for itemp, ival in combi_mask_liq.items()})
    sum_ts.update({'cloudnet': np.nansum(cloudnet_liq_mask, axis=0) * rg_res})
    sum_ts.update({'neuralnet': np.nansum(ann_liquid_mask, axis=0) * rg_res})
    return sum_ts


def init_nan_array(shape):
    """ Generates an numpy array of dimension 'shape' filled with NaN values.

    Args:
        - shape (tuple): dimensions of the nan array

    Return:
        - nan array (np.array): nan array of dimension 'shape'
    """
    return np.full(shape, fill_value=np.nan)


def wrapper(mat_data, **kwargs):
    """Wrap a larda container around .mat file data to use the pyLARDA.Transformations library .

    Args:
        mat_data (.mat) : Matlab .mat file datad

    Kwargs:
        var_name (string) : variable name in .mat file
        var_unit (string) : unit of the variable
        var_lims (list) : boundaries of the variable (for plotting)

    Return:
        container (dict) : sliced container
    """
    var_name = kwargs['var_name'] if 'var_name' in kwargs else ''
    var_unit = kwargs['var_unit'] if 'var_unit' in kwargs else ''

    assert isinstance(var_name, str) and len(var_name) > 0, 'Error utility.wrapper! Check var_name argument!'

    time_var, range_var, system = '', '', ''
    if 'ts_class_time' in mat_data.keys():  # set variable lists for cloudnet classification dict
        time_var, range_var = 'ts_class_time', 'h_class'
        system = 'cloudnet-classification'
    elif 'ts_cat_time' in mat_data.keys():  # set variable lists for cloudnet categorization dict
        time_var, range_var = 'ts_cat_time', 'h_cat'
        system = 'cloudnet-categorization'
    elif 'ts_polly_time' in mat_data.keys():  # set variable lists for PollyXT dict
        time_var, range_var = 'ts_polly_time', 'h_class'
    elif 'ts_NN_time' in mat_data.keys():  # set variable lists for predicted Ed Luke ANN prediction dict
        time_var, range_var = 'ts_NN_time', 'h_class'
    elif 'ts_sp_time' in mat_data.keys():  # set variable lists for Mira Doppler radar dict
        time_var, range_var = 'ts_sp_time', 'h_class'
        system = 'mira'
    elif 'ts_rs_time' in mat_data.keys():  # set variable lists for radiosonde dict
        time_var, range_var = 'ts_rs_time', 'h_rs'
        system = 'radio-sonde'

    if var_name == 'target_class_ts':
        name = 'CLASS'
        colormap = 'cloudnet_target_new'
    elif var_name == 'combi_liq_mask':
        system = 'ann-vs-cloudnet-cloud-droplet-mask'
        name = 'CLASS'
        colormap = 'four_colors'
    elif var_name == 'combi_scatter_mask':
        system = 'ann-scatter-to-class-label'
        name = 'CLASS'
        colormap = 'four_colors2'
    elif var_name == 'ldr_cc_ts':
        system = 'cloudnet-categorization'
        name = var_name
        colormap = 'LDR'
    else:
        name = var_name
        colormap = 'jet'

    var = mat_data[var_name].copy()

    dt_list = [datenum2datetime(dn) for dn in mat_data[time_var]]
    time = [h.dt_to_ts(dt) for dt in dt_list]
    var_lims = kwargs['var_lims'] if 'var_lims' in kwargs else [np.nanmin(var), np.nanmax(var)]
    if var_name == 'bsc_NN_ts' and 'var_converter' in kwargs and kwargs['var_converter'] == 'log':
        var = np.ma.masked_invalid(var.T)
        var = np.ma.power(10.0, var)
        var_lims = np.ma.power(10.0, var_lims)
    else:
        var = var.T

    if len(var.shape) == 2:
        dimlabel = ['time', 'range']
    elif len(var.shape) == 1:
        dimlabel = ['time']
    else:
        raise ValueError('Variable must be 1D or 2D!')

    larda_container = {'dimlabel': dimlabel,
                       'filename': 'accept .mat file',
                       'paraminfo': {},
                       'rg_unit': 'km',
                       'colormap': colormap,
                       'var_unit': var_unit,
                       'var_lims': var_lims,
                       'system': system,
                       'name': name,
                       'rg': np.array(mat_data[range_var]),
                       'ts': np.array(time),
                       'mask': np.isnan(var),
                       'var': var}

    return larda_container


def cdr2ldr(cdr):
    cdr = np.array(cdr)
    return np.power(10.0, cdr) / (2 + np.power(10.0, cdr))


def ldr2cdr(ldr):
    ldr = np.array(ldr)
    return np.ma.log10(-2 * ldr / (ldr - 1)) / (np.log10(2) + np.log10(5))


def add_boxes(ax, boxes, **kwargs):
    """ Routine for adding boxes to an existing axis"""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    # Create list for all the error patches
    linewidth = kwargs['line_width'] if 'line_width' in kwargs else 4
    Nbins = kwargs['Nbins'] if 'Nbins' in kwargs else 120
    threshold_boxes = []
    edgecolors = ['black', 'red', 'darkblue', 'blueviolet', 'orange']
    linestyles = ['-', '--', '-.', ':', ':']
    for i, (i_box_name, i_box_val) in enumerate(boxes.items()):
        if 'linear' in i_box_name:
            xaxis = np.linspace(0, 0.3, Nbins)
            yaxis = boxes[i_box_name]['slope'] * xaxis + boxes[i_box_name]['intersection']
            lin = ax.plot(xaxis, yaxis, color=edgecolors[i], linestyle=linestyles[i], linewidth=linewidth, label=i_box_name)
            threshold_boxes.append(lin[0])
        else:
            width, height = i_box_val['dpl'], -2.5 - i_box_val['bsc']
            print(i)
            rect = Rectangle((0, i_box_val['bsc']), width, height,
                             facecolor='None', linestyle=linestyles[i], edgecolor=edgecolors[i], clip_on=False, label=i_box_name)
            threshold_boxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(threshold_boxes, edgecolors=edgecolors, linestyles=linestyles, facecolors='None', linewidths=linewidth)
    ax.add_collection(pc)
    ax.legend(handles=threshold_boxes, loc='upper right', framealpha=1.0, prop=kwargs)

    return ax


def add_ll_thichkness(ax, dt_list, sum_ll_thickness, **kwargs):
    font_size = kwargs['font_size'] if 'font_size' in kwargs else 12
    font_weight = kwargs['font_weight'] if 'font_weight' in kwargs else 'normal'
    y_lim = kwargs['y_lim'] if 'y_lim' in kwargs else [0, 2]
    smooth = kwargs['smooth'] if 'smooth' in kwargs else False
    cn_varname = 'Cloudnet_smoothed' if smooth else 'Cloudnet'
    nn_varname = 'ANN_smoothed' if smooth else 'ANN'

    ax1 = ax.twinx()
    ax1.plot(dt_list, sum_ll_thickness[0], color='black', linestyle='-', alpha=0.75, label=cn_varname)
    ax1.plot(dt_list, sum_ll_thickness[1], color='red', linestyle='-', alpha=0.75, label=nn_varname)

    ax1.set_ylim(y_lim)
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.tick_params(axis='both', which='both', right=True)
    ax1.tick_params(axis='both', which='major', labelsize=font_size, width=3, length=5.5)
    ax1.tick_params(axis='both', which='minor', width=2, length=3)
    ax1 = tr.set_xticks_and_xlabels(ax1, dt_list[-1] - dt_list[0])
    ax1.set_ylabel('liquid layer thickness [m]', fontsize=font_size, fontweight=font_weight)

    return ax1


def calc_overlapp_supersat_liquidmask(rs_data, liq_mask):
    overlapp = np.argwhere(np.logical_and(liq_mask, rs_data['abv_thresh_mask']))
    idx_tot_pixel = np.argwhere(liq_mask[:, rs_data['ts_avbl_mask']])
    if idx_tot_pixel.shape[0] > 0:
        return overlapp.shape[0] / idx_tot_pixel.shape[0] * 100
    else:
        return 0


def remove_cloud_edges(mask, n=3):
    """
    This function returns the 2D binary mask of an array shrunk by n pixels.
    Args:
        mask (numpy.array) : where True and False corresponds to masked value, i.e. no signal, and False = a non masked value, i.e. signal respectively
        n (int) : number by which the mask is shrunk, default: 3

    Returns:
        mask reduced by n pixels around the edges

    """
    # row-wise operation
    arr = np.full(mask.shape, False)
    for irow in range(n, mask.shape[0] - n):
        arr[irow:irow + n, mask[irow, :] < mask[irow + 1, :]] = True
        arr[irow - n:irow, mask[irow, :] > mask[irow + 1, :]] = True
    # column-wise operation
    for icol in range(n, mask.shape[1] - n):
        arr[mask[:, icol] < mask[:, icol + 1], icol:icol + n] = True
        arr[mask[:, icol] > mask[:, icol + 1], icol - n:icol] = True

    mask[arr] = True
    return mask


def get_nan_mask(arr):
    mask = np.full(arr.shape, False)
    mask[arr >= 0.0] = True
    return mask


def find_bases_tops(mask, rg_list):
    """
    This function finds cloud bases and tops for a provided binary cloud mask.
    Args:
        mask (np.array, dtype=bool) : bool array containing False = signal, True=no-signal
        rg_list (list) : list of range values

    Returns:
        cloud_prop (list) : list containing a dict for every time step consisting of cloud bases/top indices, range and width
        cloud_mask (np.array) : integer array, containing +1 for cloud tops, -1 for cloud bases and 0 for fill_value
    """
    cloud_prop = []
    cloud_mask = np.full(mask.shape, 0, dtype=np.int)
    for iT in range(mask.shape[0]):
        cloud = [(k, sum(1 for j in g)) for k, g in groupby(mask[iT, :])]
        idx_cloud_edges = np.cumsum([prop[1] for prop in cloud])
        bases, tops = idx_cloud_edges[0:][::2][:-1], idx_cloud_edges[1:][::2]
        if tops.size > 0 and tops[-1] == mask.shape[1]:
            tops[-1] = mask.shape[1] - 1
        cloud_mask[iT, bases] = -1
        cloud_mask[iT, tops] = +1
        cloud_prop.append({'idx_cb': bases, 'val_cb': rg_list[bases],  # cloud bases
                           'idx_ct': tops, 'val_ct': rg_list[tops],  # cloud tops
                           'width': [ct - cb for ct, cb in zip(rg_list[tops], rg_list[bases])]
                           })
    return cloud_prop, cloud_mask


def get_indices(categories, cat_list):
    if len(cat_list) == 1:
        return categories[cat_list[0]]
    elif len(cat_list) > 1:
        return np.logical_or.reduce(tuple(categories[icat] for icat in cat_list))
    else:
        return np.logical_or.reduce(tuple(categories[icat] for icat in range(len(categories))))
