# -*- coding: utf-8 -*-
"""Plot distribution of datasets"""

import base64
from distutils.version import LooseVersion
import dask_cudf_profiling.base as base
import dask.array as da
import dask.dataframe as dd
import matplotlib
import numpy as np
import cupy
import time

#dask_cudf profiling edit. 
#Adding verbose call
verbose = True

# Fix #68, this call is not needed and brings side effects in some use cases
# Backend name specifications are not case-sensitive; e.g., ‘GTKAgg’ and ‘gtkagg’ are equivalent.
# See https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
BACKEND = matplotlib.get_backend()
if matplotlib.get_backend().lower() != BACKEND.lower():
    # If backend is not set properly a call to describe will hang
    matplotlib.use(BACKEND)
from matplotlib import pyplot as plt
try:
    from StringIO import BytesIO
except ImportError:
    from io import BytesIO
try:
    from urllib import quote
except ImportError:
    from urllib.parse import quote

# TODO: We can reduce the histogram time but just not computing twice, once for histogram and once for mini-histogram

def get_histogram_freq_edges(series, bins=10):
    start = time.time()
    series = series.compute()
    end = time.time()

    if verbose:
        #time-profiling
        print("Total time elapsed in computing histogram series.compute() ", end-start)

    # Time-profiling
    start = time.time()
    series = series.dropna()
    end = time.time()

    if verbose:
        #time-profiling
        print("Total time elapsed in computing histogram series.dropna() ", end-start)

    # Time-profiling
    start = time.time()
    frequencies, edges = cupy.histogram(x=cupy.array(series) , bins=bins)
    end = time.time()

    if verbose:
        #time-profiling
        print("Total time elapsed in computing histogram cupy.histogram() ", end-start)
    
    return (frequencies, edges)


def _plot_histogram(frequencies, edges, figsize=(6, 4), facecolor='#337ab7'):
    """Plot an histogram from the data and return the AxesSubplot object.

    Parameters
    ----------
    series : Series
        The data to plot
    figsize : tuple
        The size of the figure (width, height) in inches, default (6,4)
    facecolor : str
        The color code.

    Returns
    -------
    matplotlib.AxesSubplot
        The plot.
    """
    fig = plt.figure(figsize=figsize)
    plot = fig.add_subplot(111)

    #dask_cudf_profiling edit
    #converting to cudf.core.Series and then to pandas Series
    #print("Converting dask_cudf.Series to pandas.Series for plotting histogram")
    #series = series.compute()
    #series = series.to_pandas()

    #dask_cudf_profiling edit
    #converting to cudf.core.Series and using cupy.histogram to get the histogram frequency and edges
    # Time-profiling
    

    # frequencies, edges = cupy.histogram(x=cupy.array(series) , bins=bins)
    center = (edges[:-1] + edges[1:]) / 2
    plot.bar(center.tolist(), frequencies.tolist(), facecolor=facecolor)

    #plot.hist(series.dropna().values, bins=bins, facecolor=facecolor)  # TODO: make da.histogram work

    return plot


def histogram(frequencies, edges, **kwargs):
    """Plot an histogram of the data.

    Parameters
    ----------
    series: Series
        The data to plot.

    Returns
    -------
    str
        The resulting image encoded as a string.
    """
    imgdata = BytesIO()
    plot = _plot_histogram(frequencies, edges, **kwargs)
    plot.figure.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0)
    plot.figure.savefig(imgdata)
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
    # TODO Think about writing this to disk instead of caching them in strings
    plt.close(plot.figure)
    return result_string


def mini_histogram(frequencies, edges, **kwargs):
    """Plot a small (mini) histogram of the data.

    Parameters
    ----------
    series: Series
        The data to plot.

    Returns
    -------
    str
        The resulting image encoded as a string.
    """
    imgdata = BytesIO()
    plot = _plot_histogram(frequencies, edges, figsize=(2, 0.75), **kwargs)
    plot.axes.get_yaxis().set_visible(False)

    if LooseVersion(matplotlib.__version__) <= '1.5.9':
        plot.set_axis_bgcolor("w")
    else:
        plot.set_facecolor("w")

    xticks = plot.xaxis.get_major_ticks()
    for tick in xticks[1:-1]:
        tick.set_visible(False)
        tick.label.set_visible(False)
    for tick in (xticks[0], xticks[-1]):
        tick.label.set_fontsize(8)
    plot.figure.subplots_adjust(left=0.15, right=0.85, top=1, bottom=0.35, wspace=0, hspace=0)
    plot.figure.savefig(imgdata)
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
    plt.close(plot.figure)
    return result_string

def get_histograms(series, **kwargs):
    """Plot a histogram and mini-histogram of the data.

    Parameters
    ----------
    series: Series
        The data to plot.

    Returns
    -------
    tuple(str,str)
        resulting image of histogram and mini-histogram encoded as a string
    """
    #Getting histogram stats from the series 
    frequencies, edges = get_histogram_freq_edges(series)
    #Histogram code
    hist_result_string = histogram(frequencies, edges)
    
    #Mini-Histogram code
    mini_hist_result_string = mini_histogram(frequencies, edges)

    return (hist_result_string, mini_hist_result_string)
    


def correlation_matrix(corrdf, title, **kwargs):
    """Plot image of a matrix correlation.
    Parameters
    ----------
    corrdf: DataFrame
        The matrix correlation to plot.
    title: str
        The matrix title
    Returns
    -------
    str, The resulting image encoded as a string.
    """
    imgdata = BytesIO()
    fig_cor, axes_cor = plt.subplots(1, 1)
    labels = corrdf.columns
    if verbose:
        #dask_cudf profiling edit
        print("Data type at plot.correlation_matrix",type(corrdf),flush=True)
    #converting the dataframe to float
    corrdf = corrdf.astype("float32")
    matrix_image = axes_cor.imshow(corrdf, vmin=-1, vmax=1, interpolation="nearest", cmap='bwr')
    plt.title(title, size=18)
    plt.colorbar(matrix_image)
    axes_cor.set_xticks(np.arange(0, corrdf.shape[0], corrdf.shape[0] * 1.0 / len(labels)))
    axes_cor.set_yticks(np.arange(0, corrdf.shape[1], corrdf.shape[1] * 1.0 / len(labels)))
    axes_cor.set_xticklabels(labels, rotation=90)
    axes_cor.set_yticklabels(labels)

    matrix_image.figure.savefig(imgdata, bbox_inches='tight')
    imgdata.seek(0)
    result_string = 'data:image/png;base64,' + quote(base64.b64encode(imgdata.getvalue()))
    plt.close(matrix_image.figure)
    return result_string
