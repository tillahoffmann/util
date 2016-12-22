import itertools as it
import logging
import os
from matplotlib import pyplot as plt, transforms as mtransforms, collections as mcollections
import numpy as np
from scipy.stats import gaussian_kde
import numbers

from .util import autospace, acorr


logger = logging.getLogger(__name__)


def kde_plot(x, factor=0.1, ax=None, **kwargs):
    """
    Plot a univariate kernel density estimate.

    Parameters
    ----------
    x : array_like
        values to plot
    ax : Axes
        axes to plot into
    factor : float
        factor by which to extend the range
    kwargs : dict
        additional keyword arguments passed to `ax.plot`
    """
    ax = ax or plt.gca()
    kde = gaussian_kde(x)
    linx = autospace(x, factor=factor)
    y = kde(linx)
    return ax.plot(linx, y, **kwargs)


def density_plot(samples, burn_in=0, name=None, value=None, bins=10, ax=None, **kwargs):
    """
    Plot the density of a parameter (and a vertical indicating the true value).

    Parameters
    ----------
    samples : array_like
        samples of the parameters
    burn_in : int
        number of initial values to discard
    name : str
        name of the parameter
    value : float
        true value
    bins : int
        number of bins for the histogram
    ax : Axes
        axes to plot into
    """
    ax = ax or plt.gca()
    x = samples[burn_in:]

    # Create a histogram
    if bins is not None:
        ax.hist(x, bins, normed=True, histtype='stepfilled', facecolor='silver', edgecolor='none')

    # Plot the kde
    kde_plot(x, ax=ax, **kwargs)
    if name:
        ax.set_xlabel(name)

    # Plot true values
    if value is not None:
        ax.axvline(value, ls='dotted', color='r')


def grid_density_plot(samples, burn_in=0, parameters=None, values=None, nrows=None, ncols=None, bins=10, **kwargs):
    """
    Plot the marginal densities of parameters  (and vertical lines indicating the true values).

    Parameters
    ----------
    samples : array_like
        samples of the parameters
    burn_in : int
        number of initial values to discard
    parameters : dictionary
        indices of the parameters to plot as keys and names as values (default is all)
    values : iterable
        true values corresponding to the indices in `parameters`
    nrows : int
        number of rows in the plot
    ncols : int
        number of columns in the plot
    bins : int
        number of bins for the histograms
    """
    if parameters is None:
        parameters = {i: str(i) for i in np.arange(samples.shape[1])}
    if values is None:
        values = []

    # Determine the number of rows and columns if not specified
    n = len(parameters)
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(float(n) / ncols))
    elif nrows is None:
        nrows = int(np.ceil(float(n) / ncols))
    elif ncols is None:
        ncols = int(np.ceil(float(n) / nrows))

    fig, axes = plt.subplots(nrows, ncols)

    # Plot all parameters
    for ax, parameter, value in it.zip_longest(np.ravel(axes), parameters, values, fillvalue=None):
        # Skip if we have run out of parameters
        if parameter is None:
            break

        # Plot the individual density estimate
        density_plot(samples[:, parameter], burn_in, parameters[parameter], value, bins, ax, **kwargs)


    fig.tight_layout()

    return fig, axes


def trace_plot(samples, fun_values, burn_in=0, parameters=None, values=None):
    """
    Plot the trace of parameters (and horizontal lines indicating the true values).

    Parameters
    ----------
    samples : array_like
        samples of the parameters
    fun_values : array_like
        values of the objective function
    burn_in : int
        number of initial values to discard
    parameters : iterable
        indices of the parameters to plot (default is all)
    values : iterable
        true values corresponding to the indices in `parameters`
    """
    if parameters is None:
        parameters = {i: str(i) for i in range(samples.shape[1])}
    if values is None:
        values = []

    fig, (ax1, ax2) = plt.subplots(1, 2, True)

    # Plot the trace
    for parameter, value in it.zip_longest(parameters, values, fillvalue=None):
        line, = ax1.plot(samples[burn_in:, parameter], label=parameters[parameter])
        # Plot the true values
        if value is not None:
            ax1.axhline(value, ls='dotted', color=line.get_color())

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Parameter values')
    ax1.legend(loc=0, frameon=False)

    ax2.plot(fun_values[burn_in:])
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Function values')

    fig.tight_layout()

    x = samples[burn_in:]

    logger.info("acceptance ratio %f" , np.mean(x[1:] != x[:-1]))

    return fig, (ax1, ax2)


def autocorrelation_plot(samples, parameters=None, lag=50, ax=None, **kwargs):
    """
    Plot the autocorrelation of samples.

    Parameters
    ----------
    samples : np.ndarray
        samples to plot the autocorrelation for
    parameters : dict or None
        dictionary of parameter names keyed by index
    lag : int
        maximum lag to consider
    ax
        axes to plot into
    kwargs : dict
        keyword arguments passed on to `ax.plot`
    """
    ax = ax or plt.gca()
    if parameters is None:
        parameters = {i: str(i) for i in range(samples.shape[1])}

    # Plot the autocorrelation
    lines = []
    for parameter, label in parameters.items():
        line, = ax.plot(acorr(samples[:, parameter])[:lag], label=label, **kwargs)
        lines.append(line)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Autocorrelation')
    ax.legend(loc=0, frameon=False)

    return lines


def comparison_plot(samples, values, burn_in=0, parameters=None, ax=None, **kwargs):
    """
    Plot inferred against actual parameters.

    Parameters
    ----------
    samples
    values
    burn_in
    parameters

    Returns
    -------

    """
    if parameters is None:
        parameters = range(samples.shape[1])

    assert len(parameters) == len(values), "shape mismatch"

    data = []
    for parameter, value in zip(parameters, values):
        x = samples[burn_in:, parameter]
        data.append((value, np.mean(x), np.std(x)))

    x, y, yerr = np.transpose(data)
    ax = ax or plt.gca()
    kwargs_default = {
        'ls': 'none',
        'marker': '.',
    }
    kwargs_default.update(kwargs)
    ax.errorbar(x, y, yerr, **kwargs_default)

    xmin, xmax = np.min(values), np.max(values)
    ax.plot((xmin, xmax), (xmin, xmax), color='k', ls=':')


def get_style(style):
    """
    Get matplotlib style specification.

    Parameters
    ----------
    style : str
        name of the style
    """
    # Use the default style or load it if it is available
    if style in plt.style.available or os.path.exists(style) or style == 'default':
        return style

    # Construct a filename in the package
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stylelib', style + '.mplstyle')
    if os.path.exists(filename):
        return filename

    raise ValueError("could not locate style specification '{}'".format(style))


def savefigs(fig, filename, *formats, **kwargs):
    """
    Save a figure in multiple formats.

    Parameters
    ----------
    fig : Figure
    filename : str
    formats : list
    """
    if formats:
        # Get the base name without extension
        basename, ext = os.path.splitext(filename)
        if ext:
            formats = (ext,) + formats
        # Iterate over all formats
        for format in formats:
            # Prepend a dot if necessary
            if not format.startswith('.'):
                format = '.' + format
            # Save the file
            fig.savefig(basename + format, **kwargs)
    else:
        fig.savefig(filename, **kwargs)


def label_subplots(axes, fmt='({alphabetic})', x=0.05, y=0.95, ha='left', va='top', offset=0, **kwargs):
    """
    Label subplots with ascending characters or numbers.

    Parameters
    ----------
    axes : list
        list of subplots to label
    fmt : str
        format string
    x : float
        horizontal position in axis units
    y : float
        vertical position in axis units
    ha : str
        horizontal text alignment
    va : str
        vertical text alignment
    offset : int
        offset
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for i, ax in enumerate(np.ravel(axes)):
        i += offset
        ax.text(x, y, fmt.format(alphabetic=alphabet[i], numeric=i + 1), transform=ax.transAxes,
                ha=ha, va=va, **kwargs)


def rescaled_axes(scale_x=1, scale_y=1, tx=0, ty=0, ax=None):
    """
    Create a linearly rescaled copy of an axis.
    """
    if scale_x == 1 and scale_y == 1:
        return None

    ax = ax or plt.gca()
    assert hasattr(ax, 'twin'), "the axes must support the `twin` function"

    aux_trans = mtransforms.Affine2D().translate(tx, ty).scale(scale_x, scale_y)
    ax2 = ax.twin(aux_trans)

    # Hide some of the labels if the scale is not different from one
    if scale_x == 1:
        ax2.axis["top"].toggle(all=False, ticks=True)

    if scale_y == 1:
        ax2.axis["right"].toggle(all=False, ticks=True)

    return ax2


def plot_categories(categories, y, yerr=None, shift=0, ticks=True, tick_rotation=-45, ax=None, **kwargs):
    """
    Plot values for different categories (with errors).

    Parameters
    ----------
    categories : list[str]
        category names
    y : dict[float]
        values for categories keyed by name
    yerr : dict[float]
        errors for the categories keyed by name
    shift : float
        shift the data points by a small amount
    ticks : bool
        whether to show ticks
    tick_rotation : float
        angle by which to rotate ticks
    ax
        axes to plot into
    kwargs : dict
        keyword arguments passed on to ax.errorbar
    """
    # Get default axes
    ax = ax or plt.gca()

    # Add labels if desired
    x = np.arange(len(categories))
    if ticks:
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=tick_rotation)


    y = [y.get(category, np.nan) for category in categories]
    if yerr is not None:
        if yerr and not isinstance(next(iter(yerr.values())), numbers.Number):
            missing = (np.nan, np.nan)
        else:
            missing = np.nan
        yerr = np.asarray([yerr.get(category, missing) for category in categories])
        if np.ndim(yerr) == 2:
            yerr = np.transpose(yerr)

    return ax.errorbar(x + shift, y, yerr, **kwargs)


def plot_edges(edgelist, coordinates, linewidths=None, ax=None, **kwargs):
    """
    Plot edges.

    Parameters
    ----------
    edgelist : np.ndarray
        `m` by `2` array of `m` edges
    coordinates : np.ndarray
        `n` by `2` array of coordinates for `n` nodes
    linewidths : np.ndarray
        `m` line widths (edges with line width of zero are not plotted)
    ax :
        axes to plot into
    kwargs : dict
        keyword arguments passed on to `LineCollection`

    Returns
    -------
    edges : LineCollection
        collection of lines representing the edges
    """
    ax = ax or plt.gca()
    if linewidths is None:
        linewidths = np.ones(len(edgelist))
    else:
        linewidths = np.asarray(linewidths)

    i, j = np.transpose(edgelist)

    # Assemble segments and line widths
    segments = []
    _linewidths = []
    for i, j, w in zip(i, j, linewidths):
        if w:
            segments.append((coordinates[i], coordinates[j]))
            _linewidths.append(w)

    # Add to the axes
    edges = mcollections.LineCollection(segments, _linewidths, **kwargs)
    ax.add_artist(edges)
    return edges
