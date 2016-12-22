import util
import numpy as np
import pytest


@pytest.mark.parametrize('yerr', [
    None,
    'single',
    'tuple'
])
def test_plot_categories(yerr):
    categories = list('abcdefg')
    y = {c: i for i, c in enumerate(categories)}
    if yerr == 'single':
        yerr = {c: 1 for c in categories}
    elif yerr == 'tuple':
        yerr = {c: (1, 2) for c in categories}

    util.plot_categories(categories, y, yerr)


@pytest.mark.parametrize('linewidths', [None, np.random.gamma(1, size=50)])
def test_plot_edges(linewidths):
    coordinates = np.random.normal(0, 1, (100, 2))
    edgelist = np.random.randint(0, 100, (50, 2))
    util.plot_edges(edgelist, coordinates, linewidths)
