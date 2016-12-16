import util
import numpy as np
import pytest


def test_plot_categories():
    categories = list('abcdefg')
    y = {c: i for i, c in enumerate(categories)}

    util.plot_categories(categories, y)


@pytest.mark.parametrize('linewidths', [None, np.random.gamma(1, size=50)])
def test_plot_edges(linewidths):
    coordinates = np.random.normal(0, 1, (100, 2))
    edgelist = np.random.randint(0, 100, (50, 2))
    util.plot_edges(edgelist, coordinates, linewidths)
