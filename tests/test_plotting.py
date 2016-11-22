import util


def test_plot_categories():
    categories = list('abcdefg')
    y = {c: i for i, c in enumerate(categories)}

    util.plot_categories(categories, y)
