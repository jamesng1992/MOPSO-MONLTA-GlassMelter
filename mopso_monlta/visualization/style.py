"""
Publication-quality figure styling.

Provides MATLAB-compatible color palettes and rcParams for
two-column journal figures.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


# Colorblind-friendly palette (Okabe-Ito)
COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'green': '#009E73',
    'purple': '#CC79A7',
    'cyan': '#56B4E9',
    'red': '#A2142F',
    'yellow': '#EDB120',
    'black': '#000000',
}

# MATLAB default color order (for 7-state model applications)
MATLAB_COLORS = {
    'blue': '#0072BD',
    'orange': '#D95319',
    'yellow': '#EDB120',
    'purple': '#7E2F8E',
    'green': '#77AC30',
    'cyan': '#4DBEEE',
    'red': '#A2142F',
    'black': '#000000',
}


def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures.

    MATLAB-like styling with inward ticks and clean gridless appearance.
    """
    plt.style.use('default')
    mpl.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'lines.linewidth': 2.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
    })
