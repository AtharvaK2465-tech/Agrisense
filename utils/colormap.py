"""
Converts raw index/confidence arrays into RGBA image arrays for the
Leaflet map overlay and inline display — a proper red->yellow->green NDVI
colormap instead of flat grayscale, which is both more legible and closer
to what agronomists expect from a vegetation index map.
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def ndvi_to_rgba(ndvi: np.ndarray, vmin: float = -0.2, vmax: float = 0.9) -> np.ndarray:
    """Maps NDVI values to an RdYlGn colormap (red=stressed, green=healthy).
    Returns a float RGBA array in [0, 1], shape (H, W, 4)."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return cm.get_cmap("RdYlGn")(norm(ndvi))


def confidence_to_rgba(confidence: np.ndarray) -> np.ndarray:
    """Maps a 0-1 confidence array to a viridis colormap (dark=low
    confidence, bright yellow=high confidence)."""
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
    return cm.get_cmap("viridis")(norm(confidence))
