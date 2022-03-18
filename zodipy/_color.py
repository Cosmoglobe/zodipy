from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zodipy.data import DATA_DIR


dirbe_table = np.loadtxt(f"{DATA_DIR}/dirbe_color_corr.dat", skiprows=1).transpose()

DIRBE_COLORCORR_TABLES: dict[str, NDArray[np.floating]] = {
    f"band{idx}": np.asarray((dirbe_table[0], band))
    for idx, band in enumerate(dirbe_table[1:], start=1)
}
