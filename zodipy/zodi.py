import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import healpy as hp
import matplotlib.pyplot as plt

from zodipy.base import _ZodiComponent
from zodipy.components import Cloud



class Zodi:
    def __init__(self):
        pass




if __name__ == '__main__':
    cloud = Cloud(
        x0=0.011887801,
        y0=0.0054765065,
        z0=0.0021530908,
        inclination=2.0335188,
        omega=77.657956,
        n0=1.1344374e-7,
        alpha=1.3370697,
        beta=4.1415004,
        gamma=0.94206179,
        mu=0.18873176,
    )

    nside = 32
    m = np.zeros(hp.nside2npix(nside))
    for i in range(len(m)):
        vec = hp.pix2vec(nside, i)
        emission = cloud.get_emission(*vec)
        m[i] = emission

hp.mollview(m, norm='hist', coord='EG')
plt.show()
    