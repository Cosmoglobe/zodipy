import numpy as np
import random

def fit_template(bmap,rmsmap,template,mask,sample: bool):

    npix1 = np.shape(bmap)[0]
    npix2 = np.shape(template)[0]
    
    if npix1 != npix2:
        print(f"Map and template have differing nsides: {npix1} != {npix2}")
        exit()
                
    sum1 = np.sum((bmap[mask]*template[mask])/(rmsmap[mask]**2))
    sum2 = np.sum((template[mask]**2)/(rmsmap[mask]**2))
    
    norm = np.sum(template[mask]/rmsmap[mask])

    if sample:
        amp = sum1/sum2 + random.random()/np.sqrt(norm)
    else:
        amp = sum1/sum2

    return amp
