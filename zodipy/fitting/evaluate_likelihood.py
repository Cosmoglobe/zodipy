import numpy as np

def eval_like(bmap,rmsmap,template,mask):

    npix1 = np.shape(bmap)[0]
    npix2 = np.shape(template)[0]
    
    if npix1 != npix2:
        print(f"Map and template have differing nsides: {npix1} != {npix2}")
        exit()

    like = -0.5*np.sum((bmap[mask]-template[mask])**2/(rmsmap[mask]**2))

    return like
