from __future__ import annotations

from zodipy.component import (
    Band,
    BroadBand,
    Cloud,
    Comet,
    ComponentLabel,
    Fan,
    Feature,
    FeatureRRM,
    Interstellar,
    NarrowBand,
    Ring,
    RingRRM,
    ZodiacalComponent,
)

DIRBE: dict[ComponentLabel, ZodiacalComponent] = {
    ComponentLabel.CLOUD: Cloud(
        x_0=0.0119,                 #AU
        y_0=0.00548,                #AU
        z_0=-0.00215,               #AU
        i=2.03,                     #degrees
        Omega=77.7,                 #degrees
        n_0=1.13e-07,               #AU^-1
        alpha=1.34,
        beta=4.14,
        gamma=0.942,
        mu=0.189,
    ),
    ComponentLabel.BAND1: Band(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.56,                     #degrees
        Omega=80.0,                 #degrees
        n_0=5.59e-10,               #AU^-1
        delta_zeta=8.78,            #degrees
        v=0.10,
        p=4.0,
        delta_r=1.5,                #AU
    ),
    ComponentLabel.BAND2: Band(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=1.2,                      #degrees
        Omega=30.3,                 #degrees
        n_0=1.99e-09,               #AU^-1
        delta_zeta=1.99,            #degrees
        v=0.90,
        p=4.0,
        delta_r=0.94,               #AU
    ),
    ComponentLabel.BAND3: Band(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.8,                      #degrees
        Omega=80.0,                 #degrees
        n_0=1.44e-10,               #AU^-1
        delta_zeta=15.0,            #degrees
        v=0.05,
        p=4.0,
        delta_r=1.5,                #AU
    ),
    ComponentLabel.RING: Ring(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.49,                     #degrees
        Omega=22.3,                 #degrees
        n_0=1.83e-08,               #AU^-1
        R=1.03,                     #AU
        sigma_r=0.025,              #AU
        sigma_z=0.054,              #AU
    ),
    ComponentLabel.FEATURE: Feature(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.49,                     #degrees
        Omega=22.3,                 #degrees
        n_0=1.9e-08,                #AU^-1
        R=1.06,                     #AU
        sigma_r=0.10,               #AU
        sigma_z=0.091,              #AU
        theta=-10.0,                #degrees
        sigma_theta=12.1,           #degrees
    ),
}

COSMOGLOBE: dict[ComponentLabel, ZodiacalComponent] = {
    ComponentLabel.CLOUD: Cloud(
        x_0=0.01508,                #AU
        y_0=0.00088699,             #AU
        z_0=-0.0008395,             #AU
        i=2.186,                    #degrees
        Omega=75.66,                #degrees
        n_0=1.068e-07,              #AU^-1
        alpha=1.336,
        beta=3.863,
        gamma=0.9072,
        mu=0.2208,
    ),
    ComponentLabel.BAND1: Band(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.56,                     #degrees
        Omega=80.0,                 #degrees
        n_0=1.972e-10,              #AU^-1
        delta_zeta=8.78,            #degrees
        v=0.10,
        p=4.0,
        delta_r=1.5,                #AU
    ),
    ComponentLabel.BAND2: Band(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=1.2,                      #degrees
        Omega=30.3,                 #degrees
        n_0=1.571e-09,              #AU^-1
        delta_zeta=1.99,            #degrees
        v=0.90,
        p=4.0,
        delta_r=0.94,               #AU
    ),
    ComponentLabel.BAND3: Band(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.8,                      #degrees
        Omega=80.0,                 #degrees
        n_0=1.141e-11,              #AU^-1
        delta_zeta=15.0,            #degrees
        v=0.05,
        p=4.0,
        delta_r=1.5,                #AU
    ),
    ComponentLabel.RING: Ring(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.49,                     #degrees
        Omega=22.3,                 #degrees
        n_0=1.83e-08,               #AU^-1
        R=1.03,                     #AU
        sigma_r=0.025,              #AU
        sigma_z=0.054,              #AU
    ),
    ComponentLabel.FEATURE: Feature(
        x_0=0.0,                    #AU
        y_0=0.0,                    #AU
        z_0=0.0,                    #AU
        i=0.49,                     #degrees
        Omega=22.3,                 #degrees
        n_0=1.9e-08,                #AU^-1
        R=1.06,                     #AU
        sigma_r=0.10,               #AU
        sigma_z=0.091,              #AU
        theta=-10.0,                #degrees
        sigma_theta=12.1,           #degrees
    ),
}

PLANCK = DIRBE.copy()
PLANCK.pop(ComponentLabel.RING)
PLANCK.pop(ComponentLabel.FEATURE)

R_MARS = 1.5237
R_ASTEROID_BELT = 3.137
R_KUIPER_BELT = 30

RRM: dict[ComponentLabel, ZodiacalComponent] = {
    ComponentLabel.FAN: Fan(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        P=2.13,  # *
        Q=10.7,  # *
        gamma=1.3,  # *
        Z_0=0.06,  # *
        R_outer=R_MARS,
    ),
    ComponentLabel.COMET: Comet(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        P=2.5,  # *
        gamma=1,  # *
        Z_0=0.06,
        amp=0.37,  # *
        R_inner=R_MARS,  # *
        R_outer=R_KUIPER_BELT,  # *
    ),
    ComponentLabel.INNER_NARROW_BAND: NarrowBand(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        A=0.04,  # * #AMP1
        # A=0.032,  # * #AMP1
        gamma=1,  # *
        # beta_nb=1.42,  # themis
        beta_nb=9.35,  # veritas
        G=0.12,  # *
        # G=0.12,  # *
        R_inner=R_MARS,  # *
        R_outer=R_ASTEROID_BELT,  # *
    ),
    ComponentLabel.OUTER_NARROW_BAND: NarrowBand(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        A=0.032,  # * #AMP2
        # A=0.04,  # * #AMP2
        gamma=1,  # *
        # beta_nb=9.35,  # veritas
        beta_nb=1.42,  # themis
        G=0.6,  # *
        R_inner=R_MARS,  # *
        R_outer=R_ASTEROID_BELT,  # *
    ),
    ComponentLabel.BROAD_BAND: BroadBand(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=2.6,  # *
        Omega=110,  # *
        A=0.051,  # *
        gamma=1,  # *
        beta_bb=9.3,
        sigma_bb=5,
        R_inner=R_MARS,  # *
        R_outer=R_ASTEROID_BELT,  # *
    ),
    ComponentLabel.INTERSTELLAR: Interstellar(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0,
        Omega=0,
        amp=0.010,
    ),
    ComponentLabel.RING_RRM: RingRRM(
        A=0.16,
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0.49,
        Omega=22.3,
        n_0=1,
        R=1.03,
        sigma_r=0.025,
        sigma_z=0.054,
    ),
    ComponentLabel.FEATURE_RRM: FeatureRRM(
        A=0.065,
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0.49,
        Omega=22.3,
        n_0=1,
        R=1.06,
        sigma_r=0.10,
        sigma_z=0.091,
        theta=-10.0,
        sigma_theta=12.1,
    ),
}
