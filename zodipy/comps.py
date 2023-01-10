from __future__ import annotations

from zodipy._constants import R_ASTEROID_BELT, R_KUIPER_BELT, R_MARS
from zodipy._ipd_comps import (
    Band,
    BroadBand,
    Cloud,
    Comet,
    Component,
    ComponentLabel,
    Fan,
    Feature,
    FeatureRRM,
    Interstellar,
    NarrowBand,
    Ring,
    RingRRM,
)

DIRBE: dict[ComponentLabel, Component] = {
    ComponentLabel.CLOUD: Cloud(
        x_0=0.011887800744346281,
        y_0=0.0054765064662263777,
        z_0=-0.0021530908020710744,
        i=2.0335188072390769,
        Omega=77.657955554097114,
        n_0=1.1344373881427960e-07,
        alpha=1.3370696705930281,
        beta=4.1415004157586637,
        gamma=0.94206179393358036,
        mu=0.18873176489090190,
    ),
    ComponentLabel.BAND1: Band(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0.56438265154389733,
        Omega=80.0,
        n_0=5.5890290403228370e-10,
        delta_zeta=8.7850534408713035,
        v=0.10000000149011612,
        p=4.0,
        delta_r=1.5,
    ),
    ComponentLabel.BAND2: Band(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.2000000476837158,
        Omega=30.347475578624532,
        n_0=1.9877609422590801e-09,
        delta_zeta=1.9917032425777641,
        v=0.89999997615814209,
        p=4.0,
        delta_r=0.94121881201651147,
    ),
    ComponentLabel.BAND3: Band(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0.80000001192092896,
        Omega=80.0,
        n_0=1.4369827283512384e-10,
        delta_zeta=15,
        v=0.050000000745058060,
        p=4.0,
        delta_r=1.5,
    ),
    ComponentLabel.RING: Ring(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0.48707166006819241,
        Omega=22.278979678854448,
        n_0=1.8260527826501675e-08,
        R=1.0281924326308751,
        sigma_r=0.025000000372529030,
        sigma_z=0.054068037356978099,
    ),
    ComponentLabel.FEATURE: Feature(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=0.48707166006819241,
        Omega=22.278979678854448,
        n_0=2.0094267183590947e-08,
        R=1.0579182694524214,
        sigma_r=0.10287314662396611,
        sigma_z=0.091442963768716023,
        theta=-10.0,
        sigma_theta=12.115210933938741,
    ),
}

PLANCK = DIRBE.copy()
PLANCK.pop(ComponentLabel.RING)
PLANCK.pop(ComponentLabel.FEATURE)


RRM: dict[ComponentLabel, Component] = {
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
    ComponentLabel.INNER_NARROW_BAND: NarrowBand(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        A=0.032,  # * #AMP1
        gamma=1,  # *
        beta_nb=1.42,  # themis
        # beta_nb=9.35,  # veritas
        G=0.12,  # *
        R_inner=R_MARS,  # *
        R_outer=R_ASTEROID_BELT,  # *
    ),
    ComponentLabel.OUTER_NARROW_BAND: NarrowBand(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        A=0.04,  # * #AMP2
        gamma=1,  # *
        beta_nb=9.35,  # veritas
        # beta_nb=1.42,  # themis
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
        sigma_bb=6,
        R_inner=R_MARS,  # *
        R_outer=R_ASTEROID_BELT,  # *
    ),
    ComponentLabel.COMET: Comet(
        x_0=0.0,
        y_0=0.0,
        z_0=0.0,
        i=1.5,  # *
        Omega=78,  # *
        P=2.5,  # *
        gamma=1,  # *
        amp=0.37,  # *
        R_inner=R_MARS,  # *
        R_outer=R_KUIPER_BELT,  # *
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
