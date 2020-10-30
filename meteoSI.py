# -*- coding: utf-8 -*-
# met formeln fuer python
#
# ----------------------------------------------------------
# (c) Jan Schween 2005 (gnuplot) -> Mario Mech 2009 (python)
# converted to sSI units
# vaphet, pseudoAdiabLapseRate, rh2a, moist_rho2rh added by Max Maahn 2011
# ----------------------------------------------------------
#
from __future__ import division, print_function
# from math import *
# import sys
import numpy as np
import warnings
from numba import jit

try:
    import numexpr as ne

    neAvail = True
except:
    warnings.warn("numexpr not available", Warning)
    neAvail = False

Grav = 9.80665  # m/s^2 der Wert fuer mittlere Breiten
Rair = 287.04*1.e-3  # J/kg/K
Rvapor = 461.5  # J/kg/K
Cp = 1005.0  # J/kg/K specific heat capacity
Gamma = -Grav / Cp  # =-0.0097..K/m  adiabatic temperature gradient
Lv = 2.5e6  # J/kg  bei 0C Lv heat of vaporization
Mwml = 0.622  # dimlos, Molmassenverhaeltnis
Tnull = -273.15  # degC absolute zero
Kadiab = Rair / Cp  # dimensionless adiabatic exponenet
g = 9.80665  # gravitational acceleration

missingNumber = -9999

@jit(nopython=True, parallel=True)
def DENSITY(RH, T, P):
    '''
    ;                                                                      *
    ;   D E N S I T Y                                                      *
    ;                                                                      *
    ;   This function calculates the density in kg/m**3 from air tempe-    *
    ;   rature , relative humidity and pressure by equation of state of    *
    ;   ideal gas.                                                         *
    ;                                                                      *
    ;        DENSITY = P / ( R * TVIRT )                                   *
    ;                                                                      *
    ;                  R   :   gas constant =287.04 [ J/kg K ] (PICHLER )  *
    ;                                                                      *
    ;   Input:                                                             *
    ;                 RH  :   relative humidity [ 1 ]                      *
    ;                 T   :   temperature       [ K ]                      *
    ;                 P   :   pressure          [ Pa ]                     *
    ;
    '''
    TV = TVIRT(RH, T, P)
    return P / (TV * Rair)


@jit(nopython=True, parallel=True)
def TVIRT(RH, T, P):
    '''
    ;                                                                      *
    ;   T V I R T                                                          *
    ;                                                                      *
    ;   This function calculates the virtual temperature from the relative *
    ;   humidity , the air temperature and the total air pressure.         *
    ;                                                                      *
    ;          TVIRT = T * ( 1 + 0.00378 * ( RH * ESTET / P )              *
    ;                                                                      *
    ;   Input:                                                             *
    ;                 RH  :   relative humidity [ 1 ]                      *
    ;                 T   :   temperature       [ K ]                      *
    ;                 P   :   pressure          [ Pa ]                     *
    ;                                                                      *
    ;   Output: -                                                          *
    ;                                                                      *
    ;   Source : Liljequist and Cehak                                      *
    ;                                                                      *
    ;   Date of last changes : 01.09.1992
    '''
    Es = ESTET(T)

    return T * (1. + 0.00378 * (RH * Es / P))

@jit(nopython=True, parallel=True)
def ESTET(T):
    '''
    ;                                                                      *
    ;   E S T E T                                                          *
    ;                                                                      *
    ;   This function computes the saturation vapor pressure above a plain *
    ;   water surface in Pa . The TETENS formula is used.                  *
    ;                                                                      *
    ;     ESTET = 611.0 * 10 ** ( 7.5 * (T - 273.15 ) / ( T - 35.85 ) )    *
    ;                                                                      *
    ;   Input:        T   :   temperature       [ K ]                      *
    ;                                                                      *
    ;   Output: -                                                          *
    ;                                                                      *
    ;   Source :                                                           *
    ;   [Elliot,Gaffen : On the utility of Radiosonde Humidity Archievs    *
    ;   for Climate Studies ; Bulletin American Meteorological Society ,   *
    ;   Vol. 72 ,  No. 10, October 1991 ]                                  *
    ;                                                                      *
    ;   Date of last changes : 01.09.1992
    '''
    #Ttmp = T+Tnull
    #Ttmp = T
    #return 6.107e2 * np.exp((17.08 * Ttmp) / (234.2 + Ttmp))
    return 611.0 * 10.0 ** (7.5 * (T + Tnull) / (T - 35.85))

@jit(nopython=True, parallel=True)
def ABHUM(T, RH):
    """
    ;                                                                      *
    ;   A B H U M                                                          *
    ;                                                                      *
    ;  This function calculates the absolute humidity in kg/m**3 from      *
    ;  air temperature T and relative humidity RH.                         *
    ;                                                                      *
    ;  ABHUM = MW * ES * RH / ( R * T )                                    *
    ;                                                                      *
    ;  Input                                                               *
    ;         T     :   temperature   [ K ]                                *
    ;         RH    :   rel. humidity [ 1 ]                                *
    ;                                                                      *
    ;  Output                                                              *
    ;         ABHUM :   absolute humidity [ kg/m**3 ]                      *
    ;                                                                      *
    ;  Constants                                                           *
    ;         MW    :   molecular weight ( =18.016 [kg/kmol] )             *
    ;         R     :   gas constant   ( =8314.3 [J/kmol K] )              *
    ;                                                                      *
    ;   Functions                                                          *
    ;
    :return:
    """

    MW = 18.016
    R = 8314.3

    ES = ESTET(T)
    return (MW * ES * RH) / (R * T)

@jit(nopython=True, parallel=True)
def VAPHET(T):
    """
    ;   Compute specific heat of vaporisation                             *
    ;                                                                     *
    ;   Input  : T      [K]      thermodynamic temperature                *
    ;                                                                     *
    ;   Output : VAPHET [J/kg]   specific heat of vaporisation            *
    ;                                                                     *
    ;   Source: Liljequist, G.H. und K. Cehak, 1984: Allgemeine           *
    ;           Meteorologie (III.Ed.). Vieweg, 396p. Page 95             *
    ;
    """
    return (2500.8-2.372*(T-273.15)) * 1000.0


@jit(nopython=True, parallel=True)
def PSEUDO(T, WS):
    """
    ;                                                                     *
    ;   Pseudoadiabatic lapse rate                                        *
    ;                                                                     *
    ;   Input: T   [K]  thermodynamic temperature                         *
    ;          W   [1]  mixing ratio of saturation                        *
    ;                                                                     *
    ;   Output: PSEUDO [K/m] pseudoadiabatic lapse rate                   *
    ;                                                                     *
    ;   Constants: G   [m/s2]     : constant of acceleration              *
    ;              CP  [J/(kg K)] : specific heat cap. at const. press.   *
    ;              RL  [J/(kg K)] : gas constant of dry air               *
    ;              RW [J/(kg K)] : gas constant of water vapor           *
    ;                                                                     *
    ;   Source: Rogers and Yau, 1989: A Short Course in Cloud Physics     *
    ;           (III.Ed.), Pergamon Press, 293p. Page 32                  *
    ;
    """

    RL = 287.05
    RW = 461.5

    L = VAPHET(T)

    return (Grav / Cp) * (1 + (L * WS / RL / T)) / (1 + (WS * L*L / Cp / RW / T*T))


def adiab(i, T, P, RH, z):
    """
    Adiabtic liquid water content assuming pseudoadiabatic lapse rate
    throughout the whole cloud layer. Thus the assumed temperature
    profile is differnt from the measured one
    Input:
    i no of levels
    T is in K
    p is in Pa
    z is in m
    Output:
    LWC in
    translated to Python from adiab.pro by mx.
    """

    #   Set actual cloud base temperature to the measured one
    #   Initialize Liquid Water Content (LWC)
    #   Compute adiabatic LWC by integration from cloud base to level I

    TCL = T[0]
    LWC = 0.0

    for j in range(i):
    #for j in range(1, i + 1):
        deltaz = z[j] - z[j-1]

        #   Compute actual cloud temperature

        #   1. Compute air density
        #   2. Compute water vapor density of saturated air
        #   3. Compute mixing ratio of saturated air
        #   4. Compute pseudoadiabatic lapse rate
        #   5. Compute actual cloud temperature

        R = DENSITY(1.0, T[j], P[j])
        RWV = ABHUM(T[j], 1.0)
        WS = RWV/(R - RWV)
        DTPS = PSEUDO(T[j], WS)
        TCL = TCL + DTPS * deltaz

        #   Compute adiabatic LWC

        #   1. Compute air density
        #   2. Compute water vapor density of saturated air
        #   3. Compute mixing ratio of saturated air
        #   4. Compute specific heat of vaporisation
        #   5. Compute adiabatic LWC

        R = DENSITY(1.0, TCL, P[j])
        RWV = ABHUM(TCL, 1.0)

        WS = RWV / (R - RWV)
        L = VAPHET(TCL)
        LWC += R * Cp / L * ((Grav / Cp) - PSEUDO(TCL, WS)) * deltaz

    return LWC


def mod_ad(T_cloud, p_cloud, rh_cloud, z_cloud, fak=1.):
    """

    Args:
        T_cloud: temperature values for cloudy pixel only
        p_cloud: pressure values for cloudy pixel only
        z_cloud: range values containing for cloudy pixel only
        fak: ???

    Returns:
        liquid water content lwc, assuming adiabaticity

    """

    # ;IN: T_cloud, p_cloud, z_cloud
    # ;OUT: lwc
    # ;Einheiten: SI!
    # translated to Python from mod_ad.pro by mx.

    n_cloud_pixel = len(T_cloud)
#    lwc = np.zeros(n_cloud_pixel - 1)
#
#    for jj in range(n_cloud_pixel - 1):
#        thick = z_cloud[jj + 1] - z_cloud[jj]
#        lwc[jj] = adiab(jj + 1, T_cloud, p_cloud, rh_cloud, z_cloud)
#        lwc[jj] *= 1.239387 - 0.144779 * np.log(thick / fak)

    thick = z_cloud[-1] - z_cloud[0]
    lwc = -adiab(n_cloud_pixel, T_cloud, p_cloud, rh_cloud, z_cloud)
    lwc *= (1.239387 - 0.144779 * np.log(thick / fak))

    return lwc


#
#def moist_rho_rh(p, T, rh, *qm):
#    '''
#    Input:
#    p is in Pa
#    T is in K
#    rh is in Pa/Pa
#    Optional, several possible:
#    qm is in kg/kg other species which contribute to the air mass! (ice, snow, cloud etc.)
#
#    Output:
#    density of moist air [kg/m^3]
#
#    Example:
#    moist_rho_rh(p,T,rh,q_ice,q_snow,q_rain,q_cloud,q_graupel,q_hail)
#
#
#    '''
#    if np.any(rh > 5):
#        raise TypeError("rh must not be in %")
#
#    q = rh2q(rh, T, p)
#
#    return moist_rho_q(p, T, q, *qm)
#
#
#def moist_rho_q(p, T, q, *qm):
#    '''
#    Input p is in Pa
#    T is in K
#    q is in kg/kg
#    Optional, several possible:
#    qm is in kg/kg other species which contribute to the air mass! (ice, snow, cloud etc.)
#
#    Output:
#    density of moist air [kg/m^3]
#
#    Example:
#    moist_rho_q(p,T,q,q_ice,q_snow,q_rain,q_cloud,q_graupel,q_hail)
#    '''
#
#    if len(qm) > 0:
#        # get rid of masked data!
#        qm = np.ma.array(qm).filled(0)
#        qm[qm < 0] = 0
#        qm = np.sum(qm, axis=0)
#    else:
#        qm = 0
#
#    moist_rho_q = p / (Rair * T * (1 + (Rvapor / Rair - 1) * q - qm))
#
#    if np.any(moist_rho_q < 0):
#        if np.any(moist_rho_q < -0.001):
#            raise ValueError(
#                "meteoSI.moist_rho_q calculated negative densities!")
#        else:
#            try:
#                moist_rho_q[moist_rho_q < 0] = 0
#            except:
#                moist_rho_q = 0
#
#    return moist_rho_q
#
#
#@jit(nopython=True, parallel=True)
#def T_virt_rh(T, rh, p):
#    """
#    Calculate the virtual temperature from air temperature,
#    pressure, and relative humidity.
#
#    Input:
#    T is in K
#    rh is in Pa/Pa
#    p is in Pa
#
#    Output:
#    T_virt in K
#    """
#    if np.any(rh > 5):
#        raise TypeError("rh must not be in %")
#    return T_virt_q(T, rh2q(rh, T, p))
#
#
#@jit(nopython=True, parallel=True)
#def T_virt_q(T, q):
#    '''
#    Calculate the virtual temperature from air temperature and specific humidity.
#
#    Input:
#    T is in K
#    q is in kg/kg
#
#    Output:
#    T_virt in K
#    '''
#    return T + T * (Rvapor / Rair - 1) * q
#


#A@jit(nopython=True, parallel=True)
#
#def e2q(e, p):
#    '''
#    Calculate the specific humidity from water vapour pressure and air pressure.
#
#    Input:
#    e is in Pa
#    p is in Pa
#
#    Output
#    q in kg/kg
#    '''
#    q = Mwml * e / (p - (1 - Mwml) * e)
#    return q
#
#
#@jit(nopython=True, parallel=True)
#def q2e(q, p):
#    '''
#    Calculate water vapour pressure from the specific humidity and air pressure.
#
#    Input:
#    q in kg/kg
#    p is in Pa
#
#    Output
#    e is in Pa
#    '''
#    e = p / ((Mwml / q) + 1 - Mwml)
#    return e
#
#
#def rh2q(rh, T, p):
#    '''
#    Calculate the specific humidity from relative humidity, air temperature,
#    and pressure.
#
#    Input:
#    T is in K
#    rh is in Pa/Pa
#    p is in Pa
#
#    Output
#    q in kg/kg
#    '''
#    if np.any(rh > 5):
#        raise TypeError("rh must not be in %")
#
#    eStar = e_sat_gg_water(T)
#    e = rh * eStar
#    q = e2q(e, p)
#    del e, eStar
#    return q
#
#
#def rh2a(rh, T):
#    '''
#    Calculate the absolute humidity from relative humidity, air temperature,
#    and pressure.
#
#    Input T is in K
#    rh is in Pa/Pa
#    p is in Pa
#    Output
#    a in kg/m^3
#
#    Source: Kraus: Chapter 8.1.2
#    '''
#
#    if np.any(rh > 5):
#        raise TypeError("rh must not be in %")
#
#    e = rh * e_sat_gg_water(T)
#    a = e / (Rvapor * T)
#    return a
#
#
#@jit(nopython=True, parallel=True)
#def a2rh(a, T):
#    '''
#    Calculate the relative from absolute humidity and air temperature.
#
#    Input
#    T is in K
#    a in kg/kg
#    Output
#    rh in Pa/Pa
#
#    Source: Kraus: Chapter 8.1.2
#    '''
#
#    e = a * (Rvapor * T)
#    rh = e / e_sat_gg_water(T)
#    return rh
#
#
#@jit(nopython=True, parallel=True)
#def q2rh(q, T, p):
#    '''
#    Calculate relative humidity from specific humidity
#
#    Input:
#    T is in K
#    p is in Pa
#    q in kg/kg
#
#    Output:
#    rh is in Pa/Pa
#    '''
#
#    if neAvail:
#        e = ne.evaluate("p/(Mwml*((1/q)+(1/(Mwml)-1)))")
#    else:
#        e = p / (Mwml * ((1 / q) + (1 / (Mwml) - 1)))
#
#    eStar = e_sat_gg_water(T)
#    rh = e / eStar
#    del e, eStar
#    return rh
#
#
#def e_sat_gg_water(T):
#    '''
#    Calculates the saturation pressure over water after Goff and Gratch (1946).
#    It is the most accurate that you can get for a temperture range from -90°C to +80°C.
#    Source: Smithsonian Tables 1984, after Goff and Gratch 1946
#    http://cires.colorado.edu/~voemel/vp.html
#    http://hurri.kean.edu/~yoh/calculations/satvap/satvap.html
#
#    Input:
#    T in Kelvin.
#    Output:
#    e_sat_gg_water in Pa.
#    '''
#    if neAvail:
#        e_sat_gg_water = ne.evaluate(
#            "100 * 1013.246 * 10**( -7.90298*(373.16/T-1) + 5.02808*log10(373.16/T) - 1.3816e-7*(10**(11.344*(1-T/373.16))-1) + 8.1328e-3 * (10**(-3.49149*(373.16/T-1))-1) )")
#    else:
#        e_sat_gg_water = 100 * 1013.246 * 10 ** (-7.90298 * (373.16 / T - 1) + 5.02808 * np.log10(
#            373.16 / T) - 1.3816e-7 * (10 ** (11.344 * (1 - T / 373.16)) - 1) + 8.1328e-3 * (10 ** (-3.49149 * (373.16 / T - 1)) - 1))
#    return e_sat_gg_water
#
#
#@jit(nopython=True, parallel=True)
#def rh_to_iwv(relhum_lev, temp_lev, press_lev, hgt_lev):
#    '''
#    Calculate the integrated water vapour
#
#    Input:
#    T is in K
#    rh is in Pa/Pa
#    p is in Pa
#    z is in m
#
#    Output
#    iwv in kg/m^2
#    '''
#    dz = np.diff(hgt_lev, axis=-1)
#    relhum = (relhum_lev[..., 0:-1] + relhum_lev[..., 1:]) / 2.
#    temp = (temp_lev[..., 0:-1] + temp_lev[..., 1:]) / 2.
#
#    xp = -1. * np.log(press_lev[..., 1:] / press_lev[..., 0:-1]) / dz
#    press = -1. * press_lev[..., 0:-1] / xp * (np.exp(-xp * dz) - 1.) / dz
#
#    q = rh2q(relhum, temp, press)
#    rho_moist = moist_rho_q(press, temp, q)
#
#    return np.sum(q * rho_moist * dz)
#
#
#@jit(nopython=True, parallel=True)
#def detect_liq_cloud(z, t, rh):  # , rh_thres, t_thres):
#
#    # UL NOV 2007
#    # tranlated to python by mx 2011
#    # ***********
#    # INPUT
#    # z: height grid
#    # T: temperature on z
#    # rh: relative humidty on z
#    # rh_thres: relative humidity threshold for the detection on liquid clouds on z
#    # T_thres: don not detect liquid water clouds below this value (scalar)
#    # ***********
#    # OUTPUT
#    # z_top: array of cloud tops
#    # z_base: array of cloud bases
#    # z_cloud: array of cloudy height levels
#    # ***********
#
#    rh_thres = 0.95  # 1
#    t_thres = 253.15  # K
#    # import pdb; pdb.set_trace()
#    n = len(z)
#    # print "!",n
#    # ***determine cloud boundaries
#    # --> layers where mean rh GT rh_thres
#
#    cloud_bound_ind = np.zeros(n, dtype=int)
#    for i in np.arange(0, (n - 1)):
#        # print ((rh[i + 1] + rh[i]) / 2. > rh_thres)
#        # print ((t[i + 1] + t[i]) / 2. > t_thres)
#        if ((rh[i + 1] + rh[i]) / 2. > rh_thres) and ((t[i + 1] + t[i]) / 2. > t_thres):
#            cloud_bound_ind[i] = np.bitwise_or(1, cloud_bound_ind[i])
#            cloud_bound_ind[i + 1] = np.bitwise_or(2, cloud_bound_ind[i + 1])
#        # end if
#    # end for
#    # import pdb; pdb.set_trace()
#    i_cloud = np.where(cloud_bound_ind != 0)[0]
#
#    # ***determine z_base & z_top arrays
#    # z_top = -99.
#    # z_base = -99.
#    # z_cloud = -99
#
#    i_top = []
#    i_base = []
#
#    if len(i_cloud) != 0:
#        # z_cloud = z[i_cloud]
#        i_base = np.where(cloud_bound_ind == 1)[0]
#        i_top = np.where(cloud_bound_ind == 2)[0]
#        n_base = len(i_base)
#        n_top = len(i_top)
#        if n_top != n_base:
#            print('something wrong, number of bases NE number of cloud tops!')
#            return [], [], []
#        # end if
#    # end if
#    # z_top = z[i_top]
#    # z_base = z[i_base]
#
#    return i_top, i_base, i_cloud
#
#
## end def detect_liq_cloud
#
#
#def adiab2(i, T, P, RH, z):
#    """
#    Adiabtic liquid water content assuming pseudoadiabatic lapse rate
#    throughout the whole cloud layer. Thus the assumed temperature
#    profile is differnt from the measured one
#    Input:
#    i no of levels
#    T is in K
#    p is in Pa
#    z is in m
#    Output:
#    LWC in
#    translated to Python from adiab.pro by mx.
#    """
#
#    #   Set actual cloud base temperature to the measured one
#    #   Initialize Liquid Water Content (LWC)
#    #   Compute adiabatic LWC by integration from cloud base to level I
#
#    TCL = T[0]
#    LWC = 0.0
#
#    for j in range(1, i + 1):
#        deltaz = z[j] - z[j - 1]
#
#        #   Compute actual cloud temperature
#
#        #   1. Compute air density
#        #   2. Compute water vapor density of saturated air
#        #   3. Compute mixing ratio of saturated air
#        #   4. Compute pseudoadiabatic lapse rate
#        #   5. Compute actual cloud temperature
#
#        R = moist_rho_rh(P[j], T[j], 1.)
#        RWV = rh2a(1.0, T[j])
#        # R = moist_rho_rh(P[j], T[j], RH[j])
#        # RWV = rh2a(RH[j], T[j])
#        WS = RWV / (R - RWV)
#        DTPS = pseudoAdiabLapseRate(T[j], WS)
#        TCL = TCL + DTPS * (deltaz)
#
#        #   Compute adiabatic LWC
#
#        #   1. Compute air density
#        #   2. Compute water vapor density of saturated air
#        #   3. Compute mixing ratio of saturated air
#        #   4. Compute specific heat of vaporisation
#        #   5. Compute adiabatic LWC
#
#        R = moist_rho_rh(P[j], TCL, 1.)
#        RWV = rh2a(1.0, TCL)
#
#        R = moist_rho_rh(P[j], TCL, RH[j])
#        RWV = rh2a(RH[j], TCL)
#        WS = RWV / (R - RWV)
#        L = vaphet(TCL)
#
#        LWC = LWC + (R * Cp / L * ((Grav / Cp) - pseudoAdiabLapseRate(TCL, WS)) * (deltaz))
#    # end for
#
#    return LWC

#@jit(nopython=True, parallel=True)
#
#@jit(nopython=True, parallel=True)
#def pseudoAdiabLapseRate(T, Ws):
#    """
#    Pseudoadiabatic lapse rate
#    Input: T   [K]  thermodynamic temperature
#    Ws   [1]  mixing ratio of saturation
#    Output: PSEUDO [K/m] pseudoadiabatic lapse rate
#    Constants: Grav   [m/s2]     : constant of acceleration
#        CP  [J/(kg K)]    : specific heat cap. at const. press
#        Rair  [J/(kg K)]  : gas constant of dry air
#        Rvapor [J/(kg K)] : gas constant of water vapor
#    Source: Rogers and Yau, 1989: A Short Course in Cloud Physics
#    (III.Ed.), Pergamon Press, 293p. Page 32
#    translated to Python from pseudo1.pro by mx
#    """
#
#    # Compute specific humidity of vaporisation
#    L = vaphet(T)
#
#    # Compute pseudo-adiabatic temperature gradient
#    x = (Grav / Cp) * (1 + (L * Ws / Rair / T)) / (1 + (Ws * L ** 2 / Cp / Rvapor / T ** 2))
#
#    return x
#
#
#@jit(nopython=True, parallel=True)
#def vaphet(T):
#    """
#    Compute specific heat of vaporisation
#    Input  : T      [K]      thermodynamic temperature
#    Output : VAPHET [J/kg]   specific heat of vaporisation
#    Source: Liljequist, G.H. und K. Cehak, 1984: Allgemeine
#        Meteorologie (III.Ed.). Vieweg, 396p. Page 95
#    translated to Python from vaphet.pro by mx
#    """
#
#    x = (2500.8 - 2.372 * (T - 273.15)) * 1000.0
#
#    return x
