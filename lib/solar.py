#!/usr/bin/env python

"""
Module for calculating the location of solar bodies.

Algorithms developed by Paul Schlyter.
http://www.stjarnhimlen.se/comp/ppcomp.html
"""

import datetime
import numpy as np
from numpy import pi

def rectangular_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    inclination = np.arctan2( z, np.sqrt( x ** 2 + y ** 2) )
    azimuth = np.arctan2( y, x ) % (2 * pi)
    return (r, inclination, azimuth)

def spherical_to_rectangular(r, inclination, azimuth):
    x = r * np.cos(inclination) * np.cos(azimuth)
    y = r * np.cos(inclination) * np.sin(azimuth)
    z = r * np.sin(inclination)
    return x, y, z

class Body(object):
    """
    Abstract class representing a solar body.
    """
    # The following methods should be implemented by concrete classes:
    # i : inclination to the ecliptic
    # a : semi-major axis, or mean distance from Sun
    # e : eccentricity (0=circle, 0-1=ellipse, 1=parabola)
    # N : longitude of the ascending node
    # w : argument of perihelion
    # M : mean anomaly (0 at perihelion; increases uniformly with time)
    # E : Eccentric anomaly

    def __init__(self, date=None):
        super(Body, self).__init__()
        self.date = date
    
    @property
    def date(self):
        return self._date
    
    @date.setter
    def date(self, date):
        if date is not None:
            self._d = 367 * date.year \
                      - 7 * ( date.year + ( date.month + 9 ) / 12 ) / 4 \
                      + 275 * date.month / 9 + date.day - 730530 \
                      + date.hour / 24. + date.minute / 60. / 24.
        else:
            self._d = None
        self._date = date
    
    def oblecl(self):
        # obliquity of the ecliptic
        d = self._d
        return np.deg2rad( (23.4393 - 3.563E-7 * d) % 360)
    
    def w1(self):
        # longitude of perihelion
        return self.N() + self.w()

    def L(self):
        # mean longitude
        return (self.M() + self.w()) % (2 * pi)
    
    def q(self):
        # perihelion distance
        return self.a() * (1 - self.e())
    
    def Q(self):
        # aphelion distance
        return self.a() * (1 + self.e())
    
    def P(self):
        # orbital period
        return self.a() ** 1.5
    
    def E(self):
        # Eccentric anomaly
        return self.M() + self.e() * np.sin( self.M() ) * ( 1.0 + self.e() * np.cos( self.M() ) )
    
    def xv(self):
        return self.a() * ( np.cos( self.E() ) - self.e() )
    
    def yv(self):
        return self.a() * ( np.sqrt( 1.0 - self.e() ** 2 ) * np.sin( self.E() ) )
    
    def v(self):
        return np.arctan2( self.yv(), self.xv() ) % (2 * pi)
    
    def r(self):
        return np.sqrt( self.xv() ** 2 + self.yv() ** 2 )
    
    def rectangular_ecliptic_coordinates(self):
        # (xh, yh, zh) = ecliptical coordinates
        r = self.r()
        N = self.N()
        v = self.v()
        w = self.w()
        i = self.i()
        
        xh = r * ( np.cos(N) * np.cos(v + w) - np.sin(N) * np.sin(v + w) * np.cos(i) )
        yh = r * ( np.sin(N) * np.cos(v + w) + np.cos(N) * np.sin(v + w) * np.cos(i) )
        zh = r * np.sin(v + w) * np.sin(i)
        return (xh, yh, zh)
    
    def spherical_ecliptic_coordinates(self):
        x, y, z = self.rectangular_ecliptic_coordinates()
        latitude = np.arctan2( z, np.sqrt( x ** 2 + y ** 2) )
        longitude = np.arctan2( y, x ) % (2 * pi)
        r = self.r()
        return (r, latitude, longitude)
    
    def rectangular_equatorial_coordinates(self):
        oblecl = self.oblecl()
        xh, yh, zh = self.rectangular_ecliptic_coordinates()
        xe = xh
        ye = yh * np.cos( oblecl ) - zh * np.sin( oblecl )
        ze = yh * np.sin( oblecl ) + zh * np.cos( oblecl )
        return (xe, ye, ze)
    
    def spherical_equatorial_coordinates(self):
        x, y, z = self.rectangular_equatorial_coordinates()
        r = np.sqrt( x**2 + y**2 + z**2 )
        declination = np.arctan2( z, np.sqrt( x ** 2 + y ** 2) )
        right_ascension = np.arctan2( y, x ) % (2 * pi)
        return r, declination, right_ascension
    
    def spherical_geographic_coordinates(self, gmst=None):
        if gmst is None:
            days = (self.date - datetime.datetime(2000, 1, 1, 12)).total_seconds() / (24. * 3600.)
            gmst = (18.697374558 + 24.06570982441908 * days) % 24
        r, declination, right_ascension = self.spherical_equatorial_coordinates()
        latitude = declination
        gmst_rad = gmst * 2 * pi / 24.
        longitude = (right_ascension - gmst_rad) % (2 * pi)
        if longitude > pi:
            longitude -= 2 * pi
        return r, latitude, longitude

    def rectangular_geographic_coordinates(self):
        r, latitude, longitude = self.spherical_geographic_coordinates()
        return spherical_to_rectangular(r, latitude, longitude)
    
class Sun(Body):
    def __init__(self, date=None):
        super(Sun, self).__init__(date)
        
    def i(self):
        return 0.0
    
    def a(self):
        return 1.0
        
    def N(self):
        return 0.0
    
    def w(self):
        d = self._d
        return np.deg2rad( (282.94 + 4.70935e-5 * d) % 360 )
    
    def e(self):
        d = self._d
        return 0.016709 - 1.151e-9 * d
    
    def M(self):
        d = self._d
        return np.deg2rad( (356.0470 + 0.9856002585 * d) % 360 )
    
class Moon(Body):
    def __init__(self, date=None):
        super(Moon, self).__init__(date)
        # TODO: more accurate E
    
    def i(self):
        # i = inclination to the ecliptic (plane of the Earth's orbit)
        return np.deg2rad(5.1454)
    
    def a(self):
        # a = semi-major axis, or mean distance from Sun
        return 60.2666
    
    def e(self):
        # e = eccentricity (0=circle, 0-1=ellipse, 1=parabola)
        return 0.054900
        
    def N(self):
        d = self._d
        # N = longitude of the ascending node
        return np.deg2rad( (125.1228 - 0.0529538083 * d) % 360 )
    
    def w(self):
        # w = argument of perihelion
        d = self._d
        return np.deg2rad( (318.0634 + 0.1643573223 * d) % 360 )
    
    def M(self):
        # M = mean anomaly (0 at perihelion; increases uniformly with time)
        d = self._d
        return np.deg2rad( (115.3654 + 13.0649929509 * d) % 360 )
