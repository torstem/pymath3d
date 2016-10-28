# coding=utf-8

"""
Module for Plane class.
"""

__author__ = "Morten Lind"
__copyright__ = "Morten Lind 2013-2016"
__credits__ = ["Morten Lind"]
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten@lind.dyndns.dk"
__status__ = "Development"


import math3d as m3d
import numpy as np

from .. import utils


class Plane(object):
    def __init__(self, **kwargs):
        """Create a plane representation by one of the following named
        arguments:

        * 'plane_vector': A normalized plane vector. The normal will
          be pointing away from the origo. If kw-argument
          'origo_inside' is given, this will determine the direction
          of the plane normal; otherwise origo will be set inside.

        * 'pn_pair': An ordered sequence for creating a reference
        point and a normal vector. The normal

        * 'points': A set of at least three points for fitting a
        plane.

        * 'coeffs': Four coefficients (a,b,c,d) for the plane equation
          ax+by+cz+d=0.

        The internal representation is point and normal. If given as a
        pn_pair, A boolean, 'origo_inside', is held to decide the
        direction of the normal vector, such that the origo of the
        defining coordinate system is on the inside when true.
        """

        self._origo_inside = kwargs.get('origo_inside', True)
        if 'plane_vector' in kwargs:
            pv = m3d.Vector(kwargs['plane_vector'])
            (self._p, self._n) = self.pv_to_pn(pv)
        elif 'pn_pair' in kwargs:
            (self._p, self._n) = [m3d.Vector(e) for e in kwargs['pn_pair']]
            # # Override a given origo inside.
            self._origo_inside = (self._p * self._n) > 0
            # # Make point a 'minimal' point on the plane, i.e. the
            # # projection of origo in the plane.
            self._p = (self._p * self._n) * self._n
        elif 'points' in kwargs:
            self.fit_plane(kwargs['points'])
        elif 'coeffs' in kwargs:
            self.coeffs = kwargs['coeffs']
        else:
            raise Exception(
                'Plane.__init__ : Must have either of constructor ' +
                'kw-arguments: "plane_vector", "pn_pair", or ' +
                '"points". Neither given!')

    def copy(self):
        return Plane(pn_pair=(self._p, self._n))

    def __repr__(self):
        return '<Plane: [{:.5f}, {:.5f}, {:.5f}]>'.format(
            *tuple(self.plane_vector.array))

    def __rmul__(self, transf):
        """Support transformation of this plane to another coordinate
        system by multiplication of an m3d.Transform from left."""
        if type(transf) != m3d.Transform:
            return NotImplemented
        tnormal = transf.orient * self._n
        tpoint = transf * self._p
        return Plane(pn_pair=(tpoint, tnormal))

    def dist(self, p):
        """Signed distance to a point, measured positive along the
        normal vector direction."""
        return (m3d.Vector(p) - self._p) * self._n

    def get_plane_vector(self):
        return self.pn_to_pv(self._p, self._n)

    def set_plane_vector(self, pv):
        (self._p, self._n) = self.pv_to_pn(pv)

    plane_vector = property(get_plane_vector, set_plane_vector)

    @property
    def point_normal(self):
        return (self._p, self._n)

    @property
    def point(self):
        return self._p

    @property
    def normal(self):
        return self._n

    def get_coeffs(self):
        """Return the four coefficients of the plane."""
        return list(self._n) + [self.dist([0, 0, 0])]

    def set_coeffs(self, coeffs):
        """Set the plane to the one given by the four coefficients."""
        self.plane_vector = m3d.Vector(coeffs[:3]) / -coeffs[3]
        # if not len(coeffs) == 4:
        #     raise Exception('Plane needs four coefficients!')
        # self._n = m3d.Vector(coeffs[:3]).normalized
        # self._p = -coeffs[3] * self._n

    coeffs = property(get_coeffs, set_coeffs)

    def fit_plane(self, points):
        """Compute the plane vector from a set of points. 'points'
        must be an array of row position vectors, such that
        points[i] is a position vector."""
        points = np.array(points)
        centre = np.sum(points, axis=0)/len(points)
        eigen = np.linalg.eig(np.cov(points.T))
        min_ev_i = np.where(eigen[0] == min(eigen[0]))[0][0]
        normal = eigen[1].T[min_ev_i]
        (self._p, self._n) = (m3d.Vector(centre), m3d.Vector(normal))

    @classmethod
    def pn_to_pv(cls, point, normal):
        """Compute the plane vector of a plane represented by a point
        and normal."""
        if not isinstance(point, m3d.Vector):
            point = m3d.Vector(point)
        if not isinstance(normal, m3d.Vector):
            normal = m3d.Vector(normal)
        # // Origo projection on plane
        p0 = (point * normal) * normal
        # // Square of offset from origo
        d2 = p0.length_squared
        # // return the plane vector
        return p0 / (d2)

    def pv_to_pn(self, pv):
        """Calculate a point-normal representation of the plane
        described by the given plane vector."""
        if isinstance(pv, m3d.Vector):
            pv = m3d.Vector(pv)
        d = pv.length
        n = pv / pv.length
        p = n / d
        if not self._origo_inside:
            n = -n
        return (p, n)

    def projection(self, point):
        """Return the projection of the 'point' on the plane."""
        if isinstance(point, m3d.Vector):
            point = m3d.Vector(point)
        return point - self._n * (point - self._p) * self._n

    def line_intersection(self, other):
        """Compute the intersection with the given line."""
        if type(other) != m3d.geometry.Line:
            raise Exception(
                'Method only implemented for math3d.geometry.Line object')
        (lp, ld) = other._p, other._d
        ndd = self._n * ld
        if np.abs(ndd) < utils.eps:
            return None
        else:
            dist = (self._p - lp) * self._n / ndd
            return dist * ld + lp

    def plane_intersection(self, other):
        """Find the line of intersection with 'other' plane. Method found in
        http://paulbourke.net/geometry/pointlineplane/
        """
        if not isinstance(other, Plane):
            raise Exception(
                'Method only implemented for math3d.geometry.Plane object')
        ld = self._n.cross(other._n)
        if ld.length < utils.eps:
            return None
        ld.normalize()
        ndot = self._n * other._n
        det = 1 - ndot ** 2
        ds = -self.coeffs[3]
        do = -other.coeffs[3]
        cs = (ds - do * ndot) / det
        co = (do - ds * ndot) / det
        lp = cs * self._n + co * other._n
        return m3d.geometry.Line(point_direction=(lp, ld))

    def intersection(self, other):
        """Polymorphic intersection method."""
        if isinstance(other, Plane):
            return self.plane_intersection(other)
        elif isinstance(other, m3d.geometry.Line):
            return self.line_intersection(other)
        else:
            raise NotImplementedError('Can not compute intersection with ' +
                                      'object of type {}'.format(type(other)))


def _test():
    # Test creation on points
    pln = Plane(points=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert (np.abs(pln.normal * m3d.Vector(1, 1, 1).normalized) - 1
            < utils.eps)
    pln0 = Plane(plane_vector=(1, 0, 0))
    pln1 = Plane(plane_vector=(0, 1, 0))
    # Test for intersection between planes
    line = pln0.intersection(pln1)
    assert line.point.x == 1 and line.point.y == 1
    assert np.abs(line.direction * m3d.Vector.ez) == 1
    # Test for intersection with unsupported object
    try:
        pln0.intersection(m3d.Vector.ex)
    except NotImplementedError as nie:
        print('Caught expected exception from intersection of plane ' +
              'with vector. "{}"'.format(str(nie)))
