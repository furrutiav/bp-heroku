"""
Implementa las propiedades de un segmento.
"""

from functools import cached_property

import cv2
import matplotlib.pyplot as plt
import numpy as np


COLUMNS = ["area contorno", "area envoltura convexa", "perimetro contorno",
           "perimetro envoltura convexa", "masa", "hull size", "size", "ratio area",
           "ratio perimetro", "elipticidad", "inclinación", "orientación", "excentricidad",
           "centroide x", "centroide y", "circularidad contorno", "circularidad hull",
           "centralidad x", "centralidad y", "skewness x", "skewness y", "H1", "H2",
           "H3", "H4"]

def handle_zero_division(default):
    def wrapper(f):
        def inner(*a):
            try:
                return f(*a)
            except ZeroDivisionError:
                return default
        return inner
    return wrapper


class BaseSegment:
    def __init__(self, array, id, n):
        self.id = f"{id}_{n}"
        self.array = array

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(self.array, interpolation="nearest")
        ax.set_title(f"Segmento {self.id}")
        ax.set_xticks([])
        ax.set_yticks([])

    def rep_vect(self):
        pass

    def rep_dict(self):
        dict = {col: val for col, val in zip(COLUMNS, self.rep_vect())}
        dict["ID"] = self.id
        return dict


class EmptySegment(BaseSegment):
    def rep_vect(self):
        return np.zeros(len(COLUMNS))


class Segment(BaseSegment):
    def __init__(self, array, id, n):
        BaseSegment.__init__(self, array, id, n)
        self.array = array
        self.mass = array[array >= 127].size
        contours, hierarchy = cv2.findContours(self.array, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        self.contour_area = cv2.contourArea(contour)
        moments = cv2.moments(self.array)
        self.moments = moments
        hu_moments = np.array(cv2.HuMoments(moments)).flatten()[0:4]
        self.tr_moments = np.sign(-hu_moments) * np.log(abs(hu_moments))
        self.hull = cv2.convexHull(contour)
        self.centrality = moments["nu12"], moments["nu21"]
        self.skewness = moments["nu30"], moments["nu03"]
        self.hull_area = cv2.contourArea(self.hull)
        self.perimetro_hull = cv2.arcLength(self.hull, True)
        self.perimetro = cv2.arcLength(contour, True)
        self.inclination = (moments["nu20"]) / (10**(-12) + moments["nu02"])

    @property
    @handle_zero_division([0, 0])
    def centroid(self):
        moments = self.moments
        cx, cy = moments["m10"] / \
            moments["m00"], moments["m01"] / moments["m00"]
        return [cx, cy]

    @property
    @handle_zero_division([0, 0])
    def circularity(self):
        return [4 * np.pi * self.contour_area / (self.perimetro**2),
                4 * np.pi * self.hull_area / (self.perimetro_hull**2)]

    @property
    @handle_zero_division(0)
    def ellipticity(self):
        H1, H2 = self.tr_moments[0:2]
        return (1 / (2 * (np.pi**2))) * (1 / (H1 * np.sqrt(4 * H2 + (1 / (np.pi**2))) - 2 * H2))

    @property
    @handle_zero_division(0)
    def orientation(self):
        moments = self.moments
        r = 2 * moments["mu11"] / (moments["mu20"] - moments["mu02"])
        return (1 / 2) * np.arctan(r)

    @property
    @handle_zero_division(0)
    def eccentricity(self):
        moments = self.moments
        m00 = moments["m00"]
        mu20_ = moments["mu20"] / m00
        mu02_ = moments["mu02"] / m00
        mu11_ = moments["mu11"] / m00
        r0 = (mu20_ + mu02_) / 2
        r1 = 4 * (mu11_**2) + (mu20_ - mu02_)**2
        l1 = r0 + np.sqrt(r1) / 2
        l2 = r0 - np.sqrt(r1) / 2
        return np.sqrt(1 - l2 / l1)

    @property
    @handle_zero_division(0)
    def ratio_area(self):
        return self.contour_area / self.hull_area

    @property
    @handle_zero_division(0)
    def ratio_perimetro(self):
        return self.perimetro / self.perimetro_hull

    def rep_vect(self):
        escala_masa = np.array([self.contour_area, self.hull_area, self.perimetro,
                                self.perimetro_hull]) / self.mass
        otros = [self.mass, self.hull.size, self.contour_area,
                 self.ratio_area, self.ratio_perimetro,
                 self.ellipticity, self.inclination, self.orientation, self.eccentricity]
        return np.concatenate((escala_masa, otros, self.centroid, self.circularity,
                               self.centrality, self.skewness, self.tr_moments))
