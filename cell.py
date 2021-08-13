"""
Implementa la clase Cell, que modela a una celda E_k de un problema E.
"""

from functools import cached_property

import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

from segment import Segment, EmptySegment

np.random.seed(0)
segmentator = DBSCAN(eps=np.sqrt(2), min_samples=2)


class NoPointException(Exception):
    pass


class Cell:
    def __init__(self, cell, n, bp_id):
        self.bp_id = bp_id
        self.n_cell = n
        self.id = f"{bp_id}_{n}"
        self.as_array = cell
        self.points = np.argwhere(self.as_array.reshape(98, 98) != 0)

    def __repr__(self):
        id, n = self.id.split('_')
        return f"{self.__class__.__name__}(bp_id={bp_id}, n={n_cell})"

    def show(self):
        """
        Muestra un plot de la celda.
        """
        plt.imshow(self.as_array.reshape(98, 98), cmap=plt.cm.gray_r,
                   interpolation="nearest")
        plt.title(self.id)
        plt.xticks([])
        plt.yticks([])

    @cached_property
    def segmentation(self):
        try:
            return segmentator.fit(self.points)
        except ValueError:
            raise NoPointException("Celda no contiene puntos")

    @cached_property
    def components(self):
        """
        Diccionario que relaciona una etiqueta con su componente.

        Retorna:
        --------
        dict_clusters: Dict[int, List]
            Asocia la etiqueta E con todos los puntos etiquetados E por el clustering.
        """
        try:
            dict_clusters = {label: []
                             for label in np.unique(self.segmentation.labels_)}
            for i, xi in enumerate(self.points):
                label_i = self.segmentation.labels_[i]
                dict_clusters[label_i].append(xi)

            filter_dict_clusters = {}
            iteration = 0
            for set_points in dict_clusters.values():
                if len(set_points) > 3:
                    filter_dict_clusters[iteration] = set_points
                    iteration += 1
            return filter_dict_clusters
        except NoPointException:
            return {}

    def segmentation_array(self):
        """
        Dado un modelo de clustering, crea un arreglo donde cada punto se asocia
        a su etiqueta.

        Retorna:
        --------
        img_cluster: np.ndarray de shape (98, 98)
            Arreglo con nans en cada punto blanco de la imagen, y un número
            correspondiente a su cluster de segmentación.
        """
        img_cluster = np.nan + np.zeros([98, 98])
        for label, point_set in self.components.items():
            x, y = np.array(point_set).T
            img_cluster[x, y] = label
        return img_cluster

    def show_segmentation(self, cmap_="tab20"):
        """
        Muestra de la segmentación realizada.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.segmentation_array(),
                  cmap=cmap_, interpolation="nearest")
        ax.set_title("Segmentación por DBSCAN")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(self.id, size=20, x=0.5, y=1.07)

    def get_component(self, n_component):
        """
        Arreglo binario con solamente una componente.

        Parámetros:
        -----------
        n_component: int
            Número correspondiente al label de un cluster.
        Retorna:
        --------
        img_points: np.ndarray de shape (98, 98)
            Indicatriz del cluster.
        """
        component = self.components[n_component]
        img_points = np.zeros([98, 98], dtype=np.uint8)
        for x, y in component:
            img_points[x, y] = 255
        return img_points

    def segment_list(self):
        if not self.components:
            return [EmptySegment([], self.id, 0)]
        return [Segment(self.get_component(n), self.id, n)
                for n in self.components.keys()]
