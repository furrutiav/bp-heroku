"""
Implementa la clase BongardProblem, que modela al problema entero.
"""

from functools import cached_property

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import cv2

from cell import Cell
from segment import COLUMNS


def img_to_cells(img):
    w, h = 100 - 2, 100 - 2
    x0, y0 = 8, 6
    gap = 10
    n = 0
    dict_img_cells = {}
    for k in range(2):
        for i in range(3):
            for j in range(2):
                x, y = x0 + 293 * k + (w + gap) * j, y0 + (h + gap) * i
                cropped_img = img[y:y + h, x:x + w]
                dict_img_cells[n] = cropped_img
                n += 1
    return dict_img_cells.values()


class BongardProblem:
    """
    Clase que tiene las 12 celdas de un problema de Bongard.

    Referencia:
    -----------
    https://www.foundalis.com/res/bps/bpidx.htm
    """

    def __init__(self, filename, id=None):
        self.filename = filename
        if id is None:
            folder = filename.split("/")[-1]
            self.id = folder.split(".")[0]
        else:
            self.id = id
        im_gray = ~cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_bw = cv2.threshold(im_gray, 175, 255, cv2.THRESH_BINARY)[1]
        cells = img_to_cells(im_bw)
        self.cells = [Cell(cell, n, self.id) for n, cell in enumerate(cells)]
        self.problem_class = None  # Todavía no se calcula

    def show(self):
        fig, ax = plt.subplots(3, 4, figsize=(3 * 4, 3 * 3))
        for n, cell in enumerate(self.cells):
            i, j = (n % 6) // 2, 2 * (n // 6) + (n % 2)
            ax[i, j].imshow(cell.as_array.reshape(98, 98),
                            interpolation="nearest", cmap="binary")
            ax[i, j].set_title(n)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            plt.suptitle(self.id, size=21)

    def show_segmentations(self, cmap_):
        """
        Muestra todas las segmentaciones creadas por DBSCAN.

        Parámetros:
        -----------
        cmap_: str
            Corresponde al mapa de colores para el plot.
        """
        fig, ax = plt.subplots(3, 4, figsize=(3 * 4, 3 * 3))
        for n, cell in enumerate(self.cells):
            i, j = (n % 6) // 2, 2 * (n // 6) + (n % 2)
            dbscan_img_cluster = cell.segmentation
            ax[i, j].imshow(dbscan_img_cluster, cmap=cmap_,
                            interpolation="nearest")
            ax[i, j].set_title(n)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            plt.suptitle(f"DBSCAN: {self.id}", size=21)

    @cached_property
    def segment_rep_table(self):
        segment_lists = [cell.segment_list() for cell in self.cells]
        flattened = [val for sublist in segment_lists for val in sublist]
        dics = [segment.rep_dict() for segment in flattened]
        dict = {key: [dic[key] for dic in dics] for key in dics[0].keys()}
        as_pd = pd.DataFrame(dict, index=dict["ID"])
        segment_indices = [int(ix.split("_")[1]) for ix in dict["ID"]]
        if segment_indices != list(range(12)):
            self.problem_class = 1
        else:
            self.problem_class = 0
            simpler_index = ['_'.join(ix.split("_")[:-1]) for ix in dict["ID"]]
            as_pd["ID_"] = simpler_index
            as_pd.set_index("ID_", inplace=True)
        return as_pd.drop(["ID"], axis=1).replace(np.nan, 0)

    def cluster_segments(self, size="default", linkage="ward"):
        HC = pd.DataFrame(index=self.segment_rep_table.index)
        if size == "default":
            size = self.segment_rep_table.shape[0] // 4
        for k in range(1, size + 1):
            clustering = AgglomerativeClustering(
                n_clusters=2 * k, linkage=linkage)
            clustering.fit(self.segment_rep_table)
            HC[k] = clustering.labels_
        return HC

    def cluster_rep(self, **kwargs):
        HC = self.cluster_segments(**kwargs)
        indexes = [int(ix.split("_")[1]) for ix in HC.index]
        size_hc = HC.shape[1]
        columns_ = [f"{cls}_{n}" for cls in range(
            1, size_hc + 1) for n in range(2 * cls)]
        rep_dic = [[] for _ in range(12)]
        for n in range(12):
            indexs = [ix for num, ix in enumerate(
                HC.index) if indexes[num] == n]
            len_indexs = len(indexs)
            cols = {c: 0 for c in columns_}
            for ix in indexs:
                rep = HC.loc[ix]
                for cls in range(1, size_hc + 1):
                    val = rep[cls]
                    cols[f"{cls}_{val}"] += 1 / len_indexs
            max_1_iter = max(cols["1_0"], cols["1_1"])
            max_2_iter = max(cols["2_0"], cols["2_1"],
                             cols["2_2"], cols["2_3"])
            max_3_iter = max(cols["3_0"], cols["3_1"], cols["3_2"],
                             cols["3_3"], cols["3_4"], cols["3_5"])
            rep_dic[n] = [len_indexs, max_1_iter,
                          max_2_iter, max_3_iter] + list(cols.values())
        final_cols = ["num_comp", "max_1_iter",
                      "max_2_iter", "max_3_iter"] + columns_
        return pd.DataFrame(rep_dic, columns=final_cols)
