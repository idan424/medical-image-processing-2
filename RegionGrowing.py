import numpy as np


def get_neighbours(pt):
    return [(pt[0] - 1, pt[1] - 1), (pt[0] - 1, pt[1]), (pt[0] - 1, pt[1] + 1),
            (pt[0],     pt[1] - 1),                     (pt[0],     pt[1] + 1),
            (pt[0] + 1, pt[1] - 1), (pt[0] + 1, pt[1]), (pt[0] + 1, pt[1] + 1)]


class RegionGrow:
    def __init__(self, img, _seed):
        self.img = img
        self.seed = _seed
        self.obj_gray_level = self.img[self.seed]
        self.obj = np.zeros(img.shape)

        self._reg_grow(self.seed)

        self.obj_area = np.sum(self.obj)

    def is_homogenous(self, pt):
        return abs(self.obj_gray_level - self.img[pt]) < 10

    def _reg_grow(self, pt):
        homog_neighbors = [nb for nb in get_neighbours(pt) if self.is_homogenous(pt)]
        new_pts = [hnb for hnb in homog_neighbors if self.obj[hnb] == 0]

        for pnt in [pt, *homog_neighbors]:
            self.obj[pnt] = 1
        if not new_pts:
            return

        [self._reg_grow(n_pt) for n_pt in new_pts]



