import numpy as np


def check_heatmap_validity(heatmap: np.ndarray, n_dims: int = 2) -> None:
    if not isinstance(heatmap, np.ndarray):
        raise TypeError(
            f"Heatmap must be a numpy array; it is {type(heatmap)}.")
    if heatmap.dtype != np.float32:
        raise TypeError(
            f"Heatmap must be of float type; it is of {heatmap.dtype} type.")
    if len(heatmap.shape) != n_dims:
        raise ValueError(
            f"Heatmap must be a {n_dims}D array; it is {len(heatmap.shape)}D.")
    if np.isnan(heatmap).any():
        raise ValueError("Heatmap must not contain nans.")
    if (heatmap > 1).any() or (heatmap < 0).any():
        raise ValueError("Heatmap must be in range [0, 1]."
                         f"heatmap.min()={heatmap.min()}, heatmap.max()={heatmap.max()}.")


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over heatmaps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    heatmap computation). At initialization, __init__ registers data
    containers for evaluation. At each iteration,
    each heatmap is passed to the accumulate() method along with its mask.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, cam_curve_interval=0.01,
                 iou_threshold_list=(30, 50, 70), multi_contour_eval=False):
        self.cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
        self.iou_threshold_list = iou_threshold_list
        self.multi_contour_eval = multi_contour_eval

    def accumulate(self, heatmap, mask):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
