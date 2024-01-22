import numpy as np

from wsol_3d_metrics.LocalizationEvaluator import LocalizationEvaluator, check_heatmap_validity


class MaskEvaluator(LocalizationEvaluator):
    """
    Introduces VxAP metric for localization evaluation over heatmaps.
    """

    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    def accumulate(self, heatmap: np.ndarray, mask: np.ndarray) -> None:
        """
        Score histograms over the heatmap values at GT positive and negative
        pixels are computed.

        Args:
            heatmap: numpy.ndarray(size=(H, W, D), dtype=float)
            mask: numpy.ndarray(size=(H, W, D), dtype=float).
        """
        check_heatmap_validity(heatmap, n_dims=3)

        gt_true_scores = heatmap[mask == 1]
        gt_false_scores = heatmap[mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(float)

    def compute(self) -> float:
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        with np.errstate(divide='ignore', invalid='ignore'):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        return auc
