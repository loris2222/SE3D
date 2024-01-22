from typing import Tuple, List

import cv2
import numpy as np

from wsol_3d_metrics.LocalizationEvaluator import LocalizationEvaluator, check_heatmap_validity

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


class BBoxEvaluator(LocalizationEvaluator):
    """
    Introduces MaxBoxAcc metric for localization evaluation over 2D heatmaps.
    """

    def __init__(self, **kwargs):
        super(BBoxEvaluator, self).__init__(**kwargs)
        self.cnt = 0
        self.num_correct = {
            iou_threshold: np.zeros(len(self.cam_threshold_list))
            for iou_threshold in self.iou_threshold_list
        }

    def compute_bboxes_from_heatmap(self, heatmap: np.ndarray, heatmap_threshold_list: List,
                                    multi_contour_eval: bool = False, is_mask: bool = False) -> Tuple[
        List[np.ndarray], List[int]]:
        """
        Args:
            heatmap: numpy.ndarray(dtype=float32, size=(H, W)) between 0 and 1
            heatmap_threshold_list: iterable
            multi_contour_eval: flag for multi-contour evaluation
            is_mask: flag for mask evaluation

        Returns:
            estimated_bboxes_at_each_thr: list of estimated boxes (list of np.array)
                at each cam threshold
            number_of_bbox_list: list of the number of boxes at each cam threshold
        """
        check_heatmap_validity(heatmap, n_dims=2)
        height, width = heatmap.shape
        heatmap_image = np.expand_dims((heatmap * 255).astype(np.uint8), 2)

        def heatmap_to_bbox(heatmap_image: np.ndarray, threshold: float, is_mask: bool = False) -> Tuple[
            np.ndarray, int]:
            if not is_mask:
                _, thr_gray_heatmap = cv2.threshold(
                    src=heatmap_image,
                    thresh=int(threshold * np.max(heatmap_image)),
                    maxval=255,
                    type=cv2.THRESH_BINARY)
            contours = cv2.findContours(
                image=heatmap_image if is_mask else thr_gray_heatmap,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )[_CONTOUR_INDEX]

            if len(contours) == 0:
                return np.asarray([[0, 0, 0, 0]]), 1

            if not multi_contour_eval:
                contours = [max(contours, key=cv2.contourArea)]

            estimated_bboxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x0, y0, x1, y1 = x, y, x + w, y + h
                x1 = min(x1, width - 1)
                y1 = min(y1, height - 1)
                estimated_bboxes.append([x0, y0, x1, y1])

            return np.asarray(estimated_bboxes), len(contours)

        estimated_bboxes_at_each_thr = []
        number_of_bbox_list = []
        for threshold in heatmap_threshold_list:
            boxes, number_of_box = heatmap_to_bbox(
                heatmap_image, threshold, is_mask=is_mask)
            estimated_bboxes_at_each_thr.append(boxes)
            number_of_bbox_list.append(number_of_box)
            if is_mask:
                break

        return estimated_bboxes_at_each_thr, number_of_bbox_list

    def check_bbox_convention(self, bboxes: np.ndarray, convention: str) -> None:
        """
        Args:
            bboxes: numpy.ndarray(dtype=np.uint8 or float, shape=(num_bboxes, 4))
            convention: string. One of ['x0y0x1y1', 'xywh'].
        Raises:
            RuntimeError if box does not meet the convention.
        """
        if (bboxes < 0).any():
            raise RuntimeError("Box coordinates must be non-negative.")

        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, 0)
        elif len(bboxes.shape) != 2:
            raise RuntimeError("Box array must have dimension (4) or "
                               "(num_bboxes, 4).")

        if bboxes.shape[1] != 4:
            raise RuntimeError("Box array must have dimension (4) or "
                               "(num_bboxes, 4).")

        if convention == 'x0y0x1y1':
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
        elif convention == 'xywh':
            widths = bboxes[:, 2]
            heights = bboxes[:, 3]
        else:
            raise ValueError(f"Unknown convention {convention}.")

        if (widths < 0).any() or (heights < 0).any():
            raise RuntimeError(
                f"Bounding boxes do not follow the {convention} convention.")

    def calculate_multiple_iou(self, bbox_a: np.ndarray, bbox_b: np.ndarray) -> np.ndarray:
        """
        Args:
            bbox_a: numpy.ndarray(dtype=np.uint8, shape=(num_a, 4))
                x0y0x1y1 convention.
            bbox_b: numpy.ndarray(dtype=np.uint8, shape=(num_b, 4))
                x0y0x1y1 convention.
        Returns:
            ious: numpy.ndarray(dtype=np.uint8, shape(num_a, num_b))
        """
        num_a = bbox_a.shape[0]
        num_b = bbox_b.shape[0]

        self.check_bbox_convention(bbox_a, 'x0y0x1y1')
        self.check_bbox_convention(bbox_b, 'x0y0x1y1')

        # num_a x 4 -> num_a x num_b x 4
        bbox_a = np.tile(bbox_a, num_b)
        bbox_a = np.expand_dims(bbox_a, axis=1).reshape((num_a, num_b, -1))

        # num_b x 4 -> num_b x num_a x 4
        bbox_b = np.tile(bbox_b, num_a)
        bbox_b = np.expand_dims(bbox_b, axis=1).reshape((num_b, num_a, -1))

        # num_b x num_a x 4 -> num_a x num_b x 4
        bbox_b = np.transpose(bbox_b, (1, 0, 2))

        # num_a x num_b
        min_x = np.maximum(bbox_a[:, :, 0], bbox_b[:, :, 0])
        min_y = np.maximum(bbox_a[:, :, 1], bbox_b[:, :, 1])
        max_x = np.minimum(bbox_a[:, :, 2], bbox_b[:, :, 2])
        max_y = np.minimum(bbox_a[:, :, 3], bbox_b[:, :, 3])

        # num_a x num_b
        area_intersect = (np.maximum(0, max_x - min_x + 1)
                          * np.maximum(0, max_y - min_y + 1))
        area_a = ((bbox_a[:, :, 2] - bbox_a[:, :, 0] + 1) *
                  (bbox_a[:, :, 3] - bbox_a[:, :, 1] + 1))
        area_b = ((bbox_b[:, :, 2] - bbox_b[:, :, 0] + 1) *
                  (bbox_b[:, :, 3] - bbox_b[:, :, 1] + 1))

        denominator = area_a + area_b - area_intersect
        degenerate_indices = np.where(denominator <= 0)
        denominator[degenerate_indices] = 1

        ious = area_intersect / denominator
        ious[degenerate_indices] = 0
        return ious

    def accumulate(self, heatmap: np.ndarray, mask: np.ndarray) -> None:
        """
        From a heatmap, a box is inferred (compute_bboxes_from_heatmaps).
        The box is compared against GT boxes. Count a heatmap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            heatmap: numpy.ndarray(size=(H, W), dtype=float)
            mask: numpy.ndarray(size=(H, W), dtype=np.uint8).
        """
        boxes_at_thresholds, number_of_bbox_list = self.compute_bboxes_from_heatmap(
            heatmap=heatmap,
            heatmap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)
        gt_bboxes, _ = self.compute_bboxes_from_heatmap(
            heatmap=mask,
            is_mask=True,
            heatmap_threshold_list=[None],
            multi_contour_eval=self.multi_contour_eval)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)
        gt_bboxes = np.concatenate(gt_bboxes, axis=0)

        multiple_iou = self.calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(gt_bboxes)
        )

        idx = 0
        sliced_multiple_iou = []
        for nr_box in number_of_bbox_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box

        for threshold in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou)
                         >= (threshold / 100))[0]
            self.num_correct[threshold][correct_threshold_indices] += 1
        self.cnt += 1

    def compute(self) -> List[float]:
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best heatmap threshold is taken
               for the final performance.
        """
        max_bbox_acc = []

        for threshold in self.iou_threshold_list:
            localization_accuracies = self.num_correct[threshold] * 100. / \
                                      float(self.cnt)
            max_bbox_acc.append(localization_accuracies.max())

        return max_bbox_acc
