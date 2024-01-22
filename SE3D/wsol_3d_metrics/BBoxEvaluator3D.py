from typing import List, Tuple

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from wsol_3d_metrics.LocalizationEvaluator import LocalizationEvaluator, check_heatmap_validity


class BBoxEvaluator3D(LocalizationEvaluator):
    """
    Introduces Max3DBoxAcc metric for localization evaluation over 3D heatmaps.
    """

    def __init__(self, **kwargs):
        super(BBoxEvaluator3D, self).__init__(**kwargs)
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
            heatmap: numpy.ndarray(dtype=float32, size=(H, W, D)) between 0 and 1
            heatmap_threshold_list: iterable
            multi_contour_eval: flag for multi-contour evaluation
            is_mask: flag for mask evaluation

        Returns:
            estimated_bboxes_at_each_thr: list of estimated boxes (list of np.ndarray)
                at each cam threshold
            number_of_bbox_list: list of the number of boxes at each cam threshold
        """
        check_heatmap_validity(heatmap, n_dims=3)
        height, width, depth = heatmap.shape
        heatmap_image = (heatmap * 255).astype(np.uint8)

        def heatmap_to_bbox(heatmap_img: np.ndarray, threshold: float, is_mask: bool = False) -> Tuple[
            np.ndarray, int]:
            """
            Args:
                heatmap_img: numpy.ndarray(dtype=uint8, size=(H, W, D))
                threshold: float
                is_mask: flag for mask evaluation
                
            Returns:
                estimated_bboxes: np.ndarray(dtype=int32, size=(N, 6))
                    N: the number of estimated boxes
                    6: (x0, y0, z0, x1, y1, z1)
                number_of_box: int
            """
            if not is_mask:
                # binarize the heatmap over a threshold
                heatmap_img = (heatmap_img > (
                        threshold * np.max(heatmap_img)))

            labeled, nr_objects = label(
                heatmap_img, structure=generate_binary_structure(3, 2))

            if nr_objects == 0:
                return np.asarray([[0, 0, 0, 0, 0, 0]]), 1

            estimated_bboxes = []
            for i in range(1, nr_objects + 1):
                # get the coordinates of bounding boxes x0y0z0x1y1z1
                coordinates = np.where(labeled == i)
                x0 = np.min(coordinates[1])
                y0 = np.min(coordinates[0])
                z0 = np.min(coordinates[2])
                x1 = min(np.max(coordinates[1]), width - 1)
                y1 = min(np.max(coordinates[0]), height - 1)
                z1 = min(np.max(coordinates[2]), depth - 1)
                estimated_bboxes.append([x0, y0, z0, x1, y1, z1])

            if not multi_contour_eval:
                # find the largest connected component
                estimated_bboxes = np.asarray(estimated_bboxes)
                box_areas = (estimated_bboxes[:, 3] - estimated_bboxes[:, 0]) * \
                            (estimated_bboxes[:, 4] - estimated_bboxes[:, 1]) * \
                            (estimated_bboxes[:, 5] - estimated_bboxes[:, 2])
                largest_bbox_index = np.argmax(box_areas)
                estimated_bboxes = estimated_bboxes[largest_bbox_index]
                estimated_bboxes = np.expand_dims(estimated_bboxes, axis=0)
                nr_objects = 1

            return np.asarray(estimated_bboxes), nr_objects

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
            bboxes: numpy.ndarray(dtype=np.uint8 or float, shape=(num_bboxes, 6))
            convention: string. One of ['x0y0z0x1y1z1', 'xyzwhd'].
        Raises:
            RuntimeError if box does not meet the convention.
        """
        if (bboxes < 0).any():
            raise RuntimeError("Box coordinates must be non-negative.")

        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, 0)
        elif len(bboxes.shape) != 2:
            raise RuntimeError("Box array must have dimension (6) or "
                               "(num_bboxes, 6).")

        if bboxes.shape[1] != 6:
            raise RuntimeError("Box array must have dimension (6) or "
                               "(num_bboxes, 6).")

        if convention == 'x0y0z0x1y1z1':
            widths = bboxes[:, 3] - bboxes[:, 0]
            heights = bboxes[:, 4] - bboxes[:, 1]
            depths = bboxes[:, 5] - bboxes[:, 2]
        elif convention == 'xyzwhd':
            widths = bboxes[:, 3]
            heights = bboxes[:, 4]
            depths = bboxes[:, 5]
        else:
            raise ValueError(f"Unknown convention {convention}.")

        if (widths < 0).any() or (heights < 0).any() or (depths < 0).any():
            raise RuntimeError(
                f"Bounding boxes do not follow the {convention} convention.")

    def calculate_multiple_iou(self, bbox_a: np.ndarray, bbox_b: np.ndarray) -> np.ndarray:
        """
        Args:
            bbox_a: numpy.ndarray(dtype=np.uint8, shape=(num_a, 6))
                x0y0z0x1y1z1 convention.
            bbox_b: numpy.ndarray(dtype=np.uint8, shape=(num_b, 6))
                x0y0z0x1y1z1 convention.
        Returns:
            ious: numpy.ndarray(dtype=np.uint8, shape(num_a, num_b))
        """
        num_a = bbox_a.shape[0]
        num_b = bbox_b.shape[0]

        self.check_bbox_convention(bbox_a, 'x0y0z0x1y1z1')
        self.check_bbox_convention(bbox_b, 'x0y0z0x1y1z1')

        # num_a x 6 -> num_a x num_b x 6
        bbox_a = np.tile(bbox_a, num_b)
        bbox_a = np.expand_dims(bbox_a, axis=1).reshape((num_a, num_b, -1))

        # num_b x 6 -> num_b x num_a x 6
        bbox_b = np.tile(bbox_b, num_a)
        bbox_b = np.expand_dims(bbox_b, axis=1).reshape((num_b, num_a, -1))

        # num_b x num_a x 6 -> num_a x num_b x 6
        bbox_b = np.transpose(bbox_b, (1, 0, 2))

        # num_a x num_b
        min_x = np.maximum(bbox_a[:, :, 0], bbox_b[:, :, 0])
        min_y = np.maximum(bbox_a[:, :, 1], bbox_b[:, :, 1])
        min_z = np.maximum(bbox_a[:, :, 2], bbox_b[:, :, 2])
        max_x = np.minimum(bbox_a[:, :, 3], bbox_b[:, :, 3])
        max_y = np.minimum(bbox_a[:, :, 4], bbox_b[:, :, 4])
        max_z = np.minimum(bbox_a[:, :, 5], bbox_b[:, :, 5])

        # num_a x num_b
        vol_intersect = (np.maximum(0, max_x - min_x + 1)
                         * np.maximum(0, max_y - min_y + 1)
                         * np.maximum(0, max_z - min_z + 1))
        vol_a = ((bbox_a[:, :, 3] - bbox_a[:, :, 0] + 1) *
                 (bbox_a[:, :, 4] - bbox_a[:, :, 1] + 1) *
                 (bbox_a[:, :, 5] - bbox_a[:, :, 2] + 1))
        vol_b = ((bbox_b[:, :, 3] - bbox_b[:, :, 0] + 1) *
                 (bbox_b[:, :, 4] - bbox_b[:, :, 1] + 1) *
                 (bbox_b[:, :, 5] - bbox_b[:, :, 2] + 1))
        denominator = vol_a + vol_b - vol_intersect
        degenerate_indices = np.where(denominator <= 0)
        denominator[degenerate_indices] = 1

        ious = vol_intersect / denominator
        ious[degenerate_indices] = 0
        return ious

    def accumulate(self, heatmap: np.ndarray, mask: np.ndarray) -> None:
        """
        From a heatmap, a box is inferred (compute_bboxes_from_heatmaps).
        The box is compared against GT boxes. Count a heatmap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            heatmap: numpy.ndarray(size=(H, W, D), dtype=float)
            mask: numpy.ndarray(size=(H, W, D), dtype=np.uint8).
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
