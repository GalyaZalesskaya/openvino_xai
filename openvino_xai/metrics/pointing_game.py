# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from openvino_xai.common.utils import logger
from openvino_xai.explainer.explanation import Explanation


class PointingGame:
    @staticmethod
    def pointing_game(saliency_map: np.ndarray, image_gt_bboxes: List[Tuple[int, int, int, int]]) -> bool:
        """
        Implements the Pointing Game metric using a saliency map and bounding boxes of the same image and class.
        Returns a boolean indicating if any of the most salient points fall within the ground truth bounding boxes.

        :param saliency_map: A 2D numpy array representing the saliency map for the image.
        :type saliency_map: np.ndarray
        :param image_gt_bboxes: A list of tuples (x, y, w, h) representing the bounding boxes of the ground truth objects.
        :type image_gt_bboxes: List[Tuple[int, int, int, int]]

        :return: True if any of the most salient points fall within any of the ground truth bounding boxes, False otherwise.
        :rtype: bool
        """
        # Find the most salient points in the saliency map
        max_indices = np.argwhere(saliency_map == np.max(saliency_map))

        # If multiple bounding boxes are available for one image
        for x, y, w, h in image_gt_bboxes:
            for max_point_y, max_point_x in max_indices:
                # Check if this point is within the ground truth bounding box
                if x <= max_point_x <= x + w and y <= max_point_y <= y + h:
                    return True
        return False

    def evaluate(
        self, explanations: List[Explanation], gt_bboxes: List[Dict[str, List[Tuple[int, int, int, int]]]]
    ) -> float:
        """
        Evaluates the Pointing Game metric over a set of images. Skips saliency maps if the gt bboxes for this class are absent.

        :param explanations: A list of explanations for each image.
        :type explanations: List[Explanation]
        :param gt_bboxes: A list of dictionaries {label_name: lists of bounding boxes} for each image.
        :type gt_bboxes: List[Dict[str, List[Tuple[int, int, int, int]]]]

        :return: Pointing game score over a list of images
        :rtype: float
        """

        assert len(explanations) == len(
            gt_bboxes
        ), "Number of explanations and ground truth bounding boxes must match and equal to number of images."

        hits = 0
        num_sal_maps = 0
        for explanation, image_gt_bboxes in zip(explanations, gt_bboxes):
            label_names = explanation.label_names
            assert label_names is not None, "Label names are required for pointing game evaluation."

            for class_idx, class_sal_map in explanation.saliency_map.items():
                label_name = label_names[int(class_idx)]

                if label_name not in image_gt_bboxes:
                    logger.info(
                        f"No ground-truth bbox for {label_name} saliency map. "
                        f"Skip pointing game evaluation for this saliency map."
                    )
                    continue

                class_gt_bboxes = image_gt_bboxes[label_name]
                hits += self.pointing_game(class_sal_map, class_gt_bboxes)
                num_sal_maps += 1

        return hits / num_sal_maps if num_sal_maps > 0 else 0.0
