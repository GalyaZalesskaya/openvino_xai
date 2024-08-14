from typing import List, Tuple

import numpy as np


class PointingGame:
    @staticmethod
    def pointing_game(saliency_map: np.ndarray, gt_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Implements the Pointing Game metric using bounding boxes.

        Parameters:
        - saliency_map: A 2D numpy array representing the saliency map for the image.
        - gt_bbox: A tuple (x, y, w, h) representing the bounding box of the ground truth object.

        Returns:
        - hit: A boolean indicating if any of the most salient point falls within the ground truth bounding box.
        """
        x, y, w, h = gt_bbox

        # Find the most salient points in the saliency map
        max_indices = np.argwhere(saliency_map == np.max(saliency_map))

        for max_point_y, max_point_x in max_indices:
            # Check if this point is within the ground truth bounding box
            if x <= max_point_x <= x + w and y <= max_point_y <= y + h:
                return True
        return False

    def evaluate(self, saliency_maps: List[np.ndarray], gt_bboxes: List[Tuple[int, int, int, int]]) -> float:
        """
        Evaluates the Pointing Game metric over a set of images.

        Parameters:
        - saliency_maps: A list of 2D numpy arrays representing the saliency maps.
        - ground_truth_bbs: A list of bounding box of the ground truth object.

        Returns:
        - score: The Pointing Game accuracy score over the dataset.
        """
        assert len(saliency_maps) == len(
            gt_bboxes
        ), "Number of saliency maps and ground truth bounding boxes must match."

        hits = sum([self.pointing_game(s_map, gt_map) for s_map, gt_map in zip(saliency_maps, gt_bboxes)])
        score = hits / len(saliency_maps)
        return score
