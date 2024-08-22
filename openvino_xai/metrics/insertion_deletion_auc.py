from typing import Any, List, Tuple

import numpy as np

from openvino_xai.explainer.explanation import Explanation, Layout
from openvino_xai.metrics.base import BaseMetric


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return np.abs((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1))


class InsertionDeletionAUC(BaseMetric):
    """
    Implementation of the Insertion and Deletion AUC by Petsiuk et al. 2018.

    References:
        Petsiuk, Vitali, Abir Das, and Kate Saenko. "Rise: Randomized input sampling
        for explanation of black-box models." arXiv preprint arXiv:1806.07421 (2018).
    """

    def insertion_deletion_auc(
        self, saliency_map: np.ndarray, class_idx: int, input_image: np.ndarray, steps: int = 100
    ) -> Tuple[float, float]:
        """
        Calculate the Insertion and Deletion AUC metrics for one saliency map for one class.

        Parameters:
        :param saliency_map: Importance scores for each pixel (H, W).
        :type saliency_map: np.ndarray
        :param class_idx: The class of saliency map to evaluate.
        :type class_idx: int
        :param input_image: The input image to the model (H, W, C).
        :type input_image: np.ndarray
        :param steps: Number of steps for inserting pixels.
        :type steps: int

        Returns:
        - insertion_auc_score: Saliency map AUC for insertion.
        - deletion_auc_score: Saliency map AUC for deletion.
        """
        # Values to start
        baseline_insertion = np.full_like(input_image, 0)
        baseline_deletion = input_image

        # Sort pixels by descending importance to find the most important pixels
        sorted_indices = np.argsort(-saliency_map.flatten())
        sorted_indices = np.unravel_index(sorted_indices, saliency_map.shape)

        insertion_scores, deletion_scores = [], []
        for i in range(steps + 1):
            temp_image_insertion, temp_image_deletion = baseline_insertion.copy(), baseline_deletion.copy()

            num_pixels = int(i * len(sorted_indices[0]) / steps)
            x_indices = sorted_indices[0][:num_pixels]
            y_indices = sorted_indices[1][:num_pixels]

            # Insert the image on the places of the important pixels
            temp_image_insertion[x_indices, y_indices] = input_image[x_indices, y_indices]
            # Remove image pixels on the places of the important pixels
            temp_image_deletion[x_indices, y_indices] = 0

            # Predict on masked image
            temp_logits_insertion = self.model_predict(temp_image_insertion)
            temp_logits_deletion = self.model_predict(temp_image_deletion)

            insertion_scores.append(temp_logits_insertion[class_idx])
            deletion_scores.append(temp_logits_deletion[class_idx])
        return auc(np.array(insertion_scores)), auc(np.array(deletion_scores))

    def evaluate(
        self, explanations: List[Explanation], input_images: List[np.ndarray], steps: int, **kwargs: Any
    ) -> Tuple[float, float, float]:
        """
        Evaluate the insertion and deletion AUC over the list of images and its saliency maps.

        :param explanations: List of explanation objects containing saliency maps.
        :type explanations: List[Explanation]
        :param input_images: List of input images as numpy arrays.
        :type input_images: List[np.ndarray]
        :param steps: Number of steps for the insertion and deletion process.
        :type steps: int

        :return: A tuple containing the mean insertion AUC, mean deletion AUC, and their difference (delta).
        :rtype: float
        """
        for explanation in explanations:
            assert explanation.layout in [Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY, Layout.MULTIPLE_MAPS_PER_IMAGE_COLOR]

        results = []
        for input_image, explanation in zip(input_images, explanations):
            for class_idx, saliency_map in explanation.saliency_map.items():
                insertion, deletion = self.insertion_deletion_auc(saliency_map, int(class_idx), input_image, steps)
                results.append([insertion, deletion])

        insertion, deletion = np.mean(np.array(results), axis=0)
        delta = insertion - deletion
        return insertion, deletion, delta
