from typing import List, Tuple

import numpy as np


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return np.abs((arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1))


class InsertionDeletionAUC:
    def __init__(self, compiled_model, preprocess_fn, postprocess_fn):
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.compiled_model = compiled_model

    def predict(self, input) -> np.ndarray:
        logits = self.compiled_model([self.preprocess_fn(input)])
        logits = self.postprocess_fn(logits)[0]
        return logits

    def insertion_deletion_auc(self, input_image, class_idx, saliency_map, steps=100):
        """
        Calculate the Insertion AUC metric for images.

        Parameters:
        - model: the model to evaluate.
        - input_image: the input image to the model (H, W, C).
        - class_idx: the class of saliency map to evaluate.
        - saliency_map: importance scores for each pixel (H, W).
        - steps: number of steps for inserting pixels.

        Returns:
        - insertion_auc_score: the calculated AUC for insertion.
        """
        # Values to start
        baseline_insertion = np.full_like(input_image, 0)
        baseline_deletion = input_image

        # Sort pixels by descending importance to find the most important pixels
        sorted_indices = np.argsort(-saliency_map.flatten())
        sorted_indices = np.unravel_index(sorted_indices, saliency_map.shape)

        insertion_scores, deletion_scores = [], []
        for i in range(steps + 1):
            temp_image_insertion = baseline_insertion.copy()
            temp_image_deletion = baseline_deletion.copy()

            num_pixels = int(i * len(sorted_indices[0]) / steps)
            x_indices = sorted_indices[0][:num_pixels]
            y_indices = sorted_indices[1][:num_pixels]

            # Insert the image on the places of the important pixels
            temp_image_insertion[x_indices, y_indices] = input_image[x_indices, y_indices]
            # Remove image pixels on the places of the important pixels
            temp_image_deletion[x_indices, y_indices] = 0

            # Predict and record the score
            # cv2.imshow("temp_image", temp_image)
            temp_logits_insertion = self.predict(temp_image_insertion)
            temp_logits_deletion = self.predict(temp_image_deletion)

            insertion_scores.append(temp_logits_insertion[class_idx])
            deletion_scores.append(temp_logits_deletion[class_idx])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        insertion_auc_score = auc(np.array(insertion_scores))
        deletion_auc_score = auc(np.array(deletion_scores))
        return insertion_auc_score, deletion_auc_score

    def evaluate(self, explanations: List, input_images: List[np.ndarray], steps: int) -> Tuple[float, float, float]:
        """
        Evaluate the insertion and deletion AUC for given explanations and input images.

        :param explanations: List of explanation objects containing saliency maps.
        :param input_images: List of input images as numpy arrays.
        :param steps: Number of steps for the insertion and deletion process.
        :return: A tuple containing the mean insertion AUC, mean deletion AUC, and their difference (delta).
        """
        insertions, deletions = [], []
        for input_image, explanation in zip(input_images, explanations):
            for class_idx, saliency_map in explanation.saliency_map.items():
                insertion, deletion = self.insertion_deletion_auc(input_image, class_idx, saliency_map, steps)
                insertions.append(insertion)
                deletions.append(deletion)

        insertion = np.mean(np.array(insertions))
        deletion = np.mean(np.array(deletion))
        delta = insertion - deletion

        return insertion, deletion, delta
