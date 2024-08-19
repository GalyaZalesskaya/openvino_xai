import numpy as np
from scipy import stats as STS

from openvino_xai import Task
from openvino_xai.explainer.explainer import Explainer, ExplainMode


class ADCC:
    def __init__(self, model, compiled_model, preprocess_fn, postprocess_fn):
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.compiled_model = compiled_model
        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

    def predict(self, input) -> int:
        logits = self.compiled_model([self.preprocess_fn(input)])[0]
        logits = self.postprocess_fn(logits)
        return logits

    def average_drop(self, image, saliency_map, model_output, class_idx=None):
        """It measures the average percentage drop in
        confidence for the target class c when the model sees only
        the explanation map, instead of the full image."""

        masked_image = image * saliency_map[:, :, None]

        # if masked_image.ndim == 2:
        #     masked_image = masked_image[: , :, None]

        confidence_on_input = np.max(model_output)
        if class_idx is None:
            class_idx = np.argmax(model_output)

        prediction_on_saliency_map = self.predict(masked_image)
        confidence_on_saliency_map = prediction_on_saliency_map[class_idx]

        return max(0.0, confidence_on_input - confidence_on_saliency_map) / confidence_on_input

    def complexity(self, saliency_map):
        """
        Saliency map has to be as less complex as possible, i.e., it must contain the minimum set of pixels that explains the prediction.
        Defined as L1 norm of the saliency map. Complexity is minimized when the number of pixels highlighted by the attribution method is low.

        """
        return abs(saliency_map).sum() / (saliency_map.shape[-1] * saliency_map.shape[-2])

    def coherency(self, image, saliency_map, model_output, class_idx=None):
        """Maximum Coherency. The CAM should contain all the
        relevant features that explain a prediction and should remove useless features in a coherent way. As a consequence,
        given an input image x and a class of interest c, the CAM
        of x should not change when conditioning x on the CAM
        itself"""
        if not (np.max(saliency_map) <= 1 and np.min(saliency_map) >= 0):
            saliency_map = saliency_map / 255  # Normalize to [0, 1]

        assert (
            np.max(saliency_map) <= 1 and np.min(saliency_map) >= 0
        ), "Saliency map should be normalized between 0 and 1"

        masked_image = image * saliency_map[:, :, None]

        if class_idx is None:
            class_idx = np.argmax(model_output)

        saliency_map_mapped_image = self.explainer(
            masked_image,
            targets=[class_idx],
            colormap=False,
        )

        # Find a way to return not scaled salinecy map [0, 1]
        saliency_map_mapped_image = saliency_map_mapped_image.saliency_map[class_idx]
        if not (np.max(saliency_map_mapped_image) <= 1 and np.min(saliency_map_mapped_image) >= 0):
            saliency_map_mapped_image = saliency_map_mapped_image / 255  # Normalize to [0, 1]

        A, B = saliency_map, saliency_map_mapped_image

        """
        # Pearson correlation coefficient
        # """
        Asq, Bsq = A.flatten(), B.flatten()

        y, _ = STS.pearsonr(Asq, Bsq)
        y = (y + 1) / 2

        return y

    def adcc(self, image, saliency_map, target_class_idx=None):
        # TODO test target_class_idx

        model_output = self.predict(image)

        avgdrop = self.average_drop(image, saliency_map, model_output, class_idx=target_class_idx)
        coh = self.coherency(image, saliency_map, model_output, class_idx=target_class_idx)
        com = self.complexity(saliency_map)

        adcc = 3 / (1 / coh + 1 / (1 - com) + 1 / (1 - avgdrop))

        return adcc
