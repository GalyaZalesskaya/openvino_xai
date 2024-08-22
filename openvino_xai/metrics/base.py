from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple

import numpy as np
import openvino.runtime as ov

from openvino_xai.common.utils import IdentityPreprocessFN
from openvino_xai.explainer.explanation import Explanation


class BaseMetric(ABC):
    """Base class for XAI quality metric."""

    def __init__(
        self,
        model_compiled: ov.ie_api.CompiledModel = None,
        preprocess_fn: Callable[[np.ndarray], np.ndarray] = IdentityPreprocessFN(),
        postprocess_fn: Callable[[np.ndarray], np.ndarray] = None,
    ):
        self.model_compiled = model_compiled
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def model_predict(self, input: np.ndarray) -> np.ndarray:
        logits = self.model_compiled([self.preprocess_fn(input)])
        logits = self.postprocess_fn(logits)[0]
        return logits

    @abstractmethod
    def evaluate(
        self, explanations: List[Explanation], *args: Any, **kwargs: Any
    ) -> float | Tuple[float, float, float]:
        """Evaluate the quality of saliency maps over the list of images"""
