# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from openvino_xai.explainer.utils import (
    convert_targets_to_numpy,
    explains_all,
    get_target_indices,
)


class Explanation:
    """
    Explanation selects target saliency maps, holds it and its layout.

    :param saliency_map: Raw saliency map, as a numpy array or as a dict.
    :type saliency_map: np.ndarray | Dict[int | str, np.ndarray]
    :param targets: List of custom labels to explain, optional. Can be list of integer indices (int),
        or list of names (str) from label_names.
    :type targets: np.ndarray | List[int | str] | int | str
    :param label_names: List of all label names.
    :type label_names: List[str] | None
    """

    def __init__(
        self,
        saliency_map: np.ndarray | Dict[int | str, np.ndarray],
        targets: np.ndarray | List[int | str] | int | str,
        label_names: List[str] | None = None,
    ):
        targets = convert_targets_to_numpy(targets)

        if isinstance(saliency_map, np.ndarray):
            self._check_saliency_map(saliency_map)
            self._saliency_map = self._format_sal_map_as_dict(saliency_map)
            self.total_num_targets = len(self._saliency_map)
        elif isinstance(saliency_map, dict):
            self._saliency_map = saliency_map
            self.total_num_targets = None
        else:
            raise ValueError(f"Expect saliency_map to be np.ndarray or dict, but got{type(saliency_map)}.")

        if "per_image_map" in self._saliency_map:
            self.layout = Layout.ONE_MAP_PER_IMAGE_GRAY
        else:
            self.layout = Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY

        if not explains_all(targets) and not self.layout == Layout.ONE_MAP_PER_IMAGE_GRAY:
            self._saliency_map = self._select_target_saliency_maps(targets, label_names)

        self.label_names = label_names

    @property
    def saliency_map(self) -> Dict[int | str, np.ndarray]:
        """Saliency map as a dict {target_id: np.ndarray}."""
        return self._saliency_map

    @saliency_map.setter
    def saliency_map(self, saliency_map: Dict[int | str, np.ndarray]):
        self._saliency_map = saliency_map

    @property
    def shape(self):
        """Shape of the saliency map."""
        idx = next(iter(self._saliency_map))
        shape = self._saliency_map[idx].shape
        return shape

    @property
    def targets(self):
        """Explained targets."""
        return list(self._saliency_map.keys())

    @staticmethod
    def _check_saliency_map(saliency_map: np.ndarray):
        if saliency_map is None:
            raise RuntimeError("Saliency map is None.")
        if saliency_map.size == 0:
            raise RuntimeError("Saliency map is zero size array.")
        if saliency_map.shape[0] > 1:
            raise RuntimeError("Batch size for saliency maps should be 1.")

    @staticmethod
    def _format_sal_map_as_dict(raw_saliency_map: np.ndarray) -> Dict[int | str, np.ndarray]:
        """Returns dict with saliency maps in format {target_id: class_saliency_map}."""
        dict_sal_map: Dict[int | str, np.ndarray]
        if raw_saliency_map.ndim == 3:
            # Per-image saliency map
            dict_sal_map = {"per_image_map": raw_saliency_map[0]}
        elif raw_saliency_map.ndim == 4:
            # Per-target saliency map
            dict_sal_map = {}
            for index, sal_map in enumerate(raw_saliency_map[0]):
                dict_sal_map[index] = sal_map
        else:
            raise ValueError(
                f"Raw saliency map has to be tree or four dimensional tensor, " f"but got {raw_saliency_map.ndim}."
            )
        return dict_sal_map

    def _select_target_saliency_maps(
        self,
        targets: np.ndarray | List[int | str],
        label_names: List[str] | None = None,
    ) -> Dict[int | str, np.ndarray]:
        assert self.layout == Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY
        target_indices = self._select_target_indices(
            targets=targets,
            label_names=label_names,
        )
        saliency_maps_selected = {i: self._saliency_map[i] for i in target_indices}
        return saliency_maps_selected

    def _select_target_indices(
        self,
        targets: np.ndarray | List[int | str],
        label_names: List[str] | None = None,
    ) -> List[int] | np.ndarray:
        target_indices = get_target_indices(targets, label_names)
        if self.total_num_targets is not None:
            if not all(0 <= target_index <= (self.total_num_targets - 1) for target_index in target_indices):
                raise ValueError(f"All targets indices have to be in range 0..{self.total_num_targets - 1}.")
        else:
            if not all(target_index in self.saliency_map for target_index in target_indices):
                raise ValueError("Provided targer index {targer_index} is not available among saliency maps.")
        return target_indices

    def save(self, dir_path: Path | str, name: str | None = None) -> None:
        """Dumps saliency map."""
        os.makedirs(dir_path, exist_ok=True)
        save_name = name if name else ""
        for cls_idx, map_to_save in self._saliency_map.items():
            map_to_save = cv2.cvtColor(map_to_save, code=cv2.COLOR_RGB2BGR)
            if isinstance(cls_idx, str):
                cv2.imwrite(os.path.join(dir_path, f"{save_name}.jpg"), img=map_to_save)
                return
            else:
                if self.label_names:
                    target_name = self.label_names[cls_idx]
                else:
                    target_name = str(cls_idx)
            image_name = f"{save_name}_target_{target_name}.jpg" if save_name else f"target_{target_name}.jpg"
            cv2.imwrite(os.path.join(dir_path, image_name), img=map_to_save)

    def plot(self, targets: np.ndarray | List[int | str] | None = None, backend="matplotlib") -> None:
        """
        Plots saliency maps using the specified backend.

        This function plots available saliency maps using the specified backend. Additionally, target classes
        can be specified by passing a list or array of target class indices or names. If a provided class is
        not available among the saliency maps, it is omitted.

        Args:
            targets (np.ndarray | List[int | str] | None): A list or array of target class indices or names to plot.
                By default, it's None, and all available saliency maps are plotted.
            backend (str): The plotting backend to use. Can be either 'matplotlib' (recommended for Jupyter)
                or 'cv' (recommended for Python scripts). Default is 'matplotlib'.
        """

        if targets is None or explains_all(targets):
            class_indices = self.targets
        else:
            target_indices = get_target_indices(targets, self.label_names)
            class_indices = [cls_idx for cls_idx in target_indices if cls_idx in self.targets]

        if backend == "matplotlib":
            self._plot_matplotlib(class_indices)
        elif backend == "cv":
            self._plot_cv(class_indices)
        else:
            raise ValueError(f"Unknown backend {backend}. Use 'matplotlib' or 'cv'.")

    def _plot_matplotlib(self, class_indices: list[int | str]) -> None:
        """Plots saliency maps using matplotlib."""
        if len(class_indices) <= 4:  # Horizontal layout
            _, axes = plt.subplots(len(class_indices), 1)
        else:  # Vertical layout
            _, axes = plt.subplots(1, len(class_indices))
        axes = [axes] if len(class_indices) == 1 else axes

        for i, cls_idx in enumerate(class_indices):
            if self.label_names and isinstance(cls_idx, int):
                label_name = f"{self.label_names[cls_idx]} ({cls_idx})"
            else:
                label_name = str(cls_idx)

            map_to_plot = self.saliency_map[cls_idx]

            axes[i].imshow(map_to_plot)
            axes[i].axis("off")  # Hide the axis
            axes[i].set_title(f"Class {label_name}")

        plt.tight_layout()
        plt.show()

    def _plot_cv(self, class_indices: list[int | str], wait_time: int = 0) -> None:
        """Plots saliency maps using OpenCV."""
        for cls_idx in class_indices:
            if self.label_names and isinstance(cls_idx, int):
                label_name = f"{self.label_names[cls_idx]} ({cls_idx})"
            else:
                label_name = str(cls_idx)

            map_to_plot = self.saliency_map[cls_idx]
            map_to_plot = cv2.cvtColor(map_to_plot, cv2.COLOR_BGR2RGB)

            cv2.imshow(f"Class {label_name}", map_to_plot)
            cv2.waitKey(wait_time)
        cv2.destroyAllWindows()


class Layout(Enum):
    """
    Enum describes different saliency map layouts.

    Saliency map can have the following layout:
        ONE_MAP_PER_IMAGE_GRAY - BHW - one map per image
        ONE_MAP_PER_IMAGE_COLOR - BHWC - one map per image, colormapped
        MULTIPLE_MAPS_PER_IMAGE_GRAY - BNHW - multiple maps per image
        MULTIPLE_MAPS_PER_IMAGE_COLOR - BNHWC - multiple maps per image, colormapped
    """

    ONE_MAP_PER_IMAGE_GRAY = "one_map_per_image_gray"
    ONE_MAP_PER_IMAGE_COLOR = "one_map_per_image_color"
    MULTIPLE_MAPS_PER_IMAGE_GRAY = "multiple_maps_per_image_gray"
    MULTIPLE_MAPS_PER_IMAGE_COLOR = "multiple_maps_per_image_color"


GRAY_LAYOUTS = {
    Layout.ONE_MAP_PER_IMAGE_GRAY,
    Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
}
COLOR_MAPPED_LAYOUTS = {
    Layout.ONE_MAP_PER_IMAGE_COLOR,
    Layout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
MULTIPLE_MAP_LAYOUTS = {
    Layout.MULTIPLE_MAPS_PER_IMAGE_GRAY,
    Layout.MULTIPLE_MAPS_PER_IMAGE_COLOR,
}
ONE_MAP_LAYOUTS = {
    Layout.ONE_MAP_PER_IMAGE_GRAY,
    Layout.ONE_MAP_PER_IMAGE_COLOR,
}
