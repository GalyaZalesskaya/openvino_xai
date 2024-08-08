# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List

import cv2
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

    def save(
        self,
        dir_path: Path | str,
        image_name_prefix: str | None = "",
        target_prefix: str | None = "target",
        target_suffix: str | None = "",
        confidence_scores: Dict[int, float] | None = None,
    ) -> None:
        """
        Dumps saliency map images to the specified directory.

        Allows flexibly name the files with the image_name_prefix, target_prefix, and target_suffix.
        For the name 'image_name_target_aeroplane.jpg': prefix = 'image_name',
        target_prefix = 'target', label name = 'aeroplane', target_suffix = ''.

        save(output_dir) -> target_aeroplane.jpg
        save(output_dir, image_name_prefix="test_map", target_prefix="") -> test_map_aeroplane.jpg
        save(output_dir, image_name_prefix="test_map") -> test_map_target_aeroplane.jpg
        save(output_dir, target_suffix="conf", confidence_scores=scores) -> target_aeroplane_conf_0.92.jpg

        Parameters:
        :param dir_path: The directory path where the saliency maps will be saved.
        :type dir_path: Path | str
        :param image_name_prefix: Optional prefix for the file names. Default is an empty string.
        :type image_name_prefix: str | None
        :param target_prefix: Optional suffix for the target. Default is "target".
        :type target_prefix: str | None
        :param target_suffix: Optional suffix for the saliency map name. Default is an empty string.
        :type target_suffix: str | None
        :param confidence_scores: Dict with confidence scores for each class to saliency maps with them1 Default is None.
        :type confidence_scores: Dict[int, float] | None

        """

        os.makedirs(dir_path, exist_ok=True)

        image_name_prefix = f"{image_name_prefix}_" if image_name_prefix != "" else image_name_prefix
        target_suffix = f"_{target_suffix}" if target_suffix != "" else target_suffix
        template = f"{{image_name_prefix}}{{target_prefix}}{{target_name}}{target_suffix}.jpg"

        target_prefix = f"{target_prefix}_" if target_prefix != "" else target_prefix
        for cls_idx, map_to_save in self._saliency_map.items():
            map_to_save = cv2.cvtColor(map_to_save, code=cv2.COLOR_RGB2BGR)
            if isinstance(cls_idx, str):
                target_name = ""
                if target_prefix == "target_":
                    # Default activation map suffix
                    target_prefix = "activation_map"
                elif target_prefix == "":
                    # Remove the underscore in case of empty suffix
                    image_name_prefix = image_name_prefix[:-1] if image_name_prefix.endswith("_") else image_name_prefix
            else:
                target_name = self.label_names[cls_idx] if self.label_names else str(cls_idx)
                if confidence_scores:
                    class_confidence = confidence_scores[cls_idx]
                    target_name = f"{target_name}_{class_confidence:.2f}"

            image_name = template.format(
                image_name_prefix=image_name_prefix, target_prefix=target_prefix, target_name=target_name
            )
            cv2.imwrite(os.path.join(dir_path, image_name), img=map_to_save)


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
