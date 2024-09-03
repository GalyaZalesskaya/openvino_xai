# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import openvino as ov
import pytest
from pathlib import Path

from openvino_xai import Task
from openvino_xai.common.utils import retrieve_otx_model
from openvino_xai.explainer.explainer import Explainer, ExplainMode
from openvino_xai.explainer.utils import (
    ActivationType,
    get_postprocess_fn,
    get_preprocess_fn,
)
from openvino_xai.metrics import ADCC, InsertionDeletionAUC, PointingGame
from tests.unit.explanation.test_explanation_utils import VOC_NAMES

datasets = pytest.importorskip("torchvision.datasets")


class DatasetType(Enum):
    COCO = "coco"
    VOC = "voc"


def coco_anns_to_gt_bboxes(
    anns: List[Dict[str, Any]] | Dict[str, Any], coco_val_labels: Dict[int, str]
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    gt_bboxes = {}
    for ann in anns:
        category_id = ann["category_id"]
        category_name = coco_val_labels[category_id]
        bbox = ann["bbox"]
        if category_name not in gt_bboxes:
            gt_bboxes[category_name] = []
        gt_bboxes[category_name].append(bbox)
    return gt_bboxes


def voc_anns_to_gt_bboxes(
    anns: List[Dict[str, Any]] | Dict[str, Any], *args: Any
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    gt_bboxes = {}
    anns = anns["annotation"]["object"]
    for ann in anns:
        category_name = ann["name"]
        bndbox = list(map(float, ann["bndbox"].values()))
        bndbox = np.array(bndbox, dtype=np.int32)
        x_min, y_min, x_max, y_max = bndbox
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        if category_name not in gt_bboxes:
            gt_bboxes[category_name] = []
        gt_bboxes[category_name].append(bbox)
    return gt_bboxes


def define_dataset_type(data_root: Path, ann_path: Path) -> DatasetType:
    if data_root and ann_path and ann_path.suffix == ".json":
        if any(image_name.endswith(".jpg") for image_name in os.listdir(data_root)):
            return DatasetType.COCO

    required_voc_dirs1 = {"JPEGImages", "ImageSets", "Annotations"}
    required_voc_dirs2 = {"Data", "ImageSets", "Annotations"}
    for _, dir, _ in os.walk(data_root):
        if required_voc_dirs1.issubset(set(dir)) or required_voc_dirs2.issubset(set(dir)):
            return DatasetType.VOC

    raise ValueError("Dataset type is not supported")