# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum
from typing import Any, Dict, List, Tuple
from time import time
from pathlib import Path

import numpy as np
import openvino as ov
import pytest
import pandas as pd

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
from tests.test_suite.dataset_utils import DatasetType, coco_anns_to_gt_bboxes, voc_anns_to_gt_bboxes, define_dataset_type

datasets = pytest.importorskip("torchvision.datasets")

class TestAccuracy:
    MODEL_NAME = "mlc_mobilenetv3_large_voc"

    @pytest.fixture(autouse=True)
    def setup(self, fxt_data_root, fxt_output_root, fxt_dataset_parameters):
        data_dir = fxt_data_root
        retrieve_otx_model(data_dir, self.MODEL_NAME)
        model_path = data_dir / "otx_models" / (self.MODEL_NAME + ".xml")
        model = ov.Core().read_model(model_path)
    
        self.dataset_parameters = fxt_dataset_parameters
        self.output_dir = fxt_output_root

        self.preprocess_fn = get_preprocess_fn(
            change_channel_order=False, #self.channel_format == "BGR",
            input_size=(224, 224),
            hwc_to_chw=True,
        )
        self.postprocess_fn = get_postprocess_fn(activation=ActivationType.SIGMOID)

        self.explainer = Explainer(
            model=model,
            task=Task.CLASSIFICATION,
            preprocess_fn=self.preprocess_fn,
            explain_mode=ExplainMode.WHITEBOX,
        )

        self.pointing_game = PointingGame()
        self.auc = InsertionDeletionAUC(model, self.preprocess_fn, self.postprocess_fn)
        self.adcc = ADCC(model, self.preprocess_fn, self.postprocess_fn, self.explainer)

    def setup_dataset(self, data_root: str, ann_path: str):
        self.dataset_type = define_dataset_type(data_root, ann_path)
        self.channel_format = "RGB" if self.dataset_type in [DatasetType.VOC, DatasetType.COCO] else "None"

        if self.dataset_type == DatasetType.COCO:
            self.dataset = datasets.CocoDetection(root=data_root, annFile=ann_path)
            self.dataset_labels_dict = {cats["id"]: cats["name"] for cats in self.dataset.coco.cats.values()}
            self.anns_to_gt_bboxes = coco_anns_to_gt_bboxes
        elif self.dataset_type == DatasetType.VOC:
            self.dataset = datasets.VOCDetection(root=data_root, download=False, year="2012", image_set="val")
            self.dataset_labels_dict = None
            self.anns_to_gt_bboxes = voc_anns_to_gt_bboxes

    def test_explainer_images(self):
        records = []
        for data_root, ann_path in self.dataset_parameters:

            # data_metric_path = self.output_dir / self.MODEL_NAME
            data_metric_path = Path("/home/gzalessk/code/openvino_xai/tests/perf/reports") / self.MODEL_NAME
            os.makedirs(data_metric_path, exist_ok=True)

            self.setup_dataset(data_root, ann_path)
            dataset_name  = "coco" if self.dataset_type == DatasetType.COCO else "voc"

            batch_size = 100
            for batch_idx in range(0, len(self.dataset), batch_size):
                lrange = batch_idx*batch_size
                rrange = min(len(self.dataset), (batch_idx+1)*batch_size)

                start_time = time()
                images, explanations, dataset_gt_bboxes = [], [], []
                # for image, anns in self.dataset[lrange:rrange]:
                for i in range(lrange, rrange):
                    image, anns = self.dataset[i]
                    image_np = np.array(image)
                    gt_bbox_dict = self.anns_to_gt_bboxes(anns, self.dataset_labels_dict)
                    targets = [target for target in gt_bbox_dict.keys() if target in VOC_NAMES]

                    explanation = self.explainer(image_np, targets=targets, label_names=VOC_NAMES, colormap=False)

                    images.append(image_np)
                    explanations.append(explanation)
                    dataset_gt_bboxes.append({key: value for key, value in gt_bbox_dict.items() if key in targets})

                record = {"range": f"{lrange}-{rrange}"}
                record.update(self.get_scores_times(explanations, images, dataset_gt_bboxes, start_time))
                records.append(record)

            df = pd.DataFrame(records)
            df.to_csv(data_metric_path / f"accuracy_{dataset_name}.csv", index=False)
        
            mean_scores = [{key: np.mean([record[key] for record in records]) for key in records[0].keys() if key!='range'}]
            df = pd.DataFrame(mean_scores)
            df.to_csv(data_metric_path / f"mean_accuracy_{dataset_name}.csv", index=False)

        return records

    def get_scores_times(self, explanations, images, dataset_gt_bboxes, start_time):
        score = {}
        explain_time = time() - start_time
        score["explain_time"] = explain_time

        previous_time = time()
        score.update(self.pointing_game.evaluate(explanations, dataset_gt_bboxes))
        score["pointing_game_time"] = time() - previous_time

        previous_time = time()
        score.update(self.auc.evaluate(explanations, images, steps=10))
        score["auc_time"] = time() - previous_time

        previous_time = time()
        score.update(self.adcc.evaluate(explanations, images))
        score["adcc_time"] = time() - previous_time

        for metric in ["explain_time", "pointing_game_time", "auc_time","adcc_time", "pointing_game", "insertion", "deletion", "delta", "adcc" ]:
            score[metric] = float(f'{score[metric]:.2f}')
        return score
