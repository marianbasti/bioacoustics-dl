# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys
import torch
import numpy as np
import os


from typing import Optional
from dataclasses import dataclass, field
from omegaconf import MISSING

from fairseq.tasks import register_task
from fairseq.logging import metrics
from sklearn import metrics as sklearn_metrics
from .pretraining_AS2M import MaeImagePretrainingTask,MaeImagePretrainingConfig

from ..data.add_class_target_dataset import AddClassTargetDataset


logger = logging.getLogger(__name__)


@dataclass
class MaeImageClassificationConfig(MaeImagePretrainingConfig):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    input_size: int = 224
    local_cache_path: Optional[str] = None

    rebuild_batches: bool = True
    label_descriptors: str = "label_descriptors.csv"
    labels: str = "lbl"


@register_task("anuraset_classification", dataclass=MaeImageClassificationConfig)
class MaeImageClassificationTask_anuraset(MaeImagePretrainingTask):
    """ """

    cfg: MaeImageClassificationConfig
    
    def __init__(
        self,
        cfg: MaeImageClassificationConfig,
    ):
        super().__init__(cfg)

        self.state.add_factory("labels", self.load_labels)
        
    def load_labels(self):
        labels = {}
        path = os.path.join(self.cfg.data, self.cfg.label_descriptors)
        with open(path, "r") as ldf:
            for line in ldf:
                if line.strip() == "":
                    continue
                idx, species = line.strip().split(",")
                assert species not in labels, species
                labels[species] = int(idx)
        return labels

    @property
    def labels(self):
        return self.state.labels


    @classmethod
    def setup_task(cls, cfg: MaeImageClassificationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: MaeImageClassificationConfig = None, **kwargs):
        logger.info(f"Loading dataset for split: {split}\n From path: {self.cfg.data}")
        try:
            super().load_dataset(split, task_cfg, **kwargs)
            logger.info(f"Base dataset loaded. Dataset size: {len(self.datasets[split])}")
        except Exception as e:
            logger.error(f"Error loading dataset for split {split}: {str(e)}")
            raise
        
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
        
        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        logger.info(f"Loading labels from: {label_path}")
        
        # Ensure the label file exists
        if not os.path.exists(label_path):
            logger.error(f"Label file not found: {label_path}")
            raise FileNotFoundError(f"Label file not found: {label_path}")
            
        skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
        
        # Read all labels first, with error handling
        try:
            all_labels = []
            with open(label_path, "r") as f:
                for line in f:
                    label_vector = [0] * len(self.labels)
                    if len(line.rstrip().split('\t')) > 1:
                        label_entries = line.rstrip().split('\t')[1].split()
                        for entry in label_entries:
                            try:
                                species, level = entry.split('=')
                                if species in self.labels:
                                    label_vector[self.labels[species]] = int(level)
                            except ValueError as e:
                                logger.warning(f"Invalid label format in line: {line.strip()}, Error: {str(e)}")
                                continue
                    all_labels.append(label_vector)
            logger.info(f"Successfully read labels from {label_path}")
        except Exception as e:
            logger.error(f"Error reading label file {label_path}: {str(e)}")
            raise
        
        # Filter out skipped indices
        labels = [label for i, label in enumerate(all_labels) if i not in skipped_indices]

        # Add logging for label processing
        logger.info(f"Number of skipped indices: {len(skipped_indices)}")
        
        if len(labels) != len(self.datasets[split]):
            error_msg = (
                f"Number of labels ({len(labels)}) does not match dataset size ({len(self.datasets[split])}) "
                f"for split {split}. This might indicate a mismatch between audio files and labels."
            )
            logger.error(error_msg)
            raise AssertionError(error_msg)

        self.datasets[split] = AddClassTargetDataset(
            self.datasets[split],
            labels,
            multi_class=True,
            add_to_input=True,
            num_classes=len(self.labels)
        )
        
        # Log final dataset info
        logger.info(f"Final dataset size for split {split}: {len(self.datasets[split])}")
            

    def build_model(self, model_cfg: MaeImageClassificationConfig, from_checkpoint=False):
        # Set num_classes based on number of species in label_descriptors
        model_cfg.num_classes = len(self.labels)
        
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if (actualized_cfg is not None) and hasattr(actualized_cfg, "pretrained_model_args"):
            model_cfg.pretrained_model_args = actualized_cfg.pretrained_model_args

        return model
    
    def calculate_stats(self, output, target):
        classes_num = target.shape[-1]
        stats = []

        # Calculate Mean Squared Error for level prediction
        mse = np.mean((output - target) ** 2)
        
        # Calculate accuracy per level
        correct_levels = np.sum(np.round(output) == target)
        total_levels = output.size
        level_accuracy = correct_levels / total_levels

        # Per-class statistics
        for k in range(classes_num):
            # Calculate metrics for each species
            mse_class = np.mean((output[:, k] - target[:, k]) ** 2)
            
            dict = {
                "MSE": mse_class,
            }
            stats.append(dict)

        # Add global metrics
        stats.append({
            "global_MSE": mse,
            "level_accuracy": level_accuracy
        })
        
        return stats
    
    
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if "correct" in logging_outputs[0]:
            zero = torch.scalar_tensor(0.0)
            correct = sum(log.get("correct", zero) for log in logging_outputs)
            metrics.log_scalar_sum("_correct", correct)

            metrics.log_derived(
                "accuracy",
                lambda meters: 100 * meters["_correct"].sum / meters["sample_size"].sum,
            )
            
        elif "_predictions" in logging_outputs[0]:
            metrics.log_concat_tensor(
                "_predictions",
                torch.cat([l["_predictions"].cpu() for l in logging_outputs], dim=0),
            )
            metrics.log_concat_tensor(
                "_targets",
                torch.cat([l["_targets"].cpu() for l in logging_outputs], dim=0),
            )

            def compute_stats(meters):
                if meters["_predictions"].tensor.shape[0] < 100:
                    return 0
                stats = self.calculate_stats(
                    meters["_predictions"].tensor, meters["_targets"].tensor
                )
                return np.nanmean([stat["AP"] for stat in stats])

            metrics.log_derived("mAP", compute_stats)            


    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
