import json
import tempfile
from pathlib import Path

import lightning
import torch
import wandb
from torchmetrics.detection import MeanAveragePrecision

from pape.configs import Config
from pape.data_types import ImageDetectionBatch
from pape.lr_schedule import CosineDecayLRSchedule
from pape.nn.image_detector import ImageDetector
from pape.nn.yolov10.loss import YOLOv10Loss


class ImageDetectionModel(lightning.LightningModule):
    def __init__(self, config: Config, steps_per_epoch: int = -1):
        super().__init__()

        if config.compile:
            detector = ImageDetector(config)
            self.detector = torch.compile(detector)
        else:
            self.detector = ImageDetector(config)

        self.loss_fn = YOLOv10Loss(config)

        self.height = config.height
        self.width = config.width
        self.min_conf = config.detection.min_conf
        self.mAP = MeanAveragePrecision(box_format="xyxy")
        self.metric_keys = [
            "map",
            "map_50",
            "map_75",
            "map_small",
            "map_medium",
            "map_large",
            "mar_1",
            "mar_10",
            "mar_100",
            "mar_small",
            "mar_medium",
            "mar_large",
        ]

        self.optimizer = config.optimizer

        self.final_lr = self.optimizer.final_lr
        self.max_lr = self.optimizer.lr
        self.total_steps = config.epochs * steps_per_epoch
        self.warmup_steps = int(self.optimizer.warmup_ratio * self.total_steps)

    def training_step(self, batch: ImageDetectionBatch, batch_idx: int):
        batch_size = batch.images.size(0)
        predictions = self.detector(batch)
        losses = self.compute_loss(predictions, batch)
        loss_box = losses[0]
        loss_cls = losses[1]
        loss_dfl = losses[2]
        loss = losses.sum()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/loss_box", loss_box, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_cls", loss_cls, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_dfl", loss_dfl, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.mAP.reset()

    def validation_step(self, batch: ImageDetectionBatch, batch_idx: int):
        batch_size = batch.images.size(0)
        predictions = self.detector(batch)
        losses = self.compute_loss(predictions, batch)
        loss_box = losses[0]
        loss_cls = losses[1]
        loss_dfl = losses[2]
        loss = losses.sum()
        self.log("val/loss", loss, batch_size=batch_size)
        self.log("val/loss_box", loss_box, batch_size=batch_size)
        self.log("val/loss_cls", loss_cls, batch_size=batch_size)
        self.log("val/loss_dfl", loss_dfl, batch_size=batch_size)

        self.update_mAP(predictions, batch)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.mAP.compute()
        self.log_dict({f"val/{k}": metrics[k] for k in self.metric_keys})

    def on_test_epoch_start(self):
        self.predictions = []

    def test_step(self, batch: ImageDetectionBatch, batch_idx: int):
        batch_size = batch.images.size(0)
        predictions = self.detector(batch)

        # convert predictions to format expected by COCO evaluation server
        # predictions shape: (batch_size, max_det, 6)
        # last dimension: [x1, y1, x2, y2, max_class_prob, class_index]
        predictions, _ = predictions  # unpack tuple
        mask = predictions[:, :, 4] >= self.min_conf  # filter low confidence boxes
        for batch_index in range(batch_size):
            image_predictions = predictions[batch_index, mask[batch_index]]
            scores = image_predictions[:, 4]
            labels = image_predictions[:, 5].to(torch.long)
            boxes = image_predictions[:, :4]

            image_id = batch.image_ids[batch_index]
            orig_height = batch.orig_heights[batch_index]
            orig_width = batch.orig_widths[batch_index]

            # rescale boxes to original image size
            height_scale = orig_height / self.height
            width_scale = orig_width / self.width
            x = boxes[:, 0] * width_scale
            y = boxes[:, 1] * height_scale
            w = (boxes[:, 2] - boxes[:, 0]) * width_scale
            h = (boxes[:, 3] - boxes[:, 1]) * height_scale
            boxes_xywh = torch.stack((x, y, w, h), dim=-1)

            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes_xywh = boxes_xywh.cpu().numpy()
            for score, label, box in zip(scores, labels, boxes_xywh, strict=True):
                self.predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [round(v, 2) for v in box.tolist()],
                        "score": round(score.item(), 3),
                    }
                )

    def on_test_epoch_end(self):
        logger = self.logger.experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "detections_test-dev2017_method_results.json"
            path = Path(tmpdir) / filename
            path.write_text(json.dumps(self.predictions))
            artifact = wandb.Artifact(filename, type="predictions")
            artifact.add_file(path, filename)
            logger.log_artifact(artifact)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            betas=(self.optimizer.beta1, self.optimizer.beta2),
            lr=self.optimizer.lr,
            weight_decay=self.optimizer.weight_decay,
        )

        scheduler = CosineDecayLRSchedule(
            optimizer=optimizer,
            final_lr=self.final_lr,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "strict": True,
                "name": "learning_rate",
            },
        }

    def compute_loss(self, predictions, batch: ImageDetectionBatch):
        loss, _ = self.loss_fn(
            predictions,
            {
                "batch_idx": batch.batch_idx,
                "cls": batch.labels,
                "bboxes": batch.boxes,
            },
        )
        return loss

    def update_mAP(self, predictions, batch: ImageDetectionBatch):
        # convert labels to format expected by torchmetrics
        batch_size = batch.images.size(0)
        targets = []
        for batch_index in range(batch_size):
            mask = batch.batch_idx == batch_index
            boxes = batch.boxes[mask]

            # rescale boxes to original image size
            cx = boxes[:, 0] * self.width
            cy = boxes[:, 1] * self.height
            w = boxes[:, 2] * self.width
            h = boxes[:, 3] * self.height

            # convert from cxcywh to xyxy
            half_width = w / 2
            half_height = h / 2
            x1 = cx - half_width
            y1 = cy - half_height
            x2 = cx + half_width
            y2 = cy + half_height
            boxes = torch.stack((x1, y1, x2, y2), dim=-1)

            labels = batch.labels[mask][:, 0]

            targets.append({"boxes": boxes, "labels": labels})

        # convert predictions to format expected by torchmetrics
        # predictions shape: (batch_size, max_det, 6)
        # last dimension: [x1, y1, x2, y2, max_class_prob, class_index]
        predictions, _ = predictions  # unpack tuple
        mask = predictions[:, :, 4] >= self.min_conf  # filter low confidence boxes
        preds = []
        for batch_index in range(batch_size):
            image_predictions = predictions[batch_index, mask[batch_index]]
            scores = image_predictions[:, 4]
            labels = image_predictions[:, 5].to(torch.long)
            boxes = image_predictions[:, :4]
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})

        self.mAP.update(preds, targets)
