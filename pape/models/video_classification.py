import lightning
import torch
import torch.nn.functional as F
import torchmetrics

from pape.configs import Config
from pape.data_types import VideoClassificationBatch
from pape.lr_schedule import CosineDecayLRSchedule
from pape.nn.video_classifier import VideoClassifier


class VideoClassificationModel(lightning.LightningModule):
    def __init__(self, config: Config, steps_per_epoch: int = -1):
        super().__init__()

        if config.compile:
            classifier = VideoClassifier(config)
            self.classifier = torch.compile(classifier)
        else:
            self.classifier = VideoClassifier(config)

        metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy("multiclass", num_classes=config.num_classes, average="micro"),
                "precision": torchmetrics.Precision("multiclass", num_classes=config.num_classes, average="macro"),
                "recall": torchmetrics.Recall("multiclass", num_classes=config.num_classes, average="macro"),
                "f1": torchmetrics.F1Score("multiclass", num_classes=config.num_classes, average="macro"),
            },
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.optimizer = config.optimizer

        self.final_lr = self.optimizer.final_lr
        self.max_lr = self.optimizer.lr
        self.total_steps = config.epochs * steps_per_epoch
        self.warmup_steps = int(self.optimizer.warmup_ratio * self.total_steps)

    def training_step(self, batch: VideoClassificationBatch, batch_idx: int):
        batch_size = batch.videos.size(0)
        logits = self.classifier(batch)
        loss = F.cross_entropy(logits, batch.labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: VideoClassificationBatch, batch_idx: int):
        batch_size = batch.videos.size(0)
        logits = self.classifier(batch)
        loss = F.cross_entropy(logits, batch.labels)
        self.log("val/loss", loss, batch_size=batch_size)

        preds = logits.argmax(dim=-1)
        self.val_metrics.update(preds, batch.labels)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

    def test_step(self, batch: VideoClassificationBatch, batch_idx: int):
        batch_size = batch.videos.size(0)
        logits = self.classifier(batch)
        loss = F.cross_entropy(logits, batch.labels)
        self.log("test/loss", loss, batch_size=batch_size)

        preds = logits.argmax(dim=-1)
        self.test_metrics.update(preds, batch.labels)

        return loss

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
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
