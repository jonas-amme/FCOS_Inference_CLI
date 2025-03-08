from typing import Dict, List, Tuple, Union
import lightning.pytorch as pl
import torch
from torch import nn 
import torch.nn.functional as F

from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, ChainedScheduler, OneCycleLR, CyclicLR, ExponentialLR, CosineAnnealingLR




class BaseDetectionModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            batch_size: int = 16,
            lr: float = 0.0001,
            optimizer: str = 'AdamW',
            scheduler: Union[str, None] = None):
        super().__init__()

        # save hparams
        self.save_hyperparameters(ignore=['model'])

        # store model 
        self.model = model 

        # set up metric 
        self.metric = MeanAveragePrecision(
            iou_thresholds=[0.5],
            class_metrics=True, 
            iou_type='bbox',
            box_format='xyxy',
            max_detection_thresholds=[1, 10, 500],
            backend='faster_coco_eval'
            )


    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor = None) -> List[Dict[str, torch.Tensor]]:
        """Forward pass during inference.

        Returns post-processed predictions as a list of dictionaries.
        """
        return self.model(x, y)
    


    def training_step(self, batch):
        """Training step."""

        # forward pass
        images, targets = batch
        loss_dict = self(images, targets)

        # compute loss 
        loss = sum([loss for loss in loss_dict.values()])

        # logging
        self.log('train/loss', loss, on_epoch=True, batch_size=self.hparams.batch_size)
        self.log_dict(
            {f'train/{name}': value for name, value in loss_dict.items()}, 
            on_epoch=True, 
            batch_size=self.hparams.batch_size
            )

        return loss 
    

    def validation_step(self, batch):
        """Validation step."""

        # forward pass 
        images, targets = batch
        preds = self(images, targets)

        # update metrics
        self.metric.update(preds, targets)



    def on_validation_epoch_end(self):
        """Compute metrics at end of epoch."""
        # compute metrics 
        metrics = self.metric.compute()

        # collect average metrics 
        ap = metrics['map_50']
        ar = metrics['mar_100']
        self.log('val/map', ap, prog_bar=True)
        self.log('val/mar', ar, prog_bar=True)

        # collect class specific metrics 
        map_per_class = metrics['map_per_class']
        mar_per_class = metrics['mar_100_per_class']
        if torch.numel(map_per_class) > 1:
            self.log_dict({f'val/ap_{str(k)}': v for k, v in enumerate(map_per_class)})
            self.log_dict({f'val/ar_{str(k)}': v for k, v in enumerate(mar_per_class)})

    	# reset metric 
        self.metric.reset()


    def configure_optimizers(self):
        """Configures optimizers and learning rate schedulers"""
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(trainable_parameters, lr=self.hparams.lr)
        elif self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.hparams.lr)
        else:
            raise NotImplementedError

        if self.hparams.scheduler == 'CyclicLR':
            scheduler = {
            'scheduler': CyclicLR(
                optimizer, 
                base_lr=1e-5,
                max_lr=1e-3,
                step_size_up=1000
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
            return [optimizer], [scheduler]
        
        elif self.hparams.scheduler == 'ExponentialLR':
            scheduler = {
            'scheduler': ExponentialLR(optimizer, gamma=0.99),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
        }
            return [optimizer], [scheduler]

        elif self.hparams.scheduler == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr, 
                total_steps=self.trainer.estimated_stepping_batches, 
                pct_start=0.3, 
                div_factor=10)
            return [optimizer], [scheduler]
        
        elif self.hparams.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-7
            )
            return [optimizer], [scheduler]

        elif self.hparams.scheduler == None:
            return [optimizer]
        else:
            raise NotImplementedError