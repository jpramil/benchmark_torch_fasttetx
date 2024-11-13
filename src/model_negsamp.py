import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch import nn
import pytorch_lightning as pl
import random
from typing import List

class FastTextModel_negsamp(nn.Module):
    """
    FastText Pytorch Model with Negative Sampling.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        num_classes: int,
        categorical_vocabulary_sizes: List[int],
        padding_idx: int = 0,
        sparse: bool = True,
    ):
        """
        Constructor for the FastTextModel class.

        Args:
            embedding_dim (int): Dimension of the text embedding space.
            vocab_size (int): Number of rows in the embedding matrix.
            num_classes (int): Number of classes.
            categorical_vocabulary_sizes (List[int]): List of number of modalities for categorical features.
            padding_idx (int, optional): Padding index for embeddings. Defaults to 0.
            sparse (bool): Indicates if the Embedding layer is sparse.
        """
        super(FastTextModel_negsamp, self).__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        # Embedding layers
        self.embeddings = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=vocab_size,
            padding_idx=padding_idx,
            sparse=sparse,
        )
        self.categorical_embeddings = {}
        for var_idx, vocab_size in enumerate(categorical_vocabulary_sizes):
            emb = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
            self.categorical_embeddings[var_idx] = emb
            setattr(self, f"emb_{var_idx}", emb)

        # Class embedding layer for negative sampling
        self.class_embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_classes)

    def forward(self, inputs: List[torch.LongTensor]) -> torch.Tensor:
        """
        Forward method for the model.

        Args:
            inputs (List[torch.LongTensor]): Input tensors.

        Returns:
            torch.Tensor: Output embeddings.
        """
        x_1 = self.embeddings(inputs[0])
        x_cat = [self.categorical_embeddings[i](inputs[i + 1]) for i in range(len(self.categorical_embeddings))]

        # Mean of token embeddings
        non_zero_tokens = x_1.sum(-1) != 0
        non_zero_tokens = non_zero_tokens.sum(-1)
        x_1 = x_1.sum(dim=-2) / non_zero_tokens.unsqueeze(-1)
        x_1 = torch.nan_to_num(x_1)

        x_in = x_1 + sum(x_cat) if x_cat else x_1
        return x_in

class FastTextModule_negsamp(pl.LightningModule):
    """
    Pytorch Lightning Module for FastTextModel with Negative Sampling.
    """

    def __init__(
        self,
        model: FastTextModel_negsamp,
        loss,
        optimizer,
        optimizer_params,
        scheduler,
        scheduler_params,
        scheduler_interval,
        num_negative_samples: int,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.num_negative_samples = num_negative_samples
        self.accuracy_fn = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.scheduler_interval = scheduler_interval

    def _sample_negatives(self, targets):
        """
        Randomly sample negative classes for each target class.

        Args:
            targets (torch.Tensor): Tensor of target class indices.

        Returns:
            torch.Tensor: Indices of sampled negative classes.
        """
        # Convert the set to a sorted list to ensure it is a sequence
        all_classes = sorted(range(self.model.num_classes))
        negative_samples = []
        for target in targets:
            negatives = random.sample([cls for cls in all_classes if cls != target.item()], self.num_negative_samples)
            negative_samples.extend(negatives)
        return torch.tensor(negative_samples, device=self.device)

    def _negative_sampling_loss(self, embeddings, targets):
        """
        Calculate the loss using negative sampling.

        Args:
            embeddings (torch.Tensor): Embeddings of inputs.
            targets (torch.Tensor): True class indices for positive samples.

        Returns:
            torch.Tensor: Negative sampling loss.
        """
        positive_logits = torch.matmul(embeddings, self.model.class_embeddings(targets).T)
        positive_loss = F.logsigmoid(positive_logits).mean()

        # Sample negative classes
        negative_targets = self._sample_negatives(targets)
        negative_logits = torch.matmul(embeddings, self.model.class_embeddings(negative_targets).T)
        negative_loss = F.logsigmoid(-negative_logits).mean()

        return -(positive_loss + negative_loss)

    def training_step(self, batch: List[torch.LongTensor], batch_idx: int) -> torch.Tensor:
        """
        Training step with negative sampling.

        Args:
            batch (List[torch.LongTensor]): Training batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        embeddings = self.model(inputs)
        loss = self._negative_sampling_loss(embeddings, targets)
        self.log("train_loss", loss, on_epoch=True)

        accuracy = self.accuracy_fn(embeddings.argmax(dim=1), targets)
        self.log('train_accuracy', accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch: List[torch.LongTensor], batch_idx: int):
        """
        Validation step with negative sampling.

        Args:
            batch (List[torch.LongTensor]): Validation batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        inputs, targets = batch[:-1], batch[-1]
        embeddings = self.model(inputs)
        loss = self._negative_sampling_loss(embeddings, targets)
        self.log("validation_loss", loss, on_epoch=True)

        accuracy = self.accuracy_fn(embeddings.argmax(dim=1), targets)
        self.log('validation_accuracy', accuracy, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        scheduler = self.scheduler(optimizer, **self.scheduler_params)
        scheduler = {
            "scheduler": scheduler,
            "monitor": "validation_loss",
            "interval": self.scheduler_interval,
        }
        return [optimizer], [scheduler]
