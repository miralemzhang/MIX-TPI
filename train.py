import os
import torch
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import trange

from pytorchtools import EarlyStopping

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        logger,
        save_dir,
        device="cuda",
        save_model_name="model.pt",
        is_progress_bar=True,
        early_stopping=True,
        patience=10,
    ):

        self.early_stopping = early_stopping
        self.patience = patience
        self.device = device
        self.model = model.to(self.device)

        self.criterion = criterion
        self.optimizer = optimizer

        self.save_dir = save_dir
        self.save_model_path = os.path.join(self.save_dir, save_model_name)
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.logger.info("Training Device: {}".format(self.device))

    def train(self, train_loader, test_loader, epochs=100):
        if self.early_stopping:
            early_stopping = EarlyStopping(patience=self.patience, trace_func=self.logger.info, path=self.save_model_path, is_save_model=False)

        best_train_loss = 0.0
        best_train_auc = 0.0
        best_train_aupr = 0.0
        best_test_auc = 0.0
        best_test_loss = 0.0
        best_test_aupr = 0.0
        best_epoch = -1
        for epoch in range(epochs):
            self.model.train()

            train_loss, train_auc, train_aupr = self._train_epoch(train_loader, epoch)
            test_loss, test_auc, test_aupr = self._validate_epoch(test_loader)

            self.logger.info(
                "Epoch: {} Average training loss : {:.6f}, train_auc: {:.6f}, train_aupr: {:.6f}, test_auc: {:.6f}, test_aupr : {:.6f}".format(
                    epoch + 1, train_loss, train_auc, train_aupr, test_auc, test_aupr
                )
            )

            if test_auc > best_test_auc:
                self.logger.info(
                    "best auc: {%.4f} -> {%.4f} at epoch {%d}" % (best_test_auc, test_auc, epoch)
                )
                torch.save(self.model.state_dict(), self.save_model_path)
                best_epoch = epoch
                best_train_auc = train_auc
                best_train_aupr = train_aupr
                best_train_loss = train_loss
                best_test_auc = test_auc
                best_test_aupr = test_aupr
                best_test_loss = test_loss

            if self.early_stopping:
                early_stopping(-test_auc, self.model)
                if early_stopping.early_stop:
                    self.logger.info("Early stopping")
                    break

        self.logger.info(
            "\n best_valid_epoch {%d}: train loss: {%.4f}, test loss: {%.4f} \n train auc: {%.4f}, test auc: {%.4f} \n train aupr: {%.4f}, test aupr: {%.4f}" % (best_epoch, best_train_loss, best_test_loss, best_train_auc, best_test_auc, best_train_aupr, best_test_aupr)
        )

        return best_train_loss, best_test_loss, best_train_auc, best_test_auc, best_train_aupr, best_test_aupr


    def _train_epoch(self, train_loader, epoch):
        epoch_loss = 0.0
        epoch_targets = []
        epoch_preds = []

        kwargs = dict(
            desc="Epoch {}".format(epoch + 1),
            leave=False,
            disable=not self.is_progress_bar,
        )

        with trange(train_loader.batch_count(), **kwargs) as t:
            for idx, data in enumerate(train_loader):

                self.optimizer.zero_grad()
                iter_output, repre_loss = self.model(data)
                iter_target = data["labels"].to(self.device)
                task_loss = self.criterion(iter_output.squeeze(), iter_target.squeeze())
                total_loss = task_loss + repre_loss
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()
                epoch_targets.extend(iter_target)
                epoch_preds.extend(iter_output)
                t.update()

        mean_epoch_loss = epoch_loss / len(train_loader)
        epoch_train_auc = roc_auc_score(
            torch.tensor(epoch_targets), torch.tensor(epoch_preds)
        )
        precision, recall, _ = precision_recall_curve(torch.tensor(epoch_targets), torch.tensor(epoch_preds))
        epoch_train_aupr = auc(recall, precision)

        return mean_epoch_loss, epoch_train_auc, epoch_train_aupr

    def _validate_epoch(self, valid_loader):
        self.model.eval()

        with torch.no_grad():
            val_loss = 0
            val_targets = []
            val_preds = []

            for data in valid_loader:
                torch.cuda.empty_cache()
                output, repre_loss = self.model(data)
                # embed = self.model.get_embedding(data)
                val_preds.extend(output)
                val_targets.extend(data["labels"])
                target = data["labels"].to(self.device)

                task_loss = self.criterion(output.squeeze(), target.squeeze())
                total_loss = task_loss + repre_loss
                val_loss += total_loss.item()

        mean_loss = val_loss / len(valid_loader)
        val_auc = roc_auc_score(torch.tensor(val_targets), torch.tensor(val_preds))
        precision, recall, _ = precision_recall_curve(torch.tensor(val_targets), torch.tensor(val_preds))
        val_aupr = auc(recall, precision)

        return mean_loss, val_auc, val_aupr