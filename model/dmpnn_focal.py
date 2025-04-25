import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import numpy as np
from torchmetrics.classification import MultilabelAUROC
import pytorch_lightning as pl



def compute_best_thresholds(y_true, y_probs, steps=20):
    thresholds = np.linspace(0.1, 0.9, steps)
    best_thresholds = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i])
        y_t = y_true[mask, i]
        y_p = y_probs[mask, i]

        best_f1 = 0
        best_thresh = 0.5
        for t in thresholds:
            pred = (y_p >= t).astype(int)
            f1 = f1_score(y_t, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_thresholds.append(best_thresh)
    print(f"Best thresholds: {best_thresholds}")
    return best_thresholds



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        # self.alpha = alpha
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # inputs, targets: (batch_size, num_tasks)
        mask = ~torch.isnan(targets)
        targets = targets.float().clone()
        targets[~mask] = 0  # avoid NaNs in loss

        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce_loss = self.bce(inputs, targets)
        loss = alpha_factor * focal_factor * bce_loss

        loss = loss * mask.float()
        return loss.sum() / mask.float().sum()
    
    



class MPNNModel_FocalLoss(pl.LightningModule):

    def __init__(self, mp, agg, ffn, batch_norm, metric_list, alpha_tensor, gamma):
        super().__init__()
        self.mp = mp
        self.agg = agg
        self.ffn = ffn
        self.batch_norm = batch_norm
        self.metric_list = metric_list
        self.train_auc = MultilabelAUROC(num_labels=12, average='macro')  # for 12 binary tasks
        self.train_outputs = []

        # ✅ Register alpha for automatic device transfer
        self.loss_fn = FocalLoss(alpha=alpha_tensor, gamma=gamma)



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    
    
    def forward(self, batch):
        x = self.mp(batch.bmg)
        x = self.agg(x, batch.bmg.batch)
        x = self.ffn(x)
        return x


    def training_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        targets = batch.Y
        mask = ~torch.isnan(targets)
        # task_mask = mask.float()
        targets = torch.nan_to_num(targets, nan=0.0)
        # loss = F.binary_cross_entropy_with_logits(
        #     logits, targets, weight=task_mask, pos_weight=self.pos_weight.to(logits.device)
        # )
        loss = self.loss_fn(logits, targets)
        self.log("train_loss", loss)
        
        # Save predictions/targets for AUC computation
        self.train_outputs.append({
            "probs": probs.detach(),
            "targets": targets.detach(),
            "mask": mask.detach()
        })
        return loss
    
    
    def on_train_epoch_end(self):
        probs = torch.cat([o["probs"] for o in self.train_outputs], dim=0)
        targets = torch.cat([o["targets"] for o in self.train_outputs], dim=0)
        mask = torch.cat([o["mask"] for o in self.train_outputs], dim=0)

        valid_rows = mask.any(dim=1)
        auc = self.train_auc(probs[valid_rows], targets[valid_rows].int())

        self.log("train_auc", auc, prog_bar=True)
        self.train_outputs.clear()  # clear for next epoch
    
    
            
    def on_validation_epoch_start(self):
        self.validation_outputs = []
        
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        targets = batch.Y
        self.validation_outputs.append({"probs": probs.detach(), "targets": targets.detach()})
        

    def on_validation_epoch_end(self):
        
        probs = torch.cat([x["probs"] for x in self.validation_outputs], dim=0).cpu().numpy()
        targets = torch.cat([x["targets"] for x in self.validation_outputs], dim=0).cpu().numpy()

        aucs = []
        best_thresholds = compute_best_thresholds(targets, probs)
        for i in range(targets.shape[1]):
            mask = ~np.isnan(targets[:, i])
            y_t = targets[mask, i]
            y_p = probs[mask, i]
            # pred = (y_p >= best_thresholds[i]).astype(int)


            try:
                auc = roc_auc_score(y_t, y_p)
                aucs.append(auc)
            except ValueError:
                continue  # skip task if not computable
        
        # 🔹 Log overall mean AUC
        if len(aucs) > 0:
            mean_auc = np.mean(aucs)
            self.log("val_mean_auc", mean_auc, prog_bar=True, on_epoch=True)

        # self.log("Best thresholds", best_thresholds, prog_bar=True, on_epoch=True)
        print('Best thresholds:', best_thresholds)
                

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        targets = batch.Y
        self.test_outputs.append({"probs": probs.detach(), "targets": targets.detach()})

    def on_test_epoch_end(self):
        # Concatenate all test batches
        probs = torch.cat([x["probs"] for x in self.test_outputs], dim=0).cpu().numpy()
        targets = torch.cat([x["targets"] for x in self.test_outputs], dim=0).cpu().numpy()
        n_tasks = targets.shape[1]

        aucs = []
        for i in range(n_tasks):
            mask = ~np.isnan(targets[:, i])
            y_true = targets[mask, i]
            y_prob = probs[mask, i]
            # y_pred = (y_prob >= 0.5).astype(int)

            # # 🔹 Log per-task recall
            # if len(np.unique(y_true)) <= 1:
            #     self.log(f"test_recall_task_{i}", 0.0, prog_bar=False, on_epoch=True)
            # else:
            #     recall = recall_score(y_true, y_pred, zero_division=0)
            #     self.log(f"test_recall_task_{i}", recall, prog_bar=False, on_epoch=True)

            # 🔹 AUC
            try:
                auc = roc_auc_score(y_true, y_prob)
                aucs.append(auc)
            except ValueError:
                continue  # skip task if AUC cannot be computed

        # 🔹 Log overall mean AUC
        if len(aucs) > 0:
            mean_auc = np.mean(aucs)
            self.log("test_mean_auc", mean_auc, prog_bar=True, on_epoch=True)

        self.test_outputs.clear()
        
        
        
    def predict_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        return {"probs": probs.detach(), "targets": batch.Y.detach()}
    
    


def compute_best_thresholds(y_true, y_probs, steps=20):
    thresholds = np.linspace(0.1, 0.9, steps)
    best_thresholds = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i])
        y_t = y_true[mask, i]
        y_p = y_probs[mask, i]

        best_f1 = 0
        best_thresh = 0.5
        for t in thresholds:
            pred = (y_p >= t).astype(int)
            f1 = f1_score(y_t, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        best_thresholds.append(best_thresh)
    print(f"Best thresholds: {best_thresholds}")
    return best_thresholds