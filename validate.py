
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class validate_Res:
  def validate(net, criterion, dataloader, device):
    net.eval()

    mean_loss = 0
    count = 0
    val_acc = []

    with torch.no_grad():
        for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
            seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1
            # calculate validation accuracy
            y_hat = logits.detach().cpu().numpy().flatten()
            y_true = labels.detach().cpu().numpy()
            y_hat = np.where(y_hat >= 0.5, 1, 0)
            accuracy = accuracy_score(y_hat, y_true)*100
            val_acc.append(accuracy)
    val_accuracy = np.mean(val_acc)

    return (mean_loss / count), val_accuracy