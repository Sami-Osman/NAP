
import torch, copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler
from validate import validate_Res
from Utility import checkpoint


class Train_Res:
  def __init__(self, device, bert_model='albert-base-v2'):
    self.bert_model = bert_model
    self.device = device
  def train_bert(self, net, criterion, opti, lr, train_loader, val_loader, epochs, lr_scheduler, bert_version=0):
    train_losses = []
    train_acc = []
    train_accuracy = []
    val_accuracies = []
    val_losses = []
    
    total_loss = 0
    best_loss = np.Inf
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 3  # print the training loss 5 times per epoch
    
    scaler = GradScaler()

    for ep in range(epochs):
      net.train()
      running_loss = 0.0
      num_corrects = 0
      num_samples = 0
      preds = []
      
      for it, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(train_loader)):

        # Converting to cuda tensors
        seq, attn_masks, token_type_ids, labels = \
          seq.to(self.device), attn_masks.to(self.device), token_type_ids.to(self.device), labels.to(self.device, dtype = torch.float)

        with autocast():
          # Obtaining the logits from the model
          logits = net(seq, attn_masks, token_type_ids)
          # Computing training loss
          loss = criterion(logits.squeeze(-1), labels.float())
          total_loss += loss.item()

        # Backpropagating the gradients Calls backward().
        scaler.scale(loss).backward()
              
        # Optimization step
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, opti.step() is then called,
        # otherwise, opti.step() is skipped.
        scaler.step(opti)
        # Updates the scale for next iteration.
        scaler.update()
        # Adjust the learning rate based on the number of iterations.
        lr_scheduler.step()
        # Clear gradients
        opti.zero_grad()
                  

  ######################################################################
        # calculate training accuracy per iteration
        y_hat = logits.detach().cpu().numpy().flatten()
        y_true = labels.detach().cpu().numpy()
        y_hat = np.where(y_hat >= 0.5, 1, 0)
        accuracy = accuracy_score(y_hat, y_true)*100
        train_acc.append(accuracy)
      

        
        if (it + 1) % print_every == 0:  # Print training loss and accuracy information
          print()
          print("Iteration {}/{} of epoch {} complete. Loss : {} ".format(it+1, nb_iterations, ep+1, loss.item()))
          print("Training Accuracy : {} %".format(accuracy))
          

      # Compute training loss and Accuracy per Epoch
      mean_loss = total_loss/nb_iterations
      total_loss = 0
      train_losses.append(mean_loss)
      train_accuracy.append(np.mean(train_acc))

      # Compute validation loss per Epoch
      val_loss, val_accuracy = validate_Res.validate(net, criterion, val_loader, self.device)  
      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)
      print("Epoch {} complete! Validation Loss : {}".format(ep+1, val_loss))
      print("Validation Accuracy : {} %".format(val_accuracy))

      if val_loss < best_loss:
        print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
        best_loss = val_loss
        # Saving the model
        net_copy = copy.deepcopy(net)  # save a copy of the model
        checkpoint_fpath = "models/v_{}_{}.pth".format(bert_version, self.bert_model)
        checkpoint.save(checkpoint_fpath, net_copy, opti, ep, round(best_loss, 5))
        print("The model has been saved in {}".format(checkpoint_fpath))
    del loss
    torch.cuda.empty_cache()
    return train_losses, val_losses, train_accuracy, val_accuracies