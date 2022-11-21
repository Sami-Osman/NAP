import torch
import numpy as np
import random, os
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

class checkpoint:
    def load(checkpoint_fpath, model, optimizer):
      """
      checkpoint_fpath: path to load from checkpoint
      model: model that we want to load checkpoint parameters into
      optimizer: optimizer defined in training
      epoch:  optimal epoch noticed during training
      valid_loss_min: minimum valid_loss saved
      """
      # load check point
      checkpoint = torch.load(checkpoint_fpath)
      # initialize state_dict from checkpoint to model
      model.load_state_dict(checkpoint['state_dict'])
      # initialize optimizer from checkpoint to optimizer
      #optimizer.load_state_dict(checkpoint['optimizer'])
      # initialize epoch from checkpoint to epoch
      epoch = checkpoint['epoch']
      # initialize valid_loss_min from checkpoint to valid_loss_min
      valid_loss_min = checkpoint['valid_loss_min']
      # return model, optimizer, epoch value, min validation loss
      #NOTE: i am returning the model for my purpose/ even though in my project training phase i saved the epoch, optimizer and best loss.
      return model

    def save(checkpoint_fpath, model, optimizer, epoch, valid_loss):
      """
      checkpoint_fpath: path to save checkpoint
      model: model that we want to save checkpoint parameters from
      optimizer: optimizer parameters at the time of min valid_loss
      epoch:  optimal epoch at the time of min valid_loss
      valid_loss_min: minimum valid_loss during training
      """
      f_path = checkpoint_fpath
      # create checkpoint variable and add important data
      state = {
              'epoch': epoch + 1,
              'valid_loss_min': valid_loss,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()
              }
      # save checkpoint data to the path given, checkpoint_path
      torch.save(state, f_path)

    def set_seed(seed):
      """ Set all seeds to make results reproducible """
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      np.random.seed(seed)
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)

    def get_my_lr_scheduler(opti, train_loader, epochs):
      num_warmup_steps = 0 # The number of steps for the warmup phase.
      num_training_steps = len(train_loader) * epochs 
      lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
      return lr_scheduler
    
    def show_loss(train_losses, val_losses):
      plt.plot(train_losses, label="Mean Training Loss")
      plt.plot(val_losses, label="Mean Validation Loss")
      plt.xlabel('No. of Epochs')
      plt.ylabel('Mean Loss')
      plt.legend(frameon=True)
      plt.show()
    def show_accuracy(train_acc, val_acc):
      plt.plot(train_acc, label="Training Accuracy")
      plt.plot(val_acc, label="Validation Accuracy")
      plt.xlabel('No. of Epochs')
      plt.ylabel('Accuracy & Loss')
      plt.ylim([0, 100]) # limit y axis between 0 and 1
      plt.legend(frameon=False)
      plt.show()

