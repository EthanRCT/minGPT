"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import get_attr, CfgNode as CN
import numpy as np


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()

        # Get attributes from config if they exist.
        checkpoint_name = get_attr(config, 'checkpoint_name', str)

        # Get attributes from model if they exist.
        self.iter_num = get_attr(model, 'iter_num', int)
        self.iter_list = get_attr(model, 'iter_list', list)
        self.checkpoint_num = get_attr(model, 'checkpoint_num', int)
        self.saved_loss = get_attr(model, 'saved_loss', list)

        # Set attributes of current state of model.
        self.itr_since_last_save = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        # Set loss
        self.loss = self.saved_loss[-1] if self.saved_loss else np.inf 

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # Get rid of the extra dimension.
            x = x.squeeze(0)
            y = y.squeeze(0)

            # Get loss state
            prev_loss = self.loss

            # forward the model
            logits, self.loss = model(x, y)
            self.curr_loss.append(self.loss.detach())

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            
            # Update iterations since last save.
            self.itr_since_last_save += 1

            # Save checkpoint if loss is lower than previous loss and we 
            # have reached the checkpoint interval.
            if self.loss <= prev_loss and self.itr_since_last_save >= config.checkpoint_iters:
                self.itr_since_last_save = 0
                
                # Save current info in a checkpoint
                checkpoint = {
                    'iter_num': self.iter_num,
                    'iter_list': self.iter_list.append(self.iter_num),
                    'checkpoint_num': self.checkpoint_num,
                    'loss': self.loss,
                    'saved_loss': self.saved_loss.append(self.loss),
                    'model_transformer': model.transformer.state_dict(),
                    'model_lm_head': model.lm_head.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }

                torch.save(checkpoint, f'checkpoints/{checkpoint_name}_{self.checkpoint_num}.pth')
                
                # Update checkpoint number
                self.checkpoint_num += 1
                
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break