import numpy as np
import torch
import os
import torch.nn.functional as F

import torch.distributed as dist 
from utils.parallel import get_dist_info
from utils.display import display_metrics_dict
from utils.evaluation import AverageMeter, accuracy, fuzziness


class NCSSLRunner:
    """The runner for Checking Neural Collapse for each layer of neural network.
       Actually, it just works as a classification runner not specific for neural collapse yet. 
       (The analysis will in analysis.py)
        Args:
            
    """
    def __init__(self, 
                 models,
                 optims,
                 losses,
                 logger,
                 work_dir,
                 n_views=2,
                 temperature=0.07,
                 max_epochs=200,
                 print_every=100,
                 val_every=20000,
                 save_every=20000,
                 resume_latest=True,
                 meta=None):
        self.models = models
        self.optims = optims
        self.losses = losses
        self.logger = logger
        self.work_dir = work_dir
        self.n_views = n_views
        self.temperature = temperature
        self._model_name = 'NCRunner'
        self._rank, self._world_size = get_dist_info()
        self._epoch = 1
        self._iter = 1
        self._inner_iter = 1
        self._max_epochs = max_epochs
        self._max_iters = max_epochs
        self._print_every = print_every
        self._val_every = val_every
        self._save_every = save_every
        self.meta = meta if meta is not None else {}

    @property
    def model_name(self) -> str:
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self) -> int:
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self) -> int:
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def epoch(self) -> int:
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self) -> int:
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self) -> int:
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def initialize_metrics_dict(self):
        metrics_names = ['loss', 'acc1', 'acc5']
        metrics_dict = {metric:AverageMeter() for metric in metrics_names}

        return metrics_names, metrics_dict

    def display_info(self, tag, epoch_i, iter_i, metrics_names, metrics_dict):
        metrics_message = display_metrics_dict(metrics_names, metrics_dict)
        epoch_message = f'Epoch[{epoch_i+1}/{self._max_epochs}], Iter[{iter_i+1}/{self._each_iters}]'
        cur_lrs = self.current_lr()
        cur_lr = list(cur_lrs.values())[0][0]
        optim_message = f'cur_lr:{cur_lr:.7f}'
        whole_message = tag + ' ' + epoch_message + ' (' + optim_message + ') :' + metrics_message
        self.logger.info(whole_message)

    def info_nce_loss(self, features):
        # create labels accroding to the batch_size
        batch_size = features.size(0) // 2
        device = features.device
        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
        
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        
        # eliminate the one-one pairs
        mask = torch.eye(labels.size(0), dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.size(0), -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.size(0), -1)

        # select the positives 
        positives = similarity_matrix[labels.bool()].view(labels.size(0), -1)
        # select the negatives
        negatives = similarity_matrix[~labels.bool()].view(labels.size(0), -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

        logits = logits / self.temperature
        return logits, labels

    def forward_loss(self, data_batch, return_input=False):
        # get task i model
        ssl_model = self.models['ssl']
        device = next(ssl_model.parameters()).device
        # get data    
        in_data, labels = data_batch # [dterm.to(device) for dterm in data_batch]
        in_data = [dterm.to(device) for dterm in in_data]
        in_data = torch.cat(in_data, dim=0)
        labels = labels.to(device)

        # 2 * batch_size
        features = ssl_model(in_data)

        ssl_logits, ssl_labels = self.info_nce_loss(features)
        loss_val = self.losses['cls'](ssl_logits, ssl_labels)

        if return_input:
            return in_data, labels, ssl_logits, ssl_labels, loss_val
        return ssl_logits, ssl_labels, loss_val

    def train(self, data_names, train_data_loaders, val_data_loaders):
        # Update the max_iters
        
        train_data_loader = train_data_loaders[data_names[0]]
        self._each_iters = len(train_data_loader)
        self._max_iters = self._max_epochs * self._each_iters

        # get the sampled data
        
        # Info 
        metrics_names, metrics_dict= self.initialize_metrics_dict()
        whole_iter_i = 1
        for epoch_i in range(self._max_epochs):
            for iter_i, data_batch in enumerate(train_data_loader):
                # zero cls optimizer grad
                self.optims['ssl'].zero_grad()

                # forward loss
                ssl_logits, ssl_labels, loss_val = self.forward_loss(data_batch, return_input=False)
                
                loss_val.backward()

                self.optims['ssl'].step()
                
                # Display and Save
                if self._rank == 0:
                    if whole_iter_i % self._print_every == 0:
                        with torch.no_grad():

                            # update metrics
                            batch_size = ssl_logits.size(0)
                            accs = accuracy(ssl_logits, ssl_labels, topk=(1,5))
                            metrics_dict['loss'].update(loss_val.detach().cpu().item(), batch_size)
                            metrics_dict['acc1'].update(accs[1], batch_size)
                            metrics_dict['acc5'].update(accs[5], batch_size)
                            metrics_dict_val = {mn:metrics_dict[mn].val for mn in metrics_names}

                            self.display_info('Train', epoch_i, iter_i, metrics_names, metrics_dict_val)
                        
                    if whole_iter_i % self._val_every == 0:
                        self.val(data_names, val_data_loaders, epoch_i, iter_i)
                        
                    if whole_iter_i % self._save_every == 0:
                        # save checkpoints
                        self.save_checkpoint(self.work_dir, f'iter_{whole_iter_i}.pth') 
                        
                whole_iter_i += 1
                self._iter = whole_iter_i
            self._epoch = epoch_i

    def val(self, data_names, val_data_loaders, epoch_i=-1, iter_i=-1):
        val_data_loader = val_data_loaders[data_names[0]]
        self.models['ssl'].eval()
        
        metrics_names, metrics_dict = self.initialize_metrics_dict()

        with torch.no_grad():
            for iter_i, data_batch in enumerate(val_data_loader):
                # forward loss
                ssl_logits, ssl_labels, loss_val = self.forward_loss(data_batch, return_input=False)
                accs = accuracy(ssl_logits, ssl_labels, topk=(1,5))
                batch_size = ssl_logits.size(0)
                metrics_dict['loss'].update(loss_val.detach().cpu().item(), batch_size)
                metrics_dict['acc1'].update(accs[1], batch_size)
                metrics_dict['acc5'].update(accs[5], batch_size)
                    
            metrics_dict_avg = {m:metrics_dict[m].avg for m in metrics_dict}
            
            self.display_info('Val', epoch_i, iter_i, metrics_names, metrics_dict_avg)
        self.models['ssl'].train()

    def run(self, data_names, 
            train_data_loaders, val_data_loaders,
            workflow, **kwargs):
        """
        workflow is not implemented
        """
        self.train(data_names, train_data_loaders, val_data_loaders)

        self.val(data_names, val_data_loaders, self._max_epochs, 0)

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        
        if isinstance(self.optims, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optims, dict):
            lr = dict()
            for name, optim in self.optims.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optims is None:
            raise RuntimeError(
                'momentum is not applicable because optims does not exist.')
        elif isinstance(self.optims, torch.optim.optims):
            momentums = _get_momentum(self.optims)
        elif isinstance(self.optims, dict):
            momentums = dict()
            for name, optim in self.optims.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer = False,
                        meta = None,) -> None:
        
        os.makedirs(os.path.join(out_dir, 'chckpoints'), exist_ok=True)
        save_filename = os.path.join(out_dir, 'chckpoints', filename_tmpl)
        
        save_dict = {'models':{}, 'optims':{}, 'meta':{}}
        # models
        for model in self.models:
            if isinstance(self.models[model], torch.Tensor):
                save_dict['models'][model] = self.models[model].cpu()
            else:
                save_dict['models'][model] = self.models[model].state_dict()

        # optimizer
        if save_optimizer:
            for optim in self.optims:
                save_dict['optims'][optim] = self.optims[optim].state_dict()

        # meta
        save_dict['meta'] = meta if meta is not None else self.meta
        save_dict['meta'].update(
            {
                'epoch':self._epoch,
                'iter':self._iter
            })
        torch.save(save_dict, save_filename)

    def load_checkpoint(
        self,
        filename,
        map_location = 'cpu',
        strict = False,
        revise_keys = [(r'^module.', '')],
    ):
        checkpoint = torch.load(filename, map_location=map_location)
        for model in self.models:
            if isinstance(self.models[model], torch.Tensor):
                self.models[model] = checkpoint['models'][model].to(self.models[model].device)
            else:
                self.models[model].load_state_dict(checkpoint['models'][model])

        return checkpoint

    def resume(self,
               checkpoint,
               resume_optimizer = False,
               map_location = 'default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']

        # resume meta information meta
        self.meta = checkpoint['meta']


        # optimizer
        if 'optims' in checkpoint and resume_optimizer:
            for optim in self.optims:
                self.optims[optim].load_state_dict(checkpoint['optims'][optim])

        self.logger.info('resumed epoch %d, iter %d', self._epoch, self._iter)
