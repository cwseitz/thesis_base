import numpy as np
import torch
from torchvision.utils import make_grid
from abc import abstractmethod
from numpy import inf
from .base import BaseTrainer
from ..utils import inf_loop, MetricTracker
from ..logger import TensorboardWriter

class SCVITrainer(BaseTrainer):
	"""
	Trainer class
	"""
	def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
				 data_loader, log_step=None, event_dir=None, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
		super().__init__(model, criterion, metric_ftns, optimizer, config)
		self.config = config
		self.device = device
		self.data_loader = data_loader
		self.event_dir = event_dir
		if len_epoch is None:
			# epoch-based training
			self.len_epoch = len(self.data_loader)
		else:
			# iteration-based training
			self.data_loader = inf_loop(data_loader)
			self.len_epoch = len_epoch
		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		if log_step is None:
			self.log_step = int(np.sqrt(data_loader.batch_size))

		self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
		self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

	def _train_epoch(self, epoch):
		"""
		Training logic for an epoch
		:param epoch: Integer, current training epoch.
		:return: A log that contains average loss and metric in this epoch.
		"""
		self.model.train()
		self.train_metrics.reset()
		for batch_idx, data in enumerate(self.data_loader):
			data = data.to(self.device, dtype=torch.float)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(data, output)
			loss.backward()
			self.optimizer.step()

			self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
			self.train_metrics.update('loss', loss.item())
			for met in self.metric_ftns:
				self.train_metrics.update(met.__name__, met(output, target))

			self.logger.debug('Train Epoch: {} {}, Rate: {}, Loss: {:.6f}'.format(
								epoch,
								self._progress(batch_idx),
								self.lr_scheduler.get_lr(),
								loss.item()))

			if batch_idx == self.len_epoch:
				break
		log = self.train_metrics.result()

		if self.do_validation:
			val_log = self._valid_epoch(epoch)
			log.update(**{'val_'+k : v for k, v in val_log.items()})

		if self.lr_scheduler is not None:
			print(self.lr_scheduler.get_lr())
			self.lr_scheduler.step()
			print(self.lr_scheduler.get_lr())
		return log


	def _progress(self, batch_idx):
		base = '[{}/{} ({:.0f}%)]'
		if hasattr(self.data_loader, 'n_samples'):
			current = batch_idx * self.data_loader.batch_size
			total = self.data_loader.n_samples
		else:
			current = batch_idx
			total = self.len_epoch
		return base.format(current, total, 100.0 * current / total)
