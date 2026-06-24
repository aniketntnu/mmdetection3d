"""Hook that detects NaN/Inf/exploding loss and skips the gradient update."""
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class NanLossSkipHook(Hook):
    """Skip optimizer step when loss is NaN, Inf, or above a threshold.

    Zeroes out all losses in-place so backward produces zero gradients.
    The optimizer step still runs but makes no effective parameter update.

    Args:
        max_loss (float): Skip if total loss exceeds this. Default 1000.
        log_interval (int): Print warning every N skipped steps. Default 1.
    """

    def __init__(self, max_loss: float = 1000.0, log_interval: int = 1):
        self.max_loss = max_loss
        self.log_interval = log_interval
        self._skip_count = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        pass  # we need before-backward hook, handled below

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        pass

    # mmengine calls this after model forward, before backward
    def after_forward(self, runner, outputs):
        if outputs is None:
            return

        # outputs is a dict of losses
        total = sum(v for v in outputs.values()
                    if isinstance(v, torch.Tensor) and v.numel() == 1)

        bad = (not torch.isfinite(total)) or (total.item() > self.max_loss)
        if bad:
            self._skip_count += 1
            if self._skip_count % self.log_interval == 0:
                runner.logger.warning(
                    f'NanLossSkipHook: skipping step (loss={total.item():.2f}, '
                    f'total skipped={self._skip_count})')
            # Zero out all loss tensors so backward = no-op
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = v * 0.0
