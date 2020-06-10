import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class WarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    A heavily modified implementation from:
    https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """

    def __init__(
        self, optimizer, total_epochs, warmup_for, after_scheduler=None, min_lr=1e-5
    ):
        assert not isinstance(after_scheduler, ReduceLROnPlateau)
        self.warmup_for = warmup_for
        if after_scheduler is None:
            after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_for - 1, eta_min=min_lr
            )

        self.after_scheduler = after_scheduler
        self.warmup_finished = False
        self.min_lr = min_lr
        super(WarmupScheduler, self).__init__(optimizer, -1)

    def get_lr(self):
        if self.last_epoch > self.warmup_for - 1:
            if not self.warmup_finished:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.warmup_finished = True
            return [max(lr, self.min_lr) for lr in self.after_scheduler.get_lr()]

        return [
            max(base_lr * (float(self.last_epoch) / (self.warmup_for)), self.min_lr)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.warmup_finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(epoch=None)
            else:
                self.after_scheduler.step(epoch=epoch - self.warmup_for)
        else:
            super(WarmupScheduler, self).step(epoch=epoch)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    n_epochs = 30
    warmup_for = 5
    lr = 0.0003
    start_lr = 0.000005
    v = torch.zeros(10)
    optimizer = torch.optim.SGD([v], lr=lr)
    scheduler_warmup = WarmupScheduler(
        optimizer, total_epochs=n_epochs, warmup_for=warmup_for, min_lr=start_lr
    )

    epochs_lr = []
    for epoch in range(n_epochs):
        print(epoch, optimizer.param_groups[0]["lr"])
        epochs_lr.append([epoch, optimizer.param_groups[0]["lr"]])

        optimizer.step()
        scheduler_warmup.step()

    epochs_lr = np.array(epochs_lr)
    plt.figure()
    plt.plot(epochs_lr[:, 0], epochs_lr[:, 1])
    plt.grid(True)
    plt.show()
