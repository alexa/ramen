# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from torch.optim import Adam


class AdamInverseSquareRootWithWarmup(Adam):
    """
    Borrowing from fairseq: inverse_square_root_schedule.py
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::
      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(params, lr=warmup_init_lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.warmup_updates = warmup_updates
        # initial learning rate
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr

        # linearly warmup for the first warmup_updates
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates**0.5

        # total number of updates
        for group in self.param_groups:
            group['num_updates'] = 0

    def get_lr(self):
        """return current learning rate"""
        for group in self.param_groups:
            return self.lr_update(group['num_updates'])

    def lr_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates*self.lr_step
        else:
            return self.decay_factor * num_updates**-0.5

    def step(self, closure=None):
        super().step(closure)
        # update learning rate
        for group in self.param_groups:
            group['num_updates'] += 1
            group['lr'] = self.lr_update(group['num_updates'])
