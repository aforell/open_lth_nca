import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """SimpleNCA model for MNIST (Py 3.9 compatible, NCHW, no device hard-coding)."""

    def __init__(self, initializer, outputs: Optional[int] = 10):
        super(Model, self).__init__()

        # MNIST defaults
        self.channel_n = 16            # total state channels (>= input_channels + num_classes recommended)
        self.input_channels = 1        # MNIST is grayscale
        self.num_classes = 10
        self.fire_rate = 0.5           # probability that a cell updates each step
        hidden_size = 128              # per-pixel MLP width (via 1x1 convs)

        if self.channel_n < self.input_channels:
            raise ValueError("channel_n must be >= input_channels.")
        # Optional: ensure enough channels to host class readout slice
        if self.channel_n < self.input_channels + self.num_classes:
            raise ValueError(
                f"channel_n ({self.channel_n}) must be >= input_channels + num_classes "
                f"({self.input_channels + self.num_classes})."
            )

        # Perception: depthwise 3x3 (requires in_channels == groups == channel_n)
        self.p0 = nn.Conv2d(self.channel_n, self.channel_n, kernel_size=3, stride=1,
                            padding=1, groups=self.channel_n, padding_mode="reflect")
        self.p1 = nn.Conv2d(self.channel_n, self.channel_n, kernel_size=3, stride=1,
                            padding=1, groups=self.channel_n, padding_mode="reflect")

        # Per-pixel MLP via 1x1 convs: (3C -> hidden -> C)
        self.fc0 = nn.Conv2d(self.channel_n * 3, hidden_size, kernel_size=1, bias=True)
        self.fc1 = nn.Conv2d(hidden_size, self.channel_n, kernel_size=1, bias=False)

        # Apply framework initializer first (honors --model_init), then enforce zero init on fc1.
        if initializer is not None:
            initializer(self)
        with torch.no_grad():
            self.fc1.weight.zero_()

        # Criterion
        self._criterion = nn.CrossEntropyLoss()

    # ----- NCA internals -----

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        # x: N, C=channel_n, H, W
        z1 = self.p0(x)
        z2 = self.p1(x)
        return torch.cat((x, z1, z2), dim=1)  # N, 3C, H, W

    def update(self, x: torch.Tensor, fire_rate: Optional[float]):
        dx = self.perceive(x)             # N, 3C, H, W
        dx = F.relu(self.fc0(dx))         # N, hidden, H, W
        dx = self.fc1(dx)                 # N, C, H, W

        fr = self.fire_rate if fire_rate is None else fire_rate
        # Create mask on same device, matching dtype
        mask = (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=dx.dtype) < fr).to(dx.dtype)
        dx = dx * mask
        return x + dx

    def _build_state_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept (N,1,H,W) raw MNIST or (N,channel_n,H,W) full state; return full state.
        """
        if x.dim() != 4:
            raise ValueError("Expected 4D NCHW tensor, got {}".format(tuple(x.shape)))

        N, C, H, W = x.shape
        if C == self.input_channels:
            if self.channel_n == self.input_channels:
                return x
            hidden = torch.zeros(
                N, self.channel_n - self.input_channels, H, W,
                device=x.device, dtype=x.dtype
            )
            return torch.cat([x, hidden], dim=1)
        elif C == self.channel_n:
            return x
        else:
            raise ValueError(
                "Expected channels {} (raw) or {} (state), got {}."
                .format(self.input_channels, self.channel_n, C)
            )

    def forward(self, x: torch.Tensor, steps: int = 32, fire_rate: Optional[float] = None):
        # Ensure full state shape for depthwise convs.
        state = self._build_state_if_needed(x)  # N, channel_n, H, W

        for _ in range(steps):
            x2 = self.update(state, fire_rate).clone()
            # Clamp first input_channels; update the rest.
            state = torch.cat((state[:, :self.input_channels], x2[:, self.input_channels:]), dim=1)

        # Global average pooling and class readout from a channel slice.
        pooled = state.mean(dim=(2, 3))  # N, C
        logits = pooled[:, self.input_channels:self.input_channels + self.num_classes]  # N, num_classes
        return logits

    # ----- Framework hooks -----

    @property
    def output_layer_names(self):
        # Output mapping lives in fc1 -> channels used as logits slice.
        return ['fc1.weight']

    @staticmethod
    def is_valid_model_name(model_name: str) -> bool:
        return model_name.startswith('mnist_simplenca')

    @staticmethod
    def get_model_from_name(model_name: str, outputs: int, initializer):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))
        return Model(initializer, outputs)

    @property
    def loss_criterion(self) -> torch.nn.Module:
        return self._criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_simplenca',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=0.1,
            training_steps='5ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc1.weight',  # don’t prune output mapping
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
