import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SumDiff(nn.Module):
    """Aggregate inclusion and skipping activations into a scalar energy."""

    def __init__(self):
        super(SumDiff, self).__init__()
        self.w = nn.Parameter(torch.randn(1))  # Learnable weight
        self.b = nn.Parameter(torch.zeros(1))  # Learnable bias

    def forward(self, x):
        """Compute the weighted sum-difference score.

        Args:
            x: Tensor of shape ``(batch_size, 2, num_filters, seq_length)``.
                Index ``0`` is treated as inclusion and index ``1`` as skipping.

        Returns:
            Tensor of shape ``(batch_size,)`` containing the scalar energy per
            example.
        """
        # x shape: (batch_size, 2, num_filters, seq_length)
        diff = x[:, 0].sum(dim=(1, 2)) - x[:, 1].sum(dim=(1, 2))
        return self.w * diff + self.b


class ResidualTuner(nn.Module):
    """Residual calibration head used after the energy score.

    This module mirrors the original Keras implementation:

    ``Dense(hidden) -> ReLU -> BatchNorm -> Dense(hidden) -> ReLU
    -> BatchNorm -> Dense(1) -> residual add``.

    The input is expected to have a trailing dimension of size ``1`` so the
    residual addition can be applied directly.
    """

    def __init__(
        self,
        hidden_units: int = 100,
        eps: float = 1e-3,
        momentum: float = 0.99,
        use_batchnorm: bool = True,
    ):
        """Initialize the tuner network.

        Args:
            hidden_units: Width of the two hidden linear layers.
            eps: Batch normalization epsilon.
            momentum: Keras-style batch normalization momentum. Internally
                converted to the PyTorch convention.
            use_batchnorm: If False, BatchNorm1d layers are replaced with
                nn.Identity. Useful when batch statistics are unreliable
                (e.g. very small batches or fine-tuning runs).
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.use_batchnorm = use_batchnorm

        self.fc1 = nn.Linear(
            1, hidden_units
        )  # in_features fixed to 1 to match Dense(?, hidden)
        self.bn1 = (
            nn.BatchNorm1d(hidden_units, eps=eps, momentum=1 - momentum)
            if use_batchnorm
            else nn.Identity()
        )

        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = (
            nn.BatchNorm1d(hidden_units, eps=eps, momentum=1 - momentum)
            if use_batchnorm
            else nn.Identity()
        )

        self.fc3 = nn.Linear(hidden_units, 1)

        if not use_batchnorm:
            logger.info(
                "ResidualTuner: BatchNorm disabled — bn1 and bn2 replaced with nn.Identity."
            )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Run the residual calibration network.

        Args:
            inp: Tensor with shape ``(..., 1)``.

        Returns:
            Tensor with the same shape as ``inp``.

        Raises:
            ValueError: If the last dimension of ``inp`` is not ``1``.
        """
        if inp.shape[-1] != 1:
            raise ValueError(
                f"ResidualTuner expects last dim == 1, got {inp.shape[-1]}"
            )

        # Flatten to (N, C) for BatchNorm1d, then restore shape
        orig_shape = inp.shape
        x = inp.reshape(-1, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.fc3(x)

        x = x.reshape(orig_shape)
        return x + inp

    @torch.no_grad()
    def load_weights_from_dict(self, weight_dict):
        """Load weights exported from the TensorFlow/Keras tuner.

        Args:
            weight_dict: Mapping containing dense and batch-normalization
                parameters. Expected keys are ``fc1_w``, ``fc1_b``,
                ``bn1_gamma``, ``bn1_beta``, ``bn1_mean``, ``bn1_var``,
                ``fc2_w``, ``fc2_b``, ``bn2_gamma``, ``bn2_beta``,
                ``bn2_mean``, ``bn2_var``, ``fc3_w``, and ``fc3_b``.

        Returns:
            The current module instance.
        """

        def _copy(dst, src, transpose=False):
            if transpose:
                src = src.t()
            dst.copy_(src.to(dtype=dst.dtype, device=dst.device))

        # ---- Dense 1 ----
        _copy(self.fc1.weight, weight_dict["fc1_w"], transpose=True)
        _copy(self.fc1.bias, weight_dict["fc1_b"])
        logger.info("ResidualTuner.load_weights_from_dict: loaded fc1 weights.")

        # ---- BN 1 ----
        if self.use_batchnorm:
            _copy(self.bn1.weight, weight_dict["bn1_gamma"])  # gamma
            _copy(self.bn1.bias, weight_dict["bn1_beta"])  # beta
            _copy(self.bn1.running_mean, weight_dict["bn1_mean"])
            _copy(self.bn1.running_var, weight_dict["bn1_var"])
            logger.info("ResidualTuner.load_weights_from_dict: loaded bn1 weights.")
        else:
            logger.warning(
                "ResidualTuner.load_weights_from_dict: BatchNorm disabled — "
                "skipping bn1 weights (bn1_gamma, bn1_beta, bn1_mean, bn1_var)."
            )

        # ---- Dense 2 ----
        _copy(self.fc2.weight, weight_dict["fc2_w"], transpose=True)
        _copy(self.fc2.bias, weight_dict["fc2_b"])
        logger.info("ResidualTuner.load_weights_from_dict: loaded fc2 weights.")

        # ---- BN 2 ----
        if self.use_batchnorm:
            _copy(self.bn2.weight, weight_dict["bn2_gamma"])
            _copy(self.bn2.bias, weight_dict["bn2_beta"])
            _copy(self.bn2.running_mean, weight_dict["bn2_mean"])
            _copy(self.bn2.running_var, weight_dict["bn2_var"])
            logger.info("ResidualTuner.load_weights_from_dict: loaded bn2 weights.")
        else:
            logger.warning(
                "ResidualTuner.load_weights_from_dict: BatchNorm disabled — "
                "skipping bn2 weights (bn2_gamma, bn2_beta, bn2_mean, bn2_var)."
            )

        # ---- Dense 3 ----
        _copy(self.fc3.weight, weight_dict["fc3_w"], transpose=True)
        _copy(self.fc3.bias, weight_dict["fc3_b"])
        logger.info("ResidualTuner.load_weights_from_dict: loaded fc3 weights.")

        return self


class PNASModel(nn.Module):
    """Sequence-only inclusion/skipping model with 20+20 interpretable filters."""

    def __init__(
        self,
        input_length=90,
        seq_in_channels=4,
        use_batchnorm=True,
    ):
        """Initialize the model architecture.

        Args:
            input_length: Total length of the input window, including flanking
                context. Defaults to ``90``.
            seq_in_channels: Number of sequence one-hot channels. Defaults to 4.
            use_batchnorm: Passed through to ResidualTuner. If False, the two
                BatchNorm1d layers in the tuner are replaced with nn.Identity.
        """
        super(PNASModel, self).__init__()
        self.input_length = input_length

        # In channels for the sequence input.
        self.seq_in_channels = seq_in_channels

        # Fixed hyperparameters from original PNAS model
        self.seq_kernel_size = 6
        self.num_seq_filters = 20

        ### Sequence layers ###
        # (valid padding) #
        self.conv_skip = nn.Conv1d(
            in_channels=self.seq_in_channels,
            out_channels=self.num_seq_filters,
            kernel_size=self.seq_kernel_size,
            padding=0,
        )
        self.conv_incl = nn.Conv1d(
            in_channels=self.seq_in_channels,
            out_channels=self.num_seq_filters,
            kernel_size=self.seq_kernel_size,
            padding=0,
        )

        ### Aggregation ###
        self.energy_seq_struct = SumDiff()

        ### Activation ###
        self.energy_activation_incl = nn.Softplus()
        self.energy_activation_skip = nn.Softplus()

        ### Tuner ###
        self.tuner = ResidualTuner(hidden_units=4, use_batchnorm=use_batchnorm)
        self.output_activation = nn.Sigmoid()

        logger.info(
            f"PNASModel initialized — input_length={input_length}, "
            f"use_batchnorm={use_batchnorm}, "
            f"total parameters: {sum(p.numel() for p in self.parameters()):,}"
        )

    @torch.no_grad()
    def load_weights_from_dict(self, parameter_dict):
        """Load a parameter dictionary exported outside of PyTorch.

        Args:
            parameter_dict: Mapping containing convolution, aggregation, and
                tuner parameters. The nested ``"tuner"`` key is forwarded to
                :meth:`ResidualTuner.load_weights_from_dict`.

        Returns:
            The current model instance.
        """

        def _to_like(t, ref):
            return t.to(dtype=ref.dtype, device=ref.device)

        def _copy_param(dst, src):
            dst.copy_(_to_like(src, dst))

        def _load_conv1d(conv: nn.Conv1d, w_key: str, b_key: str):
            _copy_param(conv.weight, parameter_dict[w_key])
            _copy_param(conv.bias, parameter_dict[b_key])

        # -------------------------
        # Sequence conv
        # -------------------------
        _load_conv1d(self.conv_incl, "conv_incl_w", "conv_incl_b")
        _load_conv1d(self.conv_skip, "conv_skip_w", "conv_skip_b")

        # -------------------------
        # SumDiff (energy_seq_struct)
        # -------------------------
        _copy_param(self.energy_seq_struct.w, parameter_dict["energy_seq_struct_w"])
        _copy_param(self.energy_seq_struct.b, parameter_dict["energy_seq_struct_b"])

        tuner_params = parameter_dict["tuner"]
        self.tuner.load_weights_from_dict(tuner_params)

        return self

    def forward(self, x_seq, return_logits=False):
        """Compute exon inclusion probabilities.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.

        Returns:
            Tensor containing sigmoid-transformed predictions for each example.
            A batch of size one will be returned as a scalar because of the
            final ``squeeze()``.
        """
        # Compute sequence activations - each is (batch_size, F_seq, L-5)
        conv_skip_out = self.conv_skip(x_seq)
        conv_incl_out = self.conv_incl(x_seq)

        # Filter activations (sequence filters only)
        activations_skip = self.energy_activation_skip(conv_skip_out)
        activations_incl = self.energy_activation_incl(conv_incl_out)

        # Apply sum-difference
        energy_in = torch.stack(
            [activations_incl, activations_skip], dim=1
        )  # (batch_size, 2, F_seq, L-5)
        energy_out = self.energy_seq_struct(energy_in).unsqueeze(-1)  # (batch_size, 1)

        # Apply tuner
        tuner_out = self.tuner(energy_out)  # (batch_size, 1)

        if return_logits:
            return tuner_out.squeeze()  # (batch_size,)

        # compute sigmoid, return (0, 1)
        out = self.output_activation(tuner_out).squeeze()  # (batch_size,)

        return out

    def compute_sequence_activations(self, x_seq, agg="mean"):
        """Summarize sequence filter activations for inclusion and skipping.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            agg: Aggregation to apply over the sequence axis. Supported values
                are ``"mean"`` and ``"sum"``.

        Returns:
            A tuple ``(a_incl, a_skip)`` where each tensor has shape
            ``(batch_size, 20)`` after aggregation.

        Raises:
            ValueError: If ``agg`` is not supported.
        """
        conv_skip_out = self.conv_skip(x_seq)
        conv_incl_out = self.conv_incl(x_seq)
        a_skip, a_incl = F.softplus(conv_skip_out), F.softplus(conv_incl_out)

        if agg == "mean":
            a_incl = torch.mean(a_incl, dim=2)
            a_skip = torch.mean(a_skip, dim=2)
        elif agg == "sum":
            a_incl = torch.sum(a_incl, dim=2)
            a_skip = torch.sum(a_skip, dim=2)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        return a_incl, a_skip

    def compute_sr_balance(self, x_seq, agg="mean"):
        """Compute the net inclusion-minus-skipping sequence score.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            agg: Aggregation mode passed to
                :meth:`compute_sequence_activations`.

        Returns:
            Tensor of shape ``(batch_size,)`` containing the summed balance per
            example.
        """
        a_incl, a_skip = self.compute_sequence_activations(x_seq, agg)
        return a_incl.sum(dim=1) - a_skip.sum(dim=1)

    def compute_sr_profile(self, x_seq):
        """Per-position inclusion-minus-skipping SR-balance track.

        Unlike :meth:`compute_sr_balance`, this does NOT pool over the position
        axis. For every 6-nt sliding-window position it returns a single
        SR-balance scalar, summed over the 20 inclusion/skipping filters. This
        is the per-column signal used by the cross-species SD loss.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.

        Returns:
            Tensor of shape ``(batch_size, input_length - seq_kernel_size + 1)``
            giving the SR-balance at each window position.
        """
        a_incl = F.softplus(self.conv_incl(x_seq)).sum(dim=1)  # (batch, L-5)
        a_skip = F.softplus(self.conv_skip(x_seq)).sum(dim=1)  # (batch, L-5)
        return a_incl - a_skip

    def load_partial_state_dict(self, state_dict):
        """Load a (possibly partial) state dict with strict=False.

        Keys present in ``state_dict`` are loaded into the model. Keys absent
        from ``state_dict`` are left at their current values — randomly
        initialized if the model is fresh. Useful for warm-starting the conv
        filters while leaving the SumDiff/tuner head random.

        Args:
            state_dict: Mapping of parameter names to tensors. Checkpoints
                saved by the training script nest weights under
                ``"model_state_dict"``; extract that key before calling here.

        Returns:
            The ``NamedTuple`` returned by ``nn.Module.load_state_dict``
            (contains ``missing_keys`` and ``unexpected_keys``).
        """
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        will_load = sorted(model_keys & ckpt_keys)
        random_init = sorted(model_keys - ckpt_keys)
        unexpected_ckpt = sorted(ckpt_keys - model_keys)

        logger.info("=== load_partial_state_dict ===")
        logger.info(f"  Model parameters:      {len(model_keys)}")
        logger.info(f"  Checkpoint parameters: {len(ckpt_keys)}")
        logger.info(f"  Will be loaded ({len(will_load)}):")
        for k in will_load:
            logger.info(f"    [LOAD]  {k}")
        if random_init:
            logger.info(f"  Kept at current/random init ({len(random_init)}):")
            for k in random_init:
                logger.info(f"    [INIT]  {k}")
        if unexpected_ckpt:
            logger.warning(
                f"  Unexpected checkpoint keys — will be ignored ({len(unexpected_ckpt)}):"
            )
            for k in unexpected_ckpt:
                logger.warning(f"    [SKIP]  {k}")

        result = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"  Load result — missing: {len(result.missing_keys)}, "
            f"unexpected: {len(result.unexpected_keys)}"
        )
        logger.info("=== load_partial_state_dict complete ===")
        return result
