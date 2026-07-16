import logging

import torch.nn as nn
import torch.nn.functional as F
import torch

logger = logging.getLogger(__name__)


class PNASModel(nn.Module):
    """Sequence-only inclusion/skipping model with 20+20 interpretable filters.

    This variant is stripped down to the motif-discovery path used by the
    cross-species SD-minimization training: two convolutional filter banks plus
    the per-position SR-balance readout. The PSI-prediction head (SumDiff energy
    aggregation, ResidualTuner calibration, and sigmoid output) has been removed
    because the SD loss never runs ``forward`` — it consumes only
    :meth:`compute_sr_profile`.
    """

    def __init__(
        self,
        seq_in_channels=4,
    ):
        """Initialize the model architecture.

        Args:
            seq_in_channels: Number of sequence one-hot channels. Defaults to 4.
        """
        super(PNASModel, self).__init__()

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

        logger.info(f"total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def compute_sr_profile(self, x_seq, return_activations=False):
        """Per-position inclusion-minus-skipping SR-balance track.

        For every 6-nt sliding-window position it returns a single SR-balance
        scalar, summed over the 20 inclusion/skipping filters. This is the
        per-column signal used by the cross-species SD loss.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            return_activations: If True, also return the per-position summed
                tanh-capped activations ``a_incl`` and ``a_skip``. Each filter's
                activation is ``tanh(softplus(conv))`` in ``[0, 1)``, so the
                per-position sum over the 20 filters is non-negative and bounded
                by the filter count; each is of shape
                ``(batch_size, input_length - 5)``, so a caller can penalize the
                activation magnitude directly.

        Returns:
            Tensor of shape ``(batch_size, input_length - seq_kernel_size + 1)``
            giving the SR-balance at each window position. If
            ``return_activations`` is True, instead returns the tuple
            ``(sr, a_incl, a_skip)``.
        """
        a_incl = torch.tanh(F.softplus(self.conv_incl(x_seq))).sum(dim=1)
        a_skip = torch.tanh(F.softplus(self.conv_skip(x_seq))).sum(dim=1)
        sr = a_incl - a_skip
        if return_activations:
            return sr, a_incl, a_skip
        return sr

    def load_partial_state_dict(self, state_dict):
        """Load a (possibly partial) state dict with strict=False.

        Keys present in ``state_dict`` are loaded into the model. Keys absent
        from ``state_dict`` are left at their current values — randomly
        initialized if the model is fresh. Useful for warm-starting the conv
        filters from another checkpoint.

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
