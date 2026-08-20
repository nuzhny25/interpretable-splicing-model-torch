import logging

import torch.nn as nn
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

    Each bank additionally carries a learned per-filter gain on a fixed budget
    (see :meth:`filter_gains`), letting the filters compete for a bounded amount
    of influence over the SR track rather than all contributing with weight 1.
    """

    def __init__(
        self,
        seq_in_channels=4,
        gain_temp=1.0,
        gain_floor=0.02,
    ):
        """Initialize the model architecture.

        Args:
            seq_in_channels: Number of sequence one-hot channels. Defaults to 4.
            gain_temp: Softmax temperature for the per-filter gains. Larger values
                flatten the allocation (slower, softer competition between
                filters); 1.0 is the default.
            gain_floor: Minimum share of the gain budget every filter keeps, as a
                fraction in ``[0, 1)``. Guarantees ``gain >= gain_floor`` so a
                filter whose gain has decayed toward zero still receives gradient
                instead of freezing out permanently.
        """
        super(PNASModel, self).__init__()

        # In channels for the sequence input.
        self.seq_in_channels = seq_in_channels

        # Fixed hyperparameters from original PNAS model
        self.seq_kernel_size = 6
        self.num_seq_filters = 20

        self.gain_temp = gain_temp
        self.gain_floor = gain_floor

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

        ### Per-filter gain logits ###
        # Softmax logits (not log-gains: they differ by the log-sum-exp constant)
        # turned into a fixed-budget gain vector by `filter_gains`. Initialized at
        # zeros, which yields a gain of exactly 1.0 for every filter — i.e. the
        # ungained model — so a fresh run and any warm start via
        # `load_partial_state_dict` both begin from the previous architecture's
        # exact behaviour. One independent budget per bank: a single softmax over
        # all 40 filters could starve one bank entirely and turn the SR balance
        # into a one-sided readout.
        self.gain_logit_incl = nn.Parameter(torch.zeros(self.num_seq_filters))
        self.gain_logit_skip = nn.Parameter(torch.zeros(self.num_seq_filters))

        logger.info(f"total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def filter_gains(self):
        """Per-filter gains for the two banks, each summing to ``num_seq_filters``.

        The gains are a fixed budget spread over the filters of a bank::

            g = K * ((1 - floor) * softmax(logit / temp) + floor / K),  K = num_seq_filters

        so ``g.sum() == K`` exactly and ``g >= floor`` elementwise. The fixed sum
        is what makes the gains safe under the scale-free SD/abs-dev loss: because
        that loss is a ratio it is invariant to a global rescaling of the SR track,
        so unconstrained gains could all drift toward zero — zeroing the activation
        L1 penalty while leaving the SD term nominally unchanged, and leaving the
        ratio numerically ill-conditioned over a dead track. A fixed budget removes
        that direction entirely: the gains can only *reallocate* emphasis between
        filters, never change the overall scale (which the conv weights and biases
        still control). Positivity also comes for free, so no inclusion filter can
        turn into a skipping filter by going negative.

        Returns:
            Tuple ``(g_incl, g_skip)``, each of shape ``(num_seq_filters,)``.
        """
        k = self.num_seq_filters
        g_incl = k * (
            (1.0 - self.gain_floor)
            * torch.softmax(self.gain_logit_incl / self.gain_temp, dim=0)
            + self.gain_floor / k
        )
        g_skip = k * (
            (1.0 - self.gain_floor)
            * torch.softmax(self.gain_logit_skip / self.gain_temp, dim=0)
            + self.gain_floor / k
        )
        return g_incl, g_skip

    def compute_sr_profile(self, x_seq, return_activations=False):
        """Per-position inclusion-minus-skipping SR-balance track.

        For every 6-nt sliding-window position it returns a single SR-balance
        scalar, summed over the 20 inclusion/skipping filters. This is the
        per-column signal used by the cross-species SD loss.

        Each filter's activation is ``sigmoid(conv)`` in ``(0, 1)``, scaled by that
        filter's learned gain from :meth:`filter_gains` before the sum over the 20
        filters that forms the track. The gain decouples a filter's *amplitude*
        from its *selectivity*:
        a bare sigmoid has its height pinned at 1 and its slope set by the conv
        weight norm, so the only way for a motif to swing the SR track hard is to
        saturate into a near-binary matcher with vanishing gradient. With a gain a
        filter can be a sharp, graded detector and still dominate the track.
        Because the gains of a bank sum to ``num_seq_filters``, the per-position
        sum stays non-negative and bounded by the filter count exactly as it was
        without them.

        Args:
            x_seq: Sequence tensor of shape ``(batch_size, 4, input_length)``.
            return_activations: If True, also return the per-position summed
                *ungained* activations ``a_incl`` and ``a_skip``, each of shape
                ``(batch_size, input_length - 5)`` and in ``[0, num_seq_filters)``,
                so a caller can penalize the activation magnitude directly.

                These deliberately exclude the gains. A penalty on the gained sum
                decomposes as ``sum_j gain_j * mean_activation_j``, which over a
                fixed budget is minimized by moving gain onto whichever filter has
                the *lowest* mean activity — and a filter that never fires is the
                cheapest of all. That corner is self-reinforcing: budget parks on a
                dead filter, which keeps it dead. Observed on the synthetic run as
                a -0.45 correlation between gain and peak activation in the
                inclusion bank, with the liveliest filter pinned at the gain floor.
                Excluding the gains here means the activity penalty gets no say in
                the allocation: the gains are driven purely by the SD/abs-dev term,
                i.e. by how much a filter helps the conservation objective, while
                the penalty still taxes each filter's activity on its own.

        Returns:
            Tensor of shape ``(batch_size, input_length - seq_kernel_size + 1)``
            giving the SR-balance at each window position. If
            ``return_activations`` is True, instead returns the tuple
            ``(sr, a_incl, a_skip)``.
        """
        g_incl, g_skip = self.filter_gains()
        s_incl = torch.sigmoid(self.conv_incl(x_seq))  # (batch, num_filters, L-5)
        s_skip = torch.sigmoid(self.conv_skip(x_seq))
        # The SR track carries the gains; the returned activation sums do not.
        sr = (g_incl.view(1, -1, 1) * s_incl).sum(dim=1) - (
            g_skip.view(1, -1, 1) * s_skip
        ).sum(dim=1)
        a_incl = s_incl.sum(dim=1)
        a_skip = s_skip.sum(dim=1)
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
