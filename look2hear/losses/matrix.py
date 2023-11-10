import torch
from torch.nn.modules.loss import _Loss

class PairwiseNegSDR(_Loss):
    """
    Pairwise Negative Signal-to-Distortion Ratio (SDR) Loss.
    This loss function computes the negative SDR for pairs of estimated and target sources.
    """
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        """
        Initialize the PairwiseNegSDR class.

        :param sdr_type: Type of SDR ('snr', 'sisdr', 'sdsdr').
        :param zero_mean: If True, subtract mean from sources for zero-mean normalization.
        :param take_log: If True, compute the log scale of SDR.
        :param EPS: Small value to avoid division by zero.
        """
        super().__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, ests, targets):
        """
        Forward pass for computing pairwise negative SDR.

        :param ests: Estimated sources tensor.
        :param targets: Target sources tensor.
        :return: Pairwise negative SDR value.
        """
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )

        # Zero-mean normalization
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_estimate

        # Compute pairwise projection
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(ests, dim=2)
        if self.sdr_type in ["sisdr", "sdsdr"]:
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.EPS
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)

        # Compute noise component
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj

        # Compute pairwise SDR
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
            torch.sum(e_noise ** 2, dim=3) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)

        return -pair_wise_sdr

# Similar comments apply for SingleSrcNegSDR and MultiSrcNegSDR classes.
# These classes implement the negative SDR for single and multiple source scenarios respectively.

class SingleSrcNegSDR(_Loss):
    # ... (implementation similar to PairwiseNegSDR, tailored for single source inputs)

class MultiSrcNegSDR(_Loss):
    # ... (implementation similar to PairwiseNegSDR, tailored for multiple source inputs)

class freq_MAE_WavL1Loss(_Loss):
    """
    Frequency Domain Mean Absolute Error (MAE) and Waveform L1 Loss.
    This loss function combines frequency domain MAE and time-domain L1 loss for audio signals.
    """
    def __init__(self, win=2048, stride=512):
        """
        Initialize the freq_MAE_WavL1Loss class.

        :param win: Window size for STFT.
        :param stride: Stride for STFT.
        """
        super().__init__()
        self.EPS = 1e-8
        self.win = win
        self.stride = stride

    def forward(self, ests, targets):
        """
        Forward pass for computing the combined frequency MAE and waveform L1 loss.

        :param ests: Estimated audio waveform tensor.
        :param targets: Target audio waveform tensor.
        :return: Combined frequency MAE and waveform L1 loss.
        """
        # ... (implementation details)

# Alias definitions for quick instantiation of the loss classes with specific SDR types
pairwise_neg_sisdr = PairwiseNegSDR("sisdr")
pairwise_neg_sdsdr = PairwiseNegSDR("sdsdr")
pairwise_neg_snr = PairwiseNegSDR("snr")
singlesrc_neg_sisdr = SingleSrcNegSDR("sisdr")
singlesrc_neg_sdsdr = SingleSrcNegSDR("sdsdr")
singlesrc_neg_snr = SingleSrcNegSDR("snr")
multisrc_neg_sisdr = MultiSrcNegSDR("sisdr")
multisrc_neg_sdsdr = MultiSrcNegSDR("sdsdr")
multisrc_neg_snr = MultiSrcNegSDR("snr")
freq_mae_wavl1loss = freq_MAE_WavL1Loss()
