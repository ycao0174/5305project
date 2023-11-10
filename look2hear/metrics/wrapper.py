import csv
import torch
import numpy as np
import logging
from ..losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Tracks and logs metrics for audio signal separation."""

    def __init__(self, save_file: str = ""):
        """Initializes metrics lists and CSV writer."""
        self.metrics = {
            "one_snr": [], "one_snr_i": [], "one_sisnr": [], "one_sisnr_i": [],
            "two_snr": [], "two_snr_i": [], "two_sisnr": [], "two_sisnr_i": []
        }
        self.csv_columns = [
            "snt_id", "one_snr", "one_snr_i", "one_si-snr", "one_si-snr_i",
            "two_snr", "two_snr_i", "two_si-snr", "two_si-snr_i"
        ]
        self.save_file = save_file
        self.pit_sisnr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.pit_snr = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")

    def _write_to_csv(self, row):
        """Writes a row of data to the CSV file."""
        with open(self.save_file, "a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_columns)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def __call__(self, mix, clean, estimate, key):
        """Processes and logs metrics for a single audio sample."""
        # Process and compute metrics
        one_snr, one_snr_i = self._calculate_metric(mix, clean, estimate, self.pit_snr, 2)
        one_sisnr, one_sisnr_i = self._calculate_metric(mix, clean, estimate, self.pit_sisnr, 2)
        two_snr, two_snr_i = self._calculate_metric(mix, clean, estimate, self.pit_snr, 0, 2)
        two_sisnr, two_sisnr_i = self._calculate_metric(mix, clean, estimate, self.pit_sisnr, 0, 2)

        # Log and accumulate metrics
        row = {
            "snt_id": key,
            "one_snr": -one_snr, "one_snr_i": -one_snr_i,
            "one_si-snr": -one_sisnr, "one_si-snr_i": -one_sisnr_i,
            "two_snr": -two_snr, "two_snr_i": -two_snr_i,
            "two_si-snr": -two_sisnr, "two_si-snr_i": -two_sisnr_i
        }
        self._write_to_csv(row)
        self._accumulate_metrics(one_snr, one_snr_i, one_sisnr, one_sisnr_i, two_snr, two_snr_i, two_sisnr, two_sisnr_i)

    def _calculate_metric(self, mix, clean, estimate, metric_func, start_idx=0, end_idx=None):
        """Calculates a specific metric."""
        end_idx = end_idx if end_idx is not None else clean.shape[1]
        _, ests_np = metric_func(
            estimate.unsqueeze(0)[:, start_idx:end_idx], clean.unsqueeze(0)[:, start_idx:end_idx], return_ests=True
        )
        metric_value = metric_func(
            ests_np, clean.unsqueeze(0)[:, start_idx:end_idx]
        )
        mix_repeated = torch.stack([mix] * clean.shape[1], dim=0)
        metric_baseline = metric_func(
            mix_repeated.unsqueeze(0)[:, start_idx:end_idx], clean.unsqueeze(0)[:, start_idx:end_idx]
        )
        metric_improvement = metric_value - metric_baseline
        return metric_value.item(), metric_improvement.item()

    def _accumulate_metrics(self, *args):
        """Accumulates metrics for final averaging."""
        for key, value in zip(self.metrics.keys(), args):
            self.metrics[key].append(value)

    def final(self):
        """Calculates and logs final average metrics."""
        avg_metrics = {key: np.mean(values) for key, values in self.metrics.items()}
        avg_metrics["snt_id"] = "avg"
        self._write_to_csv(avg_metrics)

        # Logging average metrics
        for key, value in avg_metrics.items():
            if key != "snt_id":
                logger.info(f"Mean {key.replace('_', ' ')} is {value}")

# Example usage of the MetricsTracker
# tracker = MetricsTracker("results.csv")
# tracker(mix, clean, estimate, key)
# tracker.final()
