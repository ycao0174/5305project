import csv
import torch
import numpy as np
import logging
from ..losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr

logger = logging.getLogger(__name__)

class SplitMetricsTracker:
    def __init__(self, save_file: str = ""):
        self.metrics = {"one_all_snrs": [], "one_all_snrs_i": [], "one_all_sisnrs": [], "one_all_sisnrs_i": [],
                        "two_all_snrs": [], "two_all_snrs_i": [], "two_all_sisnrs": [], "two_all_sisnrs_i": []}
        self.csv_columns = ["snt_id", "one_snr", "one_snr_i", "one_si-snr", "one_si-snr_i",
                            "two_snr", "two_snr_i", "two_si-snr", "two_si-snr_i"]
        self.save_file = save_file
        self.pit_sisnr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.pit_snr = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")

        with open(self.save_file, "w", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_columns)
            writer.writeheader()

    def __call__(self, mix, clean, estimate, key):
        estimate_unsqueezed = estimate.unsqueeze(0)
        clean_unsqueezed = clean.unsqueeze(0)

        _, ests_np = self.pit_snr(estimate_unsqueezed, clean_unsqueezed, return_ests=True)
        mix_repeated = torch.stack([mix] * clean.shape[0], dim=0).unsqueeze(0)

        metrics = self.compute_metrics(mix_repeated, clean_unsqueezed, ests_np)
        self.log_metrics(key, metrics)

    def compute_metrics(self, mix, clean, estimate):
        metrics = {}
        for i in range(2):
            clean_segment = clean[:, i:i+2] if i == 0 else clean[:, 2:3]
            estimate_segment = estimate[:, i:i+2] if i == 0 else estimate[:, 2:3]
            mix_segment = mix[:, i:i+2] if i == 0 else mix[:, 2:3]

            metrics[f"{i+1}_sisnr"], metrics[f"{i+1}_sisnr_i"] = self.calculate_metric_diff(
                mix_segment, clean_segment, estimate_segment, self.pit_sisnr)
            metrics[f"{i+1}_snr"], metrics[f"{i+1}_snr_i"] = self.calculate_metric_diff(
                mix_segment, clean_segment, estimate_segment, self.pit_snr)

        return metrics

    def calculate_metric_diff(self, mix, clean, estimate, metric_func):
        metric = metric_func(estimate, clean)
        metric_baseline = metric_func(mix, clean)
        metric_improvement = metric - metric_baseline
        return -metric.item(), -metric_improvement.item()

    def log_metrics(self, key, metrics):
        row = {"snt_id": key}
        row.update(metrics)
        self.write_to_csv(row)
        for metric, value in metrics.items():
            self.metrics[f"{metric}s"].append(value)

    def write_to_csv(self, row):
        with open(self.save_file, "a", newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_columns)
            writer.writerow(row)

    def final(self):
        average_metrics = {metric: np.mean(values) for metric, values in self.metrics.items()}
        average_metrics["snt_id"] = "avg"
        self.write_to_csv(average_metrics)
        for metric, value in average_metrics.items():
            if metric != "snt_id":
                logger.info(f"Mean {metric.replace('_', ' ')} is {value}")
        self.log_final_metrics(average_metrics)

    def log_final_metrics(self, metrics):
        for metric, value in metrics.items():
            if metric != "snt_id":
                logger.info(f"Mean {metric} is {value}")

# Example usage:
# tracker = SplitMetricsTracker("results.csv")
# tracker(mix, clean, estimate, key)
# tracker.final()
