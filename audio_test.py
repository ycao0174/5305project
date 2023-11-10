import os
import argparse
import warnings
import yaml
import torch
from look2hear.models import *  # Importing all models from look2hear.models
from look2hear.datas import *  # Importing all data handling utilities from look2hear.datas
from look2hear.metrics import MetricsTracker
from look2hear.utils import tensors_to_device, RichProgressBarTheme, MyMetricsTextColumn, BatchesProcessedColumn
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TransferSpeedColumn

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Command line arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir", default="local/mixit_conf.yml", help="Full path to save best validation model")

# Set CUDA device for GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def main(config):
    # Setup rich progress bar with custom columns
    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "•",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metricscolumn
    )

    # Define experiment directory
    config["train_conf"]["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["train_conf"]["exp"]["exp_name"]
    )
    model_path = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "best_model.pth")

    # Load the pre-trained model
    model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
        "JusperLee/TDANetBest-4ms-LRS2",
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )

    # Move model to GPU if available
    if config["train_conf"]["training"]["gpus"]:
        device = "cuda"
        model.to(device)
    model_device = next(model.parameters()).device

    # Instantiate and set up the data module
    datamodule = getattr(look2hear.datas, config["train_conf"]["datamodule"]["data_name"])(
        **config["train_conf"]["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets()

    # Directory for saving results
    ex_save_dir = os.path.join(config["train_conf"]["main_args"]["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)

    # Setup metrics tracker
    metrics = MetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))

    # Disable gradient computations for testing
    torch.no_grad().__enter__()

    # Begin testing over test set
    with progress:
        for idx in progress.track(range(len(test_set))):
            # Forward the network on the mixture
            mix, sources, key = tensors_to_device(test_set[idx], device=model_device)
            est_sources = model(mix[None])
            mix_np = mix
            sources_np = sources
            est_sources_np = est_sources.squeeze(0)

            # Update metrics
            metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)

            # Update metrics in progress bar every 50 iterations
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())

    # Finalize metrics computation
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = vars(args)

    # Load and parse configuration file
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf

    # Run the main function with the loaded configuration
    main(arg_dic)
