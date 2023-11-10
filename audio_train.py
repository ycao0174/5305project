import os
import argparse
import json
import yaml
from pprint import pprint
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from look2hear import datas, models, system, losses, utils
from look2hear.system import make_optimizer
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

import warnings

warnings.filterwarnings("ignore")


def main(config):
    data_module_class = getattr(datas, config["datamodule"]["data_name"])
    model_class = getattr(models, config["audionet"]["audionet_name"])
    scheduler_class = getattr(system.schedulers, config["scheduler"]["sche_name"], None)
    train_loss_class = getattr(losses, config["loss"]["train"]["loss_func"])
    val_loss_class = getattr(losses, config["loss"]["val"]["loss_func"])
    system_class = getattr(system, config["training"]["system"])

    # Setup data module
    datamodule = data_module_class(**config["datamodule"]["data_config"])
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader()

    # Instantiate model, optimizer, and scheduler
    model = model_class(sample_rate=config["datamodule"]["data_config"]["sample_rate"],
                        **config["audionet"]["audionet_config"])
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    if scheduler_class:
        scheduler = {
            "scheduler": scheduler_class(optimizer,
                                         len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64),
            "interval": "step",
        }

    # Saving configuration
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Instantiate loss functions
    train_loss = train_loss_class(getattr(losses, config["loss"]["train"]["sdr_type"]),
                                  **config["loss"]["train"]["config"])
    val_loss = val_loss_class(getattr(losses, config["loss"]["val"]["sdr_type"]), **config["loss"]["val"]["config"])

    # Instantiate system
    training_system = system_class(
        audio_model=model, loss_func={"train": train_loss, "val": val_loss}, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, scheduler=scheduler, config=config
    )

    # Define callbacks and logger
    callbacks = [ModelCheckpoint(dirpath=exp_dir, filename="{epoch}", monitor="val_loss/dataloader_idx_0", mode="min",
                                 save_top_k=5, verbose=True, save_last=True)]
    if config["training"].get("early_stop"):
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    comet_logger = WandbLogger(name=config["exp"]["exp_name"], save_dir=exp_dir, project="Real-work-dataset")

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=1,
        accelerator='cpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=True,
    )
    trainer.fit(training_system)
    print_only("Finished Training")

    # Save best model details
    best_k = {k: v.item() for k, v in trainer.checkpoint_callback.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(trainer.checkpoint_callback.best_model_path)
    training_system.load_state_dict(state_dict=state_dict["state_dict"])
    training_system.cpu()
    torch.save(training_system.audio_model.serialize(), os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_dir", default="local/conf.yml", help="Full path to save best validation model")

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)
