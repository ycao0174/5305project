# 5305project

## Overview
This project involves training a modified model for speech separation based on the TDANet architecture. It supports processing of LRS2 and TIMIT datasets and includes functionality for both training and evaluation of the model.

## Dependencies
- All dependencies are defined in `requirements.txt`.

## Dataset Preparation
### LRS2 Dataset
- Use `preprocess/process_lrs2.py` to prepare the LRS2 dataset.

### TIMIT Dataset
- Use `preprocess/process_timit.py` to prepare the TIMIT dataset.

### Custom Dataset
- You can also prepare your own dataset for training.

## Training
- Train the model using `audio_train.py`.
- Command: `python audio_train.py --conf_dir=config.yml`.
- Configuration details can be found in `config.yml`.

## Evaluation
- Use `audio_test.py` for model evaluation.
- Command: `python audio_test.py --conf_dir=conf.yml`.
- Configuration details can be found in `config.yml`.
- 
## Demo
- Use `inference.py` for single audio file separation.
- Command: `python inference.py --audio_path='your test audio dir' --output_dir='save file to ur dir' --record_duration='default:5s'`.
- This script separates existing audio files or, if `audio_path` does not exist, starts recording.

## Modified TDANet Model
- Our modified version of the model from the original TDANet is stored in `TDANet.py`.
- The original model can be found in `look2hear/models/TDANetli.py`.

## References and Acknowledgements
- Some of the model and evaluation code references [TDANet by JusperLee](https://github.com/JusperLee/TDANet).
- Original TDANet Paper:
@inproceedings{tdanet2023iclr,
title={An efficient encoder-decoder architecture with top-down attention for speech separation},
author={Li, Kai and Yang, Runxuan and Hu, Xiaolin},
booktitle={ICLR},
year={2023}
}
