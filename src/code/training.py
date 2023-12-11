"""Trains or fine-tunes a model for the task of monocular depth estimation
Receives 1 arguments from argparse:
  <data_path> - Path to the dataset which is split into 2 folders - train and test.
"""
import sys
import yaml
from fastai.vision.all import unet_learner, Path, resnet34, rmse, MSELossFlat
from custom_data_loading import create_data
from dagshub.fastai import DAGsHubLogger


if __name__ == "__main__":
    # Check if got all needed input for argparse
    if len(sys.argv) != 2:
        print("usage: %s <data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    with open(r"./src/code/params.yml") as f:
        params = yaml.safe_load(f)

    data = create_data(Path(sys.argv[1]))

    metrics = {'rmse': rmse}
    arch = {'resnet34': resnet34}
    loss = {'MSELossFlat': MSELossFlat()}

    learner = unet_learner(data,
                           arch.get(params['architecture']),
                           metrics=metrics.get(params['train_metric']),
                           wd=float(params['weight_decay']),
                           n_out=int(params['num_outs']),
                           loss_func=loss.get(params['loss_func']),
                           path=params['source_dir'],
                           model_dir=params['model_dir'],
                           cbs=DAGsHubLogger(
                               metrics_path="logs/train_metrics.csv",
                               hparams_path="logs/train_params.yml"))

    print("Training model...")
    learner.fine_tune(epochs=int(params['epochs']),
                      base_lr=float(params['learning_rate']))
    print("Saving model...")
    learner.save('model')
    print("Done!")
