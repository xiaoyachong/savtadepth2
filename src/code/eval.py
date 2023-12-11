import sys
import yaml
import torch
from torchvision import transforms
from fastai.vision.all import unet_learner, Path, resnet34, MSELossFlat, get_files, L, PILImageBW
from custom_data_loading import create_data
from eval_metric_calculation import compute_eval_metrics
from dagshub import dagshub_logger
from tqdm import tqdm


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: %s <test_data_path>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    with open(r"./src/code/params.yml") as f:
        params = yaml.safe_load(f)

    data_path = Path(sys.argv[1])
    data = create_data(data_path)

    arch = {'resnet34': resnet34}
    loss = {'MSELossFlat': MSELossFlat()}

    learner = unet_learner(data,
                           arch.get(params['architecture']),
                           n_out=int(params['num_outs']),
                           loss_func=loss.get(params['loss_func']),
                           path='src/',
                           model_dir='models')
    learner = learner.load('model')

    filenames = get_files(Path(data_path), extensions='.jpg')
    test_files = L([Path(i) for i in filenames])

    for i, sample in tqdm(enumerate(test_files.items),
                          desc="Predicting on test images",
                          total=len(test_files.items)):
        pred = learner.predict(sample)[0]
        pred = PILImageBW.create(pred).convert('L')
        pred.save("src/eval/" + str(sample.stem) + "_pred.png")
        if i < 10:
            pred.save("src/eval/examples/" + str(sample.stem) + "_pred.png")

    print("Calculating metrics...")
    metrics = compute_eval_metrics(test_files)

    with dagshub_logger(
            metrics_path="logs/test_metrics.csv",
            should_log_hparams=False
    ) as logger:
        # Metric logging
        logger.log_metrics(metrics)

    print("Evaluation Done!")
