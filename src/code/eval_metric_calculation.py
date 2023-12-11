import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_errors(target, prediction):
    thresh = np.maximum((target / prediction), (prediction / target))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(target - prediction) / target)
    sq_rel = np.mean(((target - prediction) ** 2) / target)

    rmse = (target - prediction) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(target) - np.log(prediction)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(prediction) - np.log(target)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(target) - np.log10(prediction))).mean()

    return a1, a2, a3, abs_rel, sq_rel, rmse, rmse_log, silog, log_10


def compute_eval_metrics(test_files):
    min_depth_eval = 1e-3
    max_depth_eval = 10

    num_samples = len(test_files)

    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    rmse = np.zeros(num_samples, np.float32)
    rmse_log = np.zeros(num_samples, np.float32)
    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)

    for i in tqdm(range(num_samples), desc="Calculating metrics for test data", total=num_samples):
        sample_path = test_files[i]
        target_path = str(sample_path.parent/(sample_path.stem + "_depth.png"))
        pred_path = "src/eval/" + str(sample_path.stem) + "_pred.png"

        target_image = Image.open(target_path)
        pred_image = Image.open(pred_path)

        target = np.asarray(target_image)
        pred = np.asarray(pred_image)

        target = target / 25.0
        pred = pred / 25.0

        pred[pred < min_depth_eval] = min_depth_eval
        pred[pred > max_depth_eval] = max_depth_eval
        pred[np.isinf(pred)] = max_depth_eval

        target[np.isinf(target)] = 0
        target[np.isnan(target)] = 0

        valid_mask = np.logical_and(target > min_depth_eval, target < max_depth_eval)

        a1[i], a2[i], a3[i], abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], silog[i], log10[i] = \
            compute_errors(target[valid_mask], pred[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        a1.mean(), a2.mean(), a3.mean(),
        abs_rel.mean(), sq_rel.mean(), rmse.mean(), rmse_log.mean(), silog.mean(), log10.mean()))

    return dict(a1=a1.mean(), a2=a2.mean(), a3=a3.mean(),
                abs_rel=abs_rel.mean(), sq_rel=sq_rel.mean(),
                rmse=rmse.mean(), rmse_log=rmse_log.mean(),
                log10=log10.mean(), silog=silog.mean())
