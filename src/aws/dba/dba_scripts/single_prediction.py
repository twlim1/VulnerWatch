"""
:Title: CVSS prediction
:Description: Load all pre-trained BERT models to predict CVSS scores
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import argparse

from datetime import datetime
from lib.common_utils import CVSS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/models/')
    parser.add_argument('--description', default='')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    params = parser.parse_args()

    model_path = params.model_path
    description = params.description
    use_gpu = params.use_gpu

    print('*' * 50)
    print('Model path: {}'.format(model_path))
    print('Description: {}'.format(description))
    print('Use GPU if available: {}'.format(use_gpu))
    print('*' * 50)

    cvss = CVSS(model_path=model_path, use_gpu=use_gpu)

    start_time = datetime.now()
    scores, metrics, confidences = cvss.predict(description)
    total_time = datetime.now() - start_time

    score_b, score_i, score_e = scores
    pred_av, pred_ac, pred_pr, pred_ui, pred_sc, pred_ci, pred_ii, pred_ai = metrics
    conf_av, conf_ac, conf_pr, conf_ui, conf_sc, conf_ci, conf_ii, conf_ai = confidences
    print('  Mean: {:.4f}'.format(sum(confidences) / len(confidences)))
    print('  Max : {:.4f}'.format(max(confidences)))
    print('  Min : {:.4f}'.format(min(confidences)))
    print('\nTotal time: {}.{} seconds\n'.format(total_time.seconds, total_time.microseconds))
