"""
:Title: CVSS prediction
:Description: Load all pre-trained BERT models to predict CVSS scores
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import argparse

from lib.common_utils import CVSS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/models/')
    parser.add_argument('--description', default='')

    params = parser.parse_args()

    model_path = params.model_path
    description = params.description

    print('*' * 50)
    print('Model path: {}'.format(model_path))
    print('Description: {}'.format(description))
    print('*' * 50)

    cvss = CVSS(model_path=model_path)

    scores, metrics, confidences = cvss.predict(description)
    score_b, score_i, score_e = scores
    pred_av, pred_ac, pred_pr, pred_ui, pred_sc, pred_ci, pred_ii, pred_ai = metrics
    conf_av, conf_ac, conf_pr, conf_ui, conf_sc, conf_ci, conf_ii, conf_ai = confidences
    print('  Mean: {:.4f}'.format(sum(confidences) / len(confidences)))
    print('  Max : {:.4f}'.format(max(confidences)))
    print('  Min : {:.4f}'.format(min(confidences)))
