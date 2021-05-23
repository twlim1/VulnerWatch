"""
:Title: CVSS prediction
:Description: Load all pre-trained BERT models to predict CVSS scores
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import argparse

from collections import OrderedDict
from lib.db_utils import DatabaseUtil
from lib.common_utils import CVSS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/models/')
    parser.add_argument('--model_name', default='bert_base_uncased_v1.0')
    parser.add_argument('--db_name', default='vulner_watch')
    parser.add_argument('--db_user', default='postgres')
    parser.add_argument('--db_pass', default='vulnerwatch')
    parser.add_argument('--db_host', default='0.0.0.0')
    parser.add_argument('--db_port', default='5432')
    params = parser.parse_args()

    model_path = params.model_path
    model_name = params.model_name
    db_name = params.db_name
    db_user = params.db_user
    db_pass = params.db_pass
    db_host = params.db_host
    db_port = params.db_port

    print('*' * 50)
    print('Model path: {}'.format(model_path))
    print('Model name: {}'.format(model_name))
    print('Database name: {}'.format(db_name))
    print('Database user: {}'.format(db_user))
    print('Database pass: {}'.format(db_pass))
    print('Database host: {}'.format(db_host))
    print('Database port: {}'.format(db_port))
    print('*' * 50)

    cvss = CVSS(model_path=model_path)
    db = DatabaseUtil(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port, enable_log=True)
    records = db.query_cves_without_score()
    print('CVEs to predict: {}'.format(len(records)))
    for record in records:
        cve_id = record[0]
        description = record[1]
        scores, metrics, confidences = cvss.predict(description)
        score_b, score_i, score_e = scores
        pred_av, pred_ac, pred_pr, pred_ui, pred_sc, pred_ci, pred_ii, pred_ai = metrics
        conf_av, conf_ac, conf_pr, conf_ui, conf_sc, conf_ci, conf_ii, conf_ai = confidences
        print('  Mean: {:.4f}'.format(sum(confidences) / len(confidences)))
        print('  Max : {:.4f}'.format(max(confidences)))
        print('  Min : {:.4f}'.format(min(confidences)))

        metrics = OrderedDict()
        metrics['attack_vector'] = pred_av
        metrics['attack_complexity'] = pred_ac
        metrics['privileges_required'] = pred_pr
        metrics['user_interaction'] = pred_ui
        metrics['scope'] = pred_sc
        metrics['confidentiality'] = pred_ci
        metrics['integrity'] = pred_ii
        metrics['availability'] = pred_ai
        metrics['v3_base_score'] = score_b
        metrics['v3_exploitability_score'] = score_e
        metrics['v3_impact_score'] = score_i
        metrics['confidence'] = sum(confidences) / len(confidences)
        metrics['lowest_confidence'] = min(confidences)
        metrics['model'] = model_name
        metrics['cve_id'] = cve_id
        db.insert_score(metrics)
