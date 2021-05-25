from flask import Flask, request

from lib.common_utils import CVSS

app = Flask(__name__)
Predictor = CVSS(model_path='/models/')

@app.route('/predict', methods=['POST'])
def index():
    try:
        description = request.form['text']
    except KeyError:
        return 'Bad input'

    scores, metrics, confidences = Predictor.predict(description)
    score_b, score_i, score_e = scores
    pred_av, pred_ac, pred_pr, pred_ui, pred_sc, pred_ci, pred_ii, pred_ai = metrics
    conf_av, conf_ac, conf_pr, conf_ui, conf_sc, conf_ci, conf_ii, conf_ai = confidences

    print('  Mean: {:.4f}'.format(sum(confidences) / len(confidences)))
    print('  Max : {:.4f}'.format(max(confidences)))
    print('  Min : {:.4f}'.format(min(confidences)))
    return f'{scores}\n{metrics}\n{confidences}'
