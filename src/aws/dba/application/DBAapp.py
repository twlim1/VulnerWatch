import os
import time
import logging

from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler

from lib.common_utils import CVSS

app = Flask(__name__)
Predictor = CVSS(model_path='/models/')

logging.basicConfig(filename='stdout.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


def refresh_data():
    app.logger.info('Scheduler task started at: {}'.format(time.strftime("%A, %d. %B %Y %I:%M:%S %p")))

    output = os.system('python ../dba_scripts/data_download_cve.py')
    app.logger.debug('Return: {}'.format(output))
    app.logger.info('Completed cve download: {}'.format(time.strftime("%A, %d. %B %Y %I:%M:%S %p")))

    # Save the GPU resources for the interactive webpage (use --use_gpu to make use of GPU resource if available)
    output = os.system('python ../dba_scripts/batch_prediction.py')
    app.logger.debug('Return: {}'.format(output))
    app.logger.info('Completed batch prediction: {}'.format(time.strftime("%A, %d. %B %Y %I:%M:%S %p")))


# In debug mode, Flask's reloader will load the flask app twice. Scheduler will run in both master process and child
# thread. A workaround is to not start scheduler in master process
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_data, trigger='cron', minute=30)
    scheduler.start()


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
