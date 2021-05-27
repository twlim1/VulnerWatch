import json

from flask import Flask, request

from lib.common_utils import CVSS
from lib.db_utils import DatabaseUtil

#
# GLOBALS
#

app = Flask(__name__)

Predictor = CVSS(model_path='/models/')
du = DatabaseUtil()

MAX_OUTPUT_SIZE = 10000

#
# REST APIS
#

@app.route('/predict', methods=['POST'])
def predict():
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

@app.route('/cves', methods=['POST'])
def cves():
    try:
        inputText = request.form['text']
        inputJson = json.loads(inputText)
    except:
        return '{"Error": "Bad input"}'

    #
    # Column option
    #
    if 'cols' in inputJson:
        cols = inputJson['cols']
        if isinstance(cols, str):
            # Convert from list in string format to native list
            cols = json.loads(cols)
    else:
        # Select all columns
        cols = ['*']

    #
    # Output size option
    #
    size = inputJson.get('size', 25)

    if int(size) > MAX_OUTPUT_SIZE:
        size = str(MAX_OUTPUT_SIZE)

    #
    # Output ordering options
    #
    order_by = inputJson.get('order_by', None)
    order_dir = inputJson.get('order_dir', 'DESC')

    if order_dir.lower() == 'ascending':
        order_dir = 'ASC'
    if order_dir.lower() == 'descending':
        order_dir = 'DESC'

    # Hit the database
    try:
        results = cve_query(cols, size, order_by, order_dir)
    except Exception as e:
        return repr(e)

    return json.dumps(results)

#
# INTERNAL APIS
#

def cve_query(columns, num_results, order_by, order_dir):
    # Make cve_id column unambiguous for the SELECT
    try:
        idx = columns.index('cve_id')
        columns[idx] = 'CVEs.cve_id'
    except ValueError:
        pass # user is not selecting this column

    #
    # JSON can't serialize date types so we cast here.
    #
    for date_col in ['published_date']:
        try:
            idx = columns.index(date_col)
            columns[idx] = f"TO_CHAR({date_col}, 'Mon DD, YYYY')"
        except ValueError:
            pass # user is not selecting this column

    #
    # Make sure NULL values are displayed
    #
    for not_null_col, replacement in [('confidence', 'N/A')]:
        try:
            idx = columns.index(not_null_col)
            columns[idx] = f"COALESCE(CAST({not_null_col} AS TEXT), '{replacement}')"
        except ValueError:
            pass # user is not selecting this column

    #
    # String all columns together
    #
    column_str = ', '.join(columns)

    #
    # Generate an "ORDER BY" string
    #
    if order_by:
        if order_by == 'cve_id':
            order_by = 'CVEs.cve_id'

        order_by_str = f'ORDER BY {order_by} {order_dir}'
    else:
        order_by_str = ''

    query = f'''
        SELECT {column_str}
        FROM CVEs, Scores
        WHERE CVEs.cve_id = Scores.cve_id
        {order_by_str}
        LIMIT {num_results}
    '''

    return du.run_raw_query(query, output=True)
