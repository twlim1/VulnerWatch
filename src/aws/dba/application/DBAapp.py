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

    r = {}
    r['score_b'], r['score_i'], r['score_e'] = scores
    r['pred_av'], r['pred_ac'], r['pred_pr'], r['pred_ui'], r['pred_sc'], r['pred_ci'], r['pred_ii'], r['pred_ai'] = [float(m) for m in metrics]
    r['conf_av'], r['conf_ac'], r['conf_pr'], r['conf_ui'], r['conf_sc'], r['conf_ci'], r['conf_ii'], r['conf_ai'] = confidences

    r['mean_conf'] = sum(confidences) / len(confidences)
    r['max_conf'] = max(confidences)
    r['min_conf'] = min(confidences)

    return json.dumps(r)

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

    #
    # Output offset option
    #
    offset = inputJson.get('offset', '0')

    #
    # Search string option
    #
    search = inputJson.get('search', '').replace(' ', '%')
    search = f"'%{search}%'"

    # Hit the database
    try:
        results = cve_query(cols, size, order_by, order_dir, offset, search)
    except Exception as e:
        return repr(e)

    return json.dumps(results)

#
# INTERNAL APIS
#

def cve_query(columns, num_results, order_by, order_dir, offset, search):
    #
    # Columns: First, make cve_id column unambiguous for the SELECT
    #
    try:
        idx = columns.index('cve_id')
        columns[idx] = 'CVEs.cve_id'
    except ValueError:
        pass # user is not selecting this column

    # JSON can't serialize date types so we cast here.
    for date_col in ['published_date']:
        try:
            idx = columns.index(date_col)
            columns[idx] = f"TO_CHAR({date_col}, 'Mon DD, YYYY')"
        except ValueError:
            pass # user is not selecting this column

    # Make sure NULL values are displayed
    for not_null_col, replacement in [('confidence', 'N/A')]:
        try:
            idx = columns.index(not_null_col)
            columns[idx] = f"COALESCE(CAST({not_null_col} AS TEXT), '{replacement}')"
        except ValueError:
            pass # user is not selecting this column

    # String all columns together
    column_str = ', '.join(columns)

    #
    # Generate an "ORDER BY" string
    #
    if not order_by:
        order_by_str = ''
    else:
        if order_by == 'cve_id':
            order_by = 'CVEs.cve_id'

        order_by_str = f'ORDER BY {order_by} {order_dir}'

    #
    # Generate a text search string first (needs to happen before columns are modified)
    #
    if not search:
        search_str = ''
    else:
        search_str = f'''
            AND
            (
                CVEs.cve_id LIKE {search} 
                OR
                description LIKE {search} 
            )
        '''

    query = f'''
        SELECT {column_str}
        FROM CVEs, Scores
        WHERE CVEs.cve_id = Scores.cve_id
        {search_str}
        {order_by_str}
        LIMIT {num_results} OFFSET {offset}
    '''

    return du.run_raw_query(query, output=True)
