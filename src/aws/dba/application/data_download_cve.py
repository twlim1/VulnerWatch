"""
:Title: Download CVE
:Description: Incrementally download CVE data since last update and save in json file
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import argparse
import requests
import time

from pytz import timezone
from datetime import datetime
from lib.db_utils import DatabaseUtil

BASE_URL = 'https://services.nvd.nist.gov/rest/json/cves/1.0'


def scrap_cve(start_page=0, total_page=1000, page_size=1000, sleep_duration=3, start_date=None):
    """
    Scrap CVE using REST API. Current total CVE is approximately 153k as of 5/15/2020
    start_page: starting page to scrap
    total_page: max number of page to scrap
    page_size: number of CVE in one page
    sleep_duration: sleep time in between each REST to avoid denial of service
    start_date: datetime object
    """
    if start_date:
        # CVE modified date has lowest granularity in minute.
        # Add one second to get the next CVE since last update
        start_date = start_date.strftime('%Y-%m-%dT%H:%M:01:000 UTC-05:00')
        print('Start Date: {}'.format(start_date))

    cve_items = list()
    for page_no in range(start_page, total_page):
        for _ in range(5):
            try:
                print('Retrieving page: {}'.format(page_no + 1))
                url = '{}?startIndex={}&resultsPerPage={}'.format(BASE_URL, page_no * page_size, page_size)
                if start_date:
                    url = '{}&modStartDate={}'.format(url, start_date)
                response = requests.get(url)
                response_json = response.json()
                break
            except Exception as e:
                print('Something is wrong. Sleep for {} sec before retrying: {}'.format(sleep_duration, e))
                time.sleep(sleep_duration)
        else:
            raise BaseException('Exhausted all attempts')

        cve_items += response_json['result']['CVE_Items']
        print('Scrapped: {}'.format(len(cve_items)))
        if len(cve_items) == response_json['totalResults']:
            print('Completed scrapping..')
            break
        time.sleep(sleep_duration)
    print('Total scrapped: {}'.format(len(cve_items)))
    return cve_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='vulner_watch')
    parser.add_argument('--db_user', default='postgres')
    parser.add_argument('--db_pass', default='vulnerwatch')
    parser.add_argument('--db_host', default='0.0.0.0')
    parser.add_argument('--db_port', default='5432')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true')
    params = parser.parse_args()

    db_name = params.db_name
    db_user = params.db_user
    db_pass = params.db_pass
    db_host = params.db_host
    db_port = params.db_port
    test_mode = params.test_mode

    print('*' * 50)
    print('Database name: {}'.format(db_name))
    print('Database user: {}'.format(db_user))
    print('Database pass: {}'.format(db_pass))
    print('Database host: {}'.format(db_host))
    print('Database port: {}'.format(db_port))
    print('Test mode: {}'.format(test_mode))
    print('*' * 50)
    db = DatabaseUtil(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port, enable_log=True)

    # CVE timestamp is Eastern timezone
    row = db.query_metadata()
    last_refresh_timestamp = row[0][0] if row and row[0] else None
    print('Last refresh: {} EST'.format(last_refresh_timestamp))
    curr_refresh_timestamp = datetime.now(tz=timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:00-05')
    print('Current refresh: {} EST'.format(curr_refresh_timestamp))

    # Sample code to sync from specific time
    if test_mode:
        last_refresh_timestamp = datetime.strptime('2021-05-19', '%Y-%m-%d')
    new_cve = scrap_cve(sleep_duration=1, start_date=last_refresh_timestamp)
    for cve in new_cve:
        db.insert_cve(cve)
    print('Total CVE: {}'.format(len(new_cve)))
    db.insert_metadata(timestamp=curr_refresh_timestamp, count=len(new_cve))

    print('CVEs record count: {}'.format(db.run_raw_query('SELECT count(*) FROM cves', True)))
    print('Scores record count: {}'.format(db.run_raw_query('SELECT count(*) FROM scores', True)))
    print('Metadata record count: {}'.format(db.run_raw_query('SELECT count(*) FROM metadata', True)))
    print('Predicted record count: {}'.format(db.run_raw_query('SELECT count(*) FROM scores where model <> \'manual\'', True)))
    print('Metadata record count: {}'.format(db.run_raw_query('SELECT * FROM metadata', True)))

