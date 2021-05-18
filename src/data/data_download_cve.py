"""
:Title: Download CVE
:Description: Incrementally download CVE data since last update and save in json file
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import requests
import json
import time
import os

from datetime import datetime

file_path = '../../data/raw/cve.json'
base_url = 'https://services.nvd.nist.gov/rest/json/cves/1.0'


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
                url = '{}?startIndex={}&resultsPerPage={}'.format(base_url, page_no * page_size, page_size)
                if start_date:
                    url = '{}&modStartDate={}'.format(url, start_date)
                response = requests.get(url)
                response_json = response.json()
                break
            except Exception:
                print('Something is wrong. Sleep for {} sec before retrying'.format(sleep_duration))
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
    cve_list = []
    cve_id_list = []
    latest_time = None

    # Check CVE was downloaded before and only download the difference
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fp:
            cve_list = json.load(fp)
        print('CVE downloaded before: {}'.format(len(cve_list)))

        for cve in cve_list:
            cve_id_list.append(cve['cve']['CVE_data_meta']['ID'])
            cve_last_modified_date = datetime.strptime(cve['lastModifiedDate'], '%Y-%m-%dT%H:%MZ')
            if not latest_time:
                latest_time = cve_last_modified_date
            else:
                if latest_time < cve_last_modified_date:
                    latest_time = cve_last_modified_date
        print('Last updated time: {}'.format(latest_time))
    else:
        print('CVE has not been downloaded before')

    if latest_time is None:
        new_cve_list = scrap_cve(sleep_duration=1)
    else:
        new_cve_list = scrap_cve(sleep_duration=1, start_date=latest_time)
        updated_cve = 0
        for new_cve in new_cve_list:
            new_cve_id = new_cve['cve']['CVE_data_meta']['ID']
            if new_cve_id in cve_id_list:
                updated_cve += 1
                index = cve_id_list.index(new_cve_id)
                cve_list.pop(index)
                cve_id_list.pop(index)
        print('Total updated CVE: {}'.format(updated_cve))
        new_cve_list += cve_list
    print('Total unique CVE: {}'.format(len(new_cve_list)))

    print('Saving CVE to file')
    with open(file_path, 'w') as fp:
        fp.write(json.dumps(new_cve_list))
