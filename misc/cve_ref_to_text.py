#!/usr/bin/env python
import os
import sys
import json
import argparse
import requests
import xmltodict
import logging
import getopt

from tqdm import tqdm
from time   import sleep
from pprint import pprint
from abc    import ABC, abstractmethod
from bs4    import BeautifulSoup as BS

logging.basicConfig(level=logging.INFO)

data_dir = 'data'
output_dir = 'output'

input_xml = os.path.join(data_dir, 'allitems.xml')
input_json = os.path.join(data_dir, 'someitems.json')

max_cves = 100
files_created = {}

#
# This code should only run the first time the script runs. Takes a few seconds.
#
if not os.path.exists(input_json): # create it if needed
    logging.info("Converting XML to JSON")
    with open(input_xml, 'rb') as f:
        data = xmltodict.parse(f)

    # Filter out older entries
    items = list(d for d in data['cve']['item'] if 'CVE-202' in d['@name'])

    with open(input_json, 'w') as f:
        json.dump(items, f)

#
# Classes
#
class Scraper(ABC):
    def __init__(self, url, lazy=False):
        self.url = url
        self.text = '' # extracted text

        if lazy: # user intends to load later
            self.html = None
            self.soup = None
        else:
            self.load()

    def load(self):
        self.html = requests.get(self.url).text
        self.soup = BS(self.html, 'html.parser')

    def debug(self, filename='debug.html'):
        with open(filename, 'w') as f:
            f.write(self.soup.prettify())

    # Returns relevant text parsed using self.url
    @abstractmethod
    def extract(self):
        pass

class OpenSUSE(Scraper):
    def extract(self):
        self.text = '\n'.join([i.text for i in self.soup.find_all('div', {'class': 'body'})])

class Fedora(Scraper):
    def extract(self):
        self.text = '\n'.join([i.text for i in self.soup.find_all('div', {'class': 'email-body'})])

class Ubuntu(Scraper):
    def extract(self):
        self.text = self.soup.find('div', {'id': 'main-content'}).text

class Cisco(Scraper):
    def extract(self):
        self.text = self.soup.find('div', {'id': 'advisorycontentbody'}).text

class Github(Scraper):
    def __init__(self, url):
        if 'gist.github.com' in url:
            self.extract_fn = self.extract_gist
        elif url.endswith('.md'):
            self.extract_fn = self.extract_md
        elif '/commit/' in url:
            self.extract_fn = self.extract_commit
        else:
            raise ValueError('unsupported url')

        super().__init__(url)

    def extract(self):
        self.extract_fn()
        if not self.text:
            # no reason to continue further
            raise RuntimeError('Github text data is empty')

    def extract_gist(self):
        self.text = '\n'.join([i.text for i in self.soup.find_all('div', {'class': 'gist-content'})])

    def extract_md(self):
        self.text = self.soup.find('div', {'id': 'readme'}).text

    def extract_commit(self):
        # not all commits have this description, errors are expected
        self.text = '\n'.join([i.text for i in self.soup.find_all('div', {'class': 'commit-desc'})])

#
# Map URLs to objects
#
mapping = {
    'lists.opensuse':   OpenSUSE,
    'lists.fedora':     Fedora,
    'usn.ubuntu':       Ubuntu,
    'tools.cisco':      Cisco,
    'github.com':       Github,
}

#
# List of commonly cited sources that aren't as straighforward to use.
#
harder = {
    # https://www.oracle.com/security-alerts/cpuoct2020.html
    # https://www.oracle.com/security-alerts/cpuoct2020traditional.html
    'oracle.com': None,

    # From CVE-2020-10002
    # https://support.apple.com/en-us/HT212011
    # https://support.apple.com/en-us/HT211928
    'support.apple': None,

    # Much too noisy/useless:
    # https://lists.apache.org/thread.html/rdf44341677cf7eec7e9aa96dcf3f37ed709544863d619cca8c36f133@%3Ccommits.airflow.apache.org%3E
    # https://lists.apache.org/thread.html/rb2b981912446a74e14fe6076c4b7c7d8502727ea0718e6a65a9b1be5@%3Cissues.zookeeper.apache.org%3E
    'lists.apache': None,
}

#
# For debugging purposes - prints truncated URL of the input object
#
def printUrl(ref):
    for suffix in ['.com', '.org', '.net', '.it', '.cc', '.ch', '.in', '.jp']:
        try:
            idx = ref['@url'].index(suffix)
            break
        except ValueError:
            continue
    else:
        idx = -5
    print(ref['@url'][:idx+4])

#
# Main
#

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_json) as f:
        data = json.load(f)
    
    counter = 0
    LIMIT = 100
    successes = 0

    logging.info("Getting Content ...")
    for datum in tqdm(data):
        # Skip CVEs without references
        if 'refs' not in datum or not datum['refs']:
            continue

        # Goal is to end up with a list of references called 'refs'
        refs = datum['refs']['ref']
        if isinstance(refs, dict):
            refs = [refs]

        for refnum, ref in enumerate(refs):
            #printUrl(ref)

            # Try every url we know how to handle
            for try_url, OBJ in mapping.items():
                if try_url in ref['@url']:
                    try:
                        obj = OBJ(ref['@url'])
                    except ValueError:
                        continue # cheesy way to force the "else: -> continue below"
                    break
            else:
                # We don't know how to handle urls that end up here
                continue

            try:
                obj.extract()
            except:
                continue # suppress and ignore all errors for now

            # Output extracted text to file
            fname = os.path.join(output_dir , f'{datum["@name"]}_{refnum}.txt')
            with open(fname, 'w') as f:
                f.write(datum['@name'] + '\n')
                f.write(ref['@url'] + '\n')
                f.write(obj.text)
                counter += 1

            try:
                files_created[try_url] += 1
            except KeyError:
                files_created[try_url] = 1

            successes += 1
            if successes > max_cves:
                pprint(files_created)
                sys.exit(0)

            if counter== LIMIT:
                logging.info(f"{counter} url parsed")
                sys.exit(0)
            sleep(.2)
            

