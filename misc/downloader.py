#!/usr/bin/env python
import os
import re
import sys
import json
import requests
import subprocess

from abc import ABC, abstractmethod
from bs4 import BeautifulSoup as BS
from pprint import pprint
from zipfile import ZipFile

class Downloader(ABC):
    # subdir:   directory name - created under data_dir
    # url:      url to save and use later
    # data_dir: main directory to put results in
    def __init__(self, subdir, url, data_dir='data'):
        self.url = url
        self.data_dir = data_dir
        self.downloaded_file = None
        self.extracted_file = None
        self.cleanup_items = []

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        self.data_dir = os.path.join(self.data_dir, subdir)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

    @abstractmethod
    def download(self):
        pass

    # Some sort of uncompress, usually.
    @abstractmethod
    def extract(self):
        pass

    # Remove unneeded files
    def cleanup(self):
        for filename in self.cleanup_items:
            try:
                os.remove(filename)
            except: # probably NotImplementedError...
                print(f'Could not clean up {filename}')

    #
    # Download implementations
    #
    def GenericDownload(self, url):
        self.downloaded_file = os.path.join(self.data_dir, os.path.basename(url))

        r = requests.get(url, stream=True)
        with open(self.downloaded_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)

    # 1. gets a webpage's html content
    # 2. get all links in the html
    # 3. filter links down based on input criteria
    # 4. sort resulting links by the input key
    # 5. download the "first" file
    #
    # url:        url where the html we want to parse resides
    # filter_str: string to filter for in html link 'href' text
    # file_ext:   string to filter for at the end of 'href' text
    # key:        see "sorted" documentation, this is passed to it
    # reverse:    see "sorted" documentation, this is passed to it
    def DownloadByLinkSearch(self, url, filter_str=None, file_ext=None, key=None,
                             reverse=False, debug=False):
        # Load webpage's html into memory
        html = requests.get(self.url).text
        soup = BS(html, 'html.parser')

        # Get 'a' tags with links
        tags = filter(lambda x: x.has_attr('href'), soup.find_all('a'))

        if filter_str is not None:
            tags = filter(lambda x: filter_str in x['href'], tags)

        if file_ext is not None:
            tags = filter(lambda x: x['href'].endswith(file_ext), tags)

        # Sort links
        links = sorted((tag['href'] for tag in tags), key=key, reverse=reverse)

        if len(links) == 0:
            raise RuntimeError(f"Could not find any links at '{self.url}'")

        # When links are local to the webpage, we need to fill them in.
        # ex: /feeds/xml/cpe/dictionary/official-cpe-dictionary_v2.3.xml.zip
        baseurl = re.match('(https?://[\w.]+)/', self.url).group(1)
        for i, link in enumerate(links):
            if link.startswith('/'):
                links[i] = baseurl + link

        if debug:
            pprint(links)

        return self.GenericDownload(links[0])

    #
    # Extract implementations
    #
    def Unzip(self, filename):
        if not filename.endswith('.zip'):
            print(f'Not a .zip file: {filename}')
            return

        with ZipFile(filename, 'r') as f:
            f.extractall(self.data_dir)

        self.extracted_file = filename.rstrip('.zip')
        self.cleanup_items.append(filename)

    # relies on the unix 'uncompress' command
    def Uncompress(self, filename):
        if not filename.endswith('.Z'):
            print(f'Not a .Z file: {filename}')
            return

        cmd = ['uncompress', '-f', filename]
        subprocess.run(cmd)

        self.extracted_file = filename.rstrip('.Z')

    #
    # Misc
    #
    def get_all(self):
        self.download()
        self.extract()
        self.cleanup()
        print(f'{self.extracted_file}')

class CWEDownloader(Downloader):
    def download(self):
        return self.GenericDownload(self.url)

    def extract(self):
        return self.Unzip(self.downloaded_file)

class CVEDownloader(Downloader):
    def download(self):
        return self.GenericDownload(self.url)

    def extract(self):
        return self.Uncompress(self.downloaded_file)

class CAPECDownloader(Downloader):
    def download(self):
        return self.GenericDownload(self.url)

    def extract(self):
        self.extracted_file = self.downloaded_file

class CPEDownloader(Downloader):
    def download(self):
        return self.DownloadByLinkSearch(self.url, file_ext='.zip', reverse=True)

    def extract(self):
        return self.Unzip(self.downloaded_file)

class CCEDownloader(Downloader):
    def download(self):
        return self.DownloadByLinkSearch(self.url, filter_str='COMBINED', reverse=True)

    def extract(self):
        self.extracted_file = self.downloaded_file

if __name__ == '__main__':
    CWEDownloader('CWE', 'https://cwe.mitre.org/data/xml/cwec_latest.xml.zip').get_all()
    CVEDownloader('CVE', 'https://cve.mitre.org/data/downloads/allitems.xml.Z').get_all()
    CAPECDownloader('CAPEC', 'https://capec.mitre.org/data/xml/capec_latest.xml').get_all()
    CPEDownloader('CPE', 'https://nvd.nist.gov/products/cpe').get_all()
    CCEDownloader('CCE', 'https://nvd.nist.gov/config/cce/index').get_all()
