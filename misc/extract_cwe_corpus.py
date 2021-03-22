#!/usr/bin/env python
import os
from bs4 import BeautifulSoup as BS

data_loc = 'data/cwec_v4.2.xml'

columns = [
    'description',
    'extended_description',
    'modes_of_introduction',
    'common_consequences',
    'potential_mitigations',
]

with open(data_loc) as f:
    soup = BS(f.read(), 'lxml')

for column in columns:
    for tag in soup.find_all(column):
        print(tag.text)
