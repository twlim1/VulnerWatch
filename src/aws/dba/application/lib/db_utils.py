import re
import psycopg2

from psycopg2 import errors
from collections import OrderedDict
from datetime import datetime
from lib.common_utils import value_to_integer

UniqueViolation = errors.lookup('23505')  # Correct way to Import the psycopg2 errors

DB_NAME = 'vulner_watch'
DB_USER = 'postgres'
DB_PASS = 'vulnerwatch'
DB_HOST = '52.36.152.180'
DB_PORT = '5432'
TABLE_METADATA = 'metadata'
TABLE_SCORES = 'scores'
TABLE_CVES = 'cves'
CREATE_DATABASE = \
    '''
    CREATE DATABASE {database}
    '''.format(database=DB_NAME)
CREATE_TABLES = \
    '''
    CREATE TABLE IF NOT EXISTS {metadata} (
        id                      SERIAL PRIMARY KEY,
        last_data_refresh       TIMESTAMP NOT NULL,
        records_updated         INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS {cves} (
        cve_id                  TEXT PRIMARY KEY,
        description             TEXT,
        published_date          TIMESTAMP NOT NULL,
        last_modified_date      TIMESTAMP NOT NULL
    );
    CREATE TABLE IF NOT EXISTS {scores} (
        id                      SERIAL PRIMARY KEY,
        attack_vector           INTEGER NOT NULL,
        attack_complexity       INTEGER NOT NULL,
        privileges_required     INTEGER NOT NULL,
        user_interaction        INTEGER NOT NULL,
        scope                   INTEGER NOT NULL,
        confidentiality         INTEGER NOT NULL,
        integrity               INTEGER NOT NULL,
        availability            INTEGER NOT NULL,
        v3_base_score           REAL NOT NULL,
        v3_exploitability_score REAL NOT NULL,
        v3_impact_score         REAL NOT NULL,
        confidence              REAL,
        lowest_confidence       REAL,
        model                   TEXT NOT NULL,
        cve_id                  TEXT REFERENCES {cves}(cve_id)
    );
    '''.format(metadata=TABLE_METADATA, scores=TABLE_SCORES, cves=TABLE_CVES)
DROP_DATABASE = \
    '''
    DROP DATABASE IF EXISTS {database};
    '''.format(database=DB_NAME)
DROP_TABLES = \
    '''
    DROP TABLE IF EXISTS {scores};
    DROP TABLE IF EXISTS {cves};
    DROP TABLE IF EXISTS {metadata};
    '''.format(metadata=TABLE_METADATA, scores=TABLE_SCORES, cves=TABLE_CVES)


class DatabaseUtil:
    def __init__(self, dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT, enable_log=True):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self.enable_log = enable_log

    def close(self):
        if self.conn:
            self.conn.close()

    def create_database(self):
        self.run_raw_query(DROP_DATABASE)
        self.run_raw_query(CREATE_DATABASE)

    def create_tables(self):
        self.run_raw_query(DROP_TABLES)
        self.run_raw_query(CREATE_TABLES)

    def select_record(self, table, columns, values_to_match):
        """
        select database rows based on input values
        :param table: table to query
        :param columns: list of columns to display
        :param values_to_match: dictoinary of value to match
        :return: None
        """
        column_list = list()
        value_list = list()
        for column, items in values_to_match.items():
            column_list.append(column)
            value_list.append(items)

        statements = ['{}=\'{}\''.format(c, v) for c, v in zip(column_list, value_list)]
        if statements:
            where_statement = ' WHERE {}'.format(' AND '.join(statements))
        else:
            where_statement = ''

        query = 'SELECT {} FROM {}{}'.format(','.join(columns), table, where_statement)
        if self.enable_log:
            print('Query: {}'.format(query))

        self.cursor.execute(query)
        records = self.cursor.fetchall()
        if self.enable_log:
            # print('Records: {}'.format(records))
            print('Total Records: {}'.format(len(records)))
        return records

    def run_raw_query(self, query, output=False):
        self.cursor.execute(query)
        return self.cursor.fetchall() if output else None

    def insert_cve(self, cve):
        metadata = OrderedDict()
        metadata['cve_id'] = cve['cve']['CVE_data_meta']['ID']
        metadata['description'] = ' '.join([text['value'] for text in cve['cve']['description']['description_data']]).replace('\'', '')
        metadata['published_date'] = datetime.strptime(cve['publishedDate'], '%Y-%m-%dT%H:%MZ').strftime('%Y-%m-%d %H:%M:00-05')
        metadata['last_modified_date'] = datetime.strptime(cve['lastModifiedDate'], '%Y-%m-%dT%H:%MZ').strftime('%Y-%m-%d %H:%M:00-05')
        try:
            query = 'INSERT INTO {table} {keys} VALUES {values};'.format(
                table=TABLE_CVES,
                keys=re.sub(r'\'', '', str(tuple(metadata.keys()))),
                values=tuple(str(v) for v in metadata.values()))
            if self.enable_log:
                print('Query: {}'.format(query))
            self.run_raw_query(query, False)
            print('Inserted CVE: {}'.format(metadata['cve_id']))
        except UniqueViolation as e:
            # CVE already in DB, delete score and update existing cve
            self.delete_score(metadata['cve_id'])
            key_value = ['{}=\'{}\''.format(k, v) for k, v in zip(metadata.keys(), metadata.values())]
            query = 'UPDATE {table} SET {key_value} WHERE cve_id=\'{cve_id}\';'.format(
                table=TABLE_CVES,
                key_value=', '.join(key_value),
                cve_id=metadata['cve_id']
            )
            if self.enable_log:
                print('Query: {}'.format(query))
            self.run_raw_query(query, False)
            print('Updated CVE: {}'.format(metadata['cve_id']))

        # Insert score that comes with CVE
        if cve.get('impact') and cve['impact'].get('baseMetricV3'):
            base_metric = cve['impact']['baseMetricV3']
            metrics = OrderedDict()
            metrics['attack_vector'] = value_to_integer('AV', base_metric['cvssV3']['attackVector'])
            metrics['attack_complexity'] = value_to_integer('AC', base_metric['cvssV3']['attackComplexity'])
            metrics['privileges_required'] = value_to_integer('PR', base_metric['cvssV3']['privilegesRequired'])
            metrics['user_interaction'] = value_to_integer('UI', base_metric['cvssV3']['userInteraction'])
            metrics['scope'] = value_to_integer('SC', base_metric['cvssV3']['scope'])
            metrics['confidentiality'] = value_to_integer('CI', base_metric['cvssV3']['confidentialityImpact'])
            metrics['integrity'] = value_to_integer('II', base_metric['cvssV3']['integrityImpact'])
            metrics['availability'] = value_to_integer('AI', base_metric['cvssV3']['availabilityImpact'])
            metrics['v3_base_score'] = base_metric['cvssV3']['baseScore']
            metrics['v3_exploitability_score'] = base_metric['exploitabilityScore']
            metrics['v3_impact_score'] = base_metric['impactScore']
            metrics['model'] = 'manual'
            metrics['cve_id'] = metadata['cve_id']
            self.insert_score(metrics)

    def insert_score(self, scores):
        query = 'INSERT INTO {table} {keys} VALUES {values};'.format(
            table=TABLE_SCORES,
            keys=re.sub(r'\'', '', str(tuple(scores.keys()))),
            values=tuple(str(v) for v in scores.values()))
        if self.enable_log:
            print('Query: {}'.format(query))
        self.run_raw_query(query, False)

    def delete_score(self, cve_id):
        query = 'DELETE from {table} WHERE cve_id=\'{cve_id}\';'.format(
            table=TABLE_SCORES, cve_id=cve_id)
        if self.enable_log:
            print('Query: {}'.format(query))
        self.run_raw_query(query, False)

    def insert_metadata(self, timestamp, count):
        query = 'INSERT INTO {table} {keys} VALUES {values};'.format(
            table=TABLE_METADATA,
            keys='(last_data_refresh, records_updated)',
            values=tuple(str(v) for v in (timestamp, count)))
        if self.enable_log:
            print('Query: {}'.format(query))
        self.run_raw_query(query, False)

    def query_metadata(self):
        query = 'SELECT MAX(last_data_refresh) FROM {table};'.format(
            table=TABLE_METADATA)
        if self.enable_log:
            print('Query: {}'.format(query))
        return self.run_raw_query(query, True)

    def query_cves_without_score(self):
        query = 'SELECT * FROM {table_cves} c ' \
                'LEFT JOIN {table_scores} s on c.cve_id = s.cve_id ' \
                'WHERE s.id is NULL;'.format(table_cves=TABLE_CVES,
                                             table_scores=TABLE_SCORES)
        if self.enable_log:
            print('Query: {}'.format(query))
        return self.run_raw_query(query, True)
