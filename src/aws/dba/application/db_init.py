"""
:Title: Initialize Database
:Description: Initialize Database
:Developer: Teck Lim
:Create date: 05/15/2021
"""
import argparse

from lib.db_utils import DatabaseUtil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', default='vulner_watch')
    parser.add_argument('--db_user', default='postgres')
    parser.add_argument('--db_pass', default='vulnerwatch')
    parser.add_argument('--db_host', default='0.0.0.0')
    parser.add_argument('--db_port', default='5432')
    params = parser.parse_args()

    db_name = params.db_name
    db_user = params.db_user
    db_pass = params.db_pass
    db_host = params.db_host
    db_port = params.db_port

    print('*' * 50)
    print('Database name: {}'.format(db_name))
    print('Database user: {}'.format(db_user))
    print('Database pass: {}'.format(db_pass))
    print('Database host: {}'.format(db_host))
    print('Database port: {}'.format(db_port))
    print('*' * 50)

    db = DatabaseUtil(dbname=None, user=db_user, password=db_pass, host=db_host, port=db_port, enable_log=True)
    db.create_database()
    db = DatabaseUtil(dbname=db_name, user=db_user, password=db_pass, host=db_host, port=db_port, enable_log=True)
    db.create_tables()
    records = db.select_record('information_schema.tables', ['table_name'], {'table_schema': 'public'})
    print(records)
