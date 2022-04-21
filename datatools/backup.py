#!/usr/bin/env python3
import datetime

from docopt import docopt
from google.cloud import bigquery
from legit_functions import bq_get_clients
from legit_functions import cfg
from datatools.log import setup_logging

def backup_tables_in_dataset(bq_client, bq_dataset, gcs_destination_path, single_table=None, json=True, gzip=True):
    date = datetime.datetime.utcnow()

    if single_table:
        tables = [ bq_client.dataset(bq_dataset).table(single_table) ]
    else:
        logger.info(f"Getting a list of all tables in {bq_dataset}.")
        tables = bq_client.list_tables(bq_dataset)

    for table_ref in tables:
        if table_ref.table_type == "TABLE":
            destination_uri = f"{gcs_destination_path}/{date:%Y%m%d}/{table_ref.table_id}/{table_ref.table_id}_{date:%Y%m%d}_*"
            dump_table(bq_client, table_ref, destination_uri=destination_uri, gzip=gzip, json=json)
        else:
            logger.info(f"Skipping {table_ref.table_id} because it is type: {table_ref.table_type}.")


def dump_table(bq_client, table_ref, destination_uri, gzip, json):
    logger.info(f"Dumping table {table_ref.table_id} to {destination_uri}...")
    job_config = bigquery.ExtractJobConfig()
    if json:
        destination_uri += ".json"
        job_config.destination_format = (
            bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON)
    else:
        destination_uri += ".csv"
    job_config.destination_format = (
        bigquery.DestinationFormat.CSV)
    if gzip:
        job_config.compression = bigquery.Compression.GZIP
        destination_uri += ".gz"

    extract_job = bq_client.extract_table(table_ref, destination_uri, job_config=job_config)

    if not extract_job.errors:
        logger.info("Job entered successfully.")
        return True
    else:
        logger.error("Unable to dump {table_ref.table_id} to {destination_uri}! Error: {extract_job.errors}")


def main():
    """ Backup a table or a dataset to GCS

    Usage:
      backup.py [-v] --dataset=<dataset> [--table=<table>] <gcs_destination>

    Options:
      -h --help                 Show this screen.
      -v --verbose              Increase verbosity for debugging.
    """

    args = docopt(main.__doc__, version='BQ dataset backup 0.1')

    global logger
    logger = setup_logging(None, verbose=args['--verbose'], interactive_only=True)
    bq_client, bq_storageclient = bq_get_clients()

    backup_tables_in_dataset(bq_client, bq_dataset=args['--dataset'],
            gcs_destination_path=args['<gcs_destination>'], json=True, gzip=True)





if __name__ == '__main__':
    main()