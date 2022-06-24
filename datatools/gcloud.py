import collections

import io

import dateutil.parser
import json
import pickle

import platform
import psutil
import sys

import datetime
import os
import tempfile
import uuid
from distutils.util import strtobool

import backoff
import google.auth
import humanfriendly
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
import google.cloud.logging  # Don't conflict with standard logging
from google.api_core.exceptions import GoogleAPICallError, ClientError

from datatools.log import getLogger
from datatools.utils import chunks, remove_punctuation

logger = getLogger()

TIMEOUT = 600
DEFAULT_PROJECT = 'dmrc-platforms'
DEFAULT_BUCKET = 'dmrc-platforms'

# https://cloud.google.com/bigquery/pricing
GOOGLE_PRICE_PER_BYTE = 5 / 10E12  # $5 per tb.


class GCloud:
    def __init__(self, project_id=None, GOOGLE_JSON_KEY=None, name=None, save_bucket=DEFAULT_BUCKET, location=None):
        if GOOGLE_JSON_KEY:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_JSON_KEY
        if not project_id:
            try:
                project_id = os.environ["DEVSHELL_PROJECT_ID"]
            except:
                project_id = DEFAULT_PROJECT

        self.project_id = project_id
        self.bq_client = None
        self.gcs_client = None
        self.logging_client = None
        self.bucket = save_bucket

        self.get_clients(project_id=self.project_id, location=location)

        if not name:
            name = os.path.basename(sys.argv[0])
        node_name = platform.uname().node
        username = psutil.Process().username()
        run_time = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        self.default_save_dir = f'gs://{self.bucket}/runs/{name}/{run_time}-{username}-{node_name}'

    def get_clients(self, project_id=None, location='us-central1'):
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if not project_id:
            project_id = default_project

        # Make clients.
        # noinspection PyTypeChecker
        self.bq_client = bigquery.Client(
            credentials=credentials,
            project=project_id,
            location=location
        )

        self.gcs_client = storage.Client(
            credentials=credentials,
            project=project_id
        )

        self.logging_client = google.cloud.logging.Client(
            credentials=credentials,
            project=project_id
        )


    @backoff.on_exception(backoff.expo, (GoogleAPICallError, ClientError), max_tries=5)
    def upload_json(self, data, uri):
        # make sure the data is serializable first
        data = self.scrub_serializable(data)

        # Try to upload as json
        data = json.dumps(data)

        logger.debug(f'Uploading file {uri}.')
        blob = google.cloud.storage.blob.Blob.from_string(uri, client=self.gcs_client)

        blob.upload_from_string(data)
        logger.info(f'Successfully uploaded file {uri} with {len(data)} lines written.')

        return uri

    @backoff.on_exception(backoff.expo, (GoogleAPICallError, ClientError), max_tries=5)
    def upload_binary(self, uri=None, data=None):
        assert data is not None

        if uri is None:
            uri = f'{self.default_save_dir}/{uuid.uuid1()}.pickle'

        logger.debug(f'Uploading file {uri}.')
        blob = google.cloud.storage.blob.Blob.from_string(uri=uri, client=self.gcs_client)

        # Try to do this in memory
        blob.upload_from_file(file_obj=io.BytesIO(pickle.dumps(data)))
        logger.info(f'Successfully uploaded file {uri}.')

        return uri

    @backoff.on_exception(backoff.expo, (GoogleAPICallError, ClientError), max_tries=5)
    def upload_dataframe_json(self, data, uri):
        # use pandas to upload
        # First, convert to serialisable formats
        rows = self.scrub_serializable(data.to_dict(orient='records'))
        rows = pd.DataFrame(rows)
        rows.to_json(uri, orient='records', lines=True)
        return uri

    def run_query(self, sql, destination=None, overwrite=False,
                  do_not_return_results=False):
        t0 = datetime.datetime.now()

        job_config = {
            'use_legacy_sql': False,
        }

        if destination:
            logger.debug(f"Saving results to {destination}.")
            job_config['destination'] = destination
            job_config['allow_large_results'] = True

        if overwrite:
            job_config['write_disposition'] = 'WRITE_TRUNCATE'
        else:
            job_config['write_disposition'] = 'WRITE_APPEND'

        job_config = bigquery.QueryJobConfig(**job_config)
        job = self.bq_client.query(sql, job_config=job_config)

        result = job.result()  # blocks until job is finished

        bytes_billed = job.total_bytes_billed
        cache_hit = job.cache_hit
        approx_cost = None
        try:
            approx_cost = bytes_billed * GOOGLE_PRICE_PER_BYTE
            bytes_billed = humanfriendly.format_size(bytes_billed)
            logger.increment_run_summary('BigQuery Bytes Billed', job.total_bytes_billed)
        except (TypeError, ValueError) as e:
            logger.error(f"Unable to interpret and save bytes billed: {bytes_billed}. Error: {e}")
            pass  # couldn't get a numeric value from BQ or we're using the wrong logger.
        except AttributeError as e:
            logger.warning(
                f"We do not appear to be using the PlatformGovernance logger. Unable to save Bytes Billed: {bytes_billed}.")

        time_taken = datetime.datetime.now() - t0
        logger.info(
            f"Query stats: Ran in {time_taken} seconds, cache hit: {cache_hit}, billed {bytes_billed}, approx cost ${approx_cost:0.2}.")

        if do_not_return_results:
            return True
        else:
            # job.result() blocks until the query has finished.
            results_df = job.result().to_dataframe()
            return results_df

    def save(self, data, **params):
        # Emergency save routine. We should be able to find some way of dumping the data.
        # Try multiple methods in order until we get a result.
        try:
            if 'bq_dest' in params and 'schema' in params:
                destination = params['bq_dest']
                self.upload_rows(rows=data, destination=destination, schema=params['schema'], **params)
                logger.info(f"Uploaded data to BigQuery: {destination}.")
                return destination
        except Exception as e:
            logger.exception(
                f'Critical failure. Unable to upload data: {e}',
                stack_info=False)
            pass

        # Try to upload to GCS
        uri = params.get('uri', f'{self.default_save_dir}/{uuid.uuid1()}')

        upload_methods = [self.upload_dataframe_json, self.upload_json, self.upload_binary,
                          self.dump_to_disk, self.dump_pickle]

        for method in upload_methods:
            try:
                return method(data, uri)
            except (GoogleAPICallError, ClientError) as e:
                logger.error(f'Error saving data to {uri}: {e}')
            except Exception as e:
                logger.error(f'Could not save dataframe to GCS: {e}')

        raise IOError(f'Critical failure. Unable to save using any method in {upload_methods}')

    def dump_to_disk(self, data):
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as out:
            if isinstance(data, pd.DataFrame):
                data.to_json(out)
            else:
                out.write(json.dumps(data))
            logger.warning(f"Successfully dumped to pickle on disk: {out.name}.")
            return out.name

    def dump_pickle(self, data):
        filename = f'data-dumped-{uuid.uuid1()}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        logger.warning(f"Successfully dumped to pickle on disk: {filename}.")
        return filename

    def upload_rows(self, schema, rows, destination, ensure_schema_compliance=False,
                    len_chunks=600, create_if_not_exists=False):
        """ Upload results to Google Bigquery """

        inserted = False
        bq_rows = rows

        # For BigQuery, we can only upload rows that match the schema.
        # Here, remove any items in the list of dicts that do not match
        if ensure_schema_compliance:
            bq_rows = [self.construct_dict_from_schema(schema, row) for row in bq_rows]

        # Make sure objects are serializable. So far, special handling for Numpy types and dates:
        bq_rows = self.scrub_serializable(bq_rows)

        table = None
        try:
            table = self.bq_client.get_table(destination)
        except Exception as e:
            logger.send_exception(
                message_body=f"Unable to save rows. Table {destination} does not exist or there was some other "
                             f"problem getting the table: {e}", subject="Error inserting rows to Google Bigquery!")

        # google recommends chunks of ~500 rows
        for index, chunk in enumerate(chunks(bq_rows, len_chunks)):
            str_error = ""
            inserted = False
            if table:
                try:
                    logger.debug(
                        f"Inserting {len(chunk)} rows to BigQuery table {destination}, chunk {index}.")

                    errors = self.bq_client.insert_rows(table, chunk)
                    if not errors:
                        inserted = True

                        logger.debug(
                            f"Successfully pushed {len(chunk)} rows to BigQuery table {destination}, attempt {index}.")
                        logger.increment_run_summary('BigQuery rows saved', len(chunk))
                    else:
                        str_error += f"Google BigQuery returned an error result: {str(errors[:2])}\n\n"

                except Exception as e:
                    str_error += "Exception pushing to BigQuery table {}, attempt {}, reason: {}\n\n".format(
                        destination, index, str(e)[:2000])
            else:
                str_error += "Could not get table, so could not push rows.\n\n"

            if not inserted:
                logger.increment_run_summary('Failed rows saved to disk', len(chunk))
                save_name = self.save(chunk, suffix='.rows')
                logger.error(
                    "Failed to upload rows! Saving {} rows to {} for later upload.".format(
                        len(rows), save_name))

                message_body = f"Error pushing to BigQuery table {destination}, chunk {index}.\n\n"
                message_body += str_error

                logger.send_exception(
                    message_body=message_body,
                    subject=f"Error inserting rows to Google Bigquery! Table: {destination}")
                logger.debug("First three rows:")
                logger.debug(chunk[:3])

        return inserted

    @staticmethod
    def nan_ints(df, convert_strings=False, subset=None):
        # Convert int, float, and object columns to int64 if possible (requires pandas >0.24 for nullable int format)
        types = ['int64', 'float64']
        if subset is None:
            subset = list(df)
        if convert_strings:
            types.append('object')
        for col in subset:
            try:
                if df[col].dtype in types:
                    df[col] = df[col].astype(float).astype('Int64')
            except:
                pass
        return df

    def construct_dict_from_schema(self, schema, d):
        """ Recursively construct a new dictionary, using only fields from d that are in schema """
        new_dict = {}
        keys_deleted = []
        for row in schema:
            key_name = row['name']
            if key_name in d:
                # Handle nested fields
                if isinstance(d[key_name], dict) and 'fields' in row:
                    new_dict[key_name] = self.construct_dict_from_schema(row['fields'], d[key_name])

                # Handle repeated fields - use the same schema as we were passed
                elif isinstance(d[key_name], list) and 'fields' in row:
                    new_dict[key_name] = [self.construct_dict_from_schema(row['fields'], item) for item in d[key_name]]

                elif isinstance(d[key_name], str) and (
                        str.upper(remove_punctuation(d[key_name])) == 'NULL' or remove_punctuation(d[key_name]) == ''):
                    # don't add null values
                    keys_deleted.append(key_name)
                    pass

                elif not d[key_name] is None:
                    if str.upper(row['type']) in ['TIMESTAMP', 'DATETIME', 'DATE']:
                        # convert string dates to datetimes
                        if not isinstance(d[key_name], datetime.datetime):
                            try:
                                _ts = None
                                if type(d[key_name]) == str:
                                    if d[key_name].isnumeric():
                                        _ts = float(d[key_name])
                                    else:
                                        new_dict[key_name] = pd.to_datetime(d[key_name])

                                if type(d[key_name]) == int or type(d[key_name]) == float or _ts:
                                    if not _ts:
                                        _ts = d[key_name]

                                    try:
                                        new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts)
                                    except (ValueError, OSError):
                                        # time is likely in milliseconds
                                        new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts / 1000)

                                elif not isinstance(d[key_name], datetime.datetime):
                                    new_dict[key_name] = pd.to_datetime(d[key_name])
                            except ValueError as e:
                                logger.error(f"Unable to parse {key_name} item {key_name} into datetime format: {e}")
                                pass
                        else:
                            # Already a datetime, move it over
                            new_dict[key_name] = d[key_name]

                        # if it's a date only field, remove time
                        if str.upper(row['type']) == 'DATE':
                            new_dict[key_name] = new_dict[key_name].date()
                    elif str.upper(row['type']) in ['INTEGER', 'FLOAT']:
                        # convert string numbers to integers
                        if isinstance(d[key_name], str):
                            try:
                                new_dict[key_name] = pd.to_numeric(d[key_name])
                            except:
                                logger.error(
                                    "Unable to parse {} item {} into numeric format".format(key_name, d[key_name]))
                                pass
                        else:
                            new_dict[key_name] = d[key_name]
                    elif str.upper(row['type']) == 'BOOLEAN':
                        if isinstance(d[key_name], str):
                            try:
                                new_dict[key_name] = bool(strtobool(d[key_name]))
                            except ValueError:
                                if new_dict[key_name] == '':
                                    pass  # no value
                                else:
                                    logger.error(
                                        "Unable to parse {} item {} into boolean format".format(key_name, d[key_name]))
                                    pass
                        else:
                            try:
                                new_dict[key_name] = bool(d[key_name])
                            except ValueError:
                                logger.error(
                                    "Unable to parse {} item {} into boolean format".format(key_name, d[key_name]))
                    else:
                        new_dict[key_name] = d[key_name]
            else:
                keys_deleted.append(key_name)

        if len(keys_deleted) > 0:
            logger.debug(
                "Cleaned dict according to schema. Did not find {} keys: {}".format(len(keys_deleted), keys_deleted))

        set_orig = set(d.keys())
        set_new = set(new_dict.keys())
        set_removed = set_orig - set_new

        if len(set_removed) > 0:
            logger.debug(
                "Cleaned dict according to schema. Did not include {} keys: {}".format(len(set_removed), set_removed))

        return new_dict

    def scrub_serializable(self, d):
        try:
            if isinstance(d, list):
                d = [self.scrub_serializable(x) for x in d]
                return d

            if isinstance(d, dict):
                for key in list(d.keys()):
                    # iterate through dict keys, but don't iterate over the dict itself
                    if d[key] is None:
                        del d[key]
                    elif isinstance(d[key], dict):
                        d[key] = self.scrub_serializable(d[key])
                    elif isinstance(d[key], list):
                        d[key] = [self.scrub_serializable(x) for x in d[key]]
                    elif hasattr(d[key], 'dtype'):
                        # Numpy objects.
                        if isinstance(d[key], np.ndarray):
                            # For arrays, convert to list and apply same processing to each element
                            d[key] = d[key].tolist()
                            d[key] = [ self.scrub_serializable(x) for x in d[key] ]
                        else:
                            # For other types, convert to their base python type
                            d[key] = d[key].item()
                    elif isinstance(d[key], datetime.date) or isinstance(d[key], datetime.datetime):
                        # ensure dates and datetimes are stored as strings in ISO format for uploading
                        d[key] = d[key].isoformat()
                    elif isinstance(d[key], uuid.UUID):
                        # if the obj is uuid, we simply return the value of uuid as a string
                        d[key] = str(d[key])
                    elif isinstance(d[key], (int, float)) and pd.isna(d[key]):
                        del d[key]

            return d
        except Exception as e:
            logger.error(f'Unable to scrub dict to serialisable format: {e}')
            raise

    def check_table_exists(self, bq_table_location):
        from google.api_core.exceptions import NotFound
        try:
            table = self.bq_client.get_table(bq_table_location)
            return True
        except NotFound as e:
            return False


def sql_search_any(field, keywords):
    keywords = [str.lower(k) for k in keywords]
    like_statement = [f"OR lower({field}) LIKE '%{k}%'" for k in keywords]
    like_statement = " ".join(like_statement)[3:]
    return f"({like_statement})"


def construct_dict_from_schema(schema, d):
    """ Recursively construct a new dictionary, using only fields from d that are in schema """
    new_dict = {}
    keys_deleted = []
    for row in schema:
        key_name = row['name']
        if key_name in d:
            # Handle nested fields
            if isinstance(d[key_name], dict) and 'fields' in row:
                new_dict[key_name] = construct_dict_from_schema(row['fields'], d[key_name])

            # Handle repeated fields - use the same schema as we were passed
            elif isinstance(d[key_name], list) and 'fields' in row:
                new_dict[key_name] = [construct_dict_from_schema(row['fields'], item) for item in d[key_name]]

            elif isinstance(d[key_name], str) and (
                    str.upper(remove_punctuation(d[key_name])) == 'NULL' or remove_punctuation(d[key_name]) == ''):
                # don't add null values
                keys_deleted.append(key_name)
                pass

            elif not d[key_name] is None:
                if str.upper(row['type']) == 'TIMESTAMP':
                    # convert dates to datetimes
                    if not isinstance(d[key_name], datetime.datetime):
                        try:
                            _ts = None
                            if type(d[key_name]) == str:
                                if d[key_name].isnumeric():
                                    _ts = float(d[key_name])
                                else:
                                    new_dict[key_name] = dateutil.parser.parse(d[key_name])

                            if type(d[key_name]) == int or type(d[key_name]) == float or _ts:
                                if not _ts:
                                    _ts = d[key_name]

                                try:
                                    new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts)
                                except (ValueError, OSError):
                                    # time is likely in milliseconds
                                    new_dict[key_name] = datetime.datetime.utcfromtimestamp(_ts / 1000)

                            elif not isinstance(d[key_name], datetime.datetime):
                                new_dict[key_name] = pd.to_datetime(d[key_name])
                        except:
                            logger.error(
                                "Unable to parse {} item {}, type {}, into date format".format(key_name, d[key_name],
                                                                                               type(d[key_name])))
                            # new_dict[key_name] = d[key_name]
                            pass
                    else:
                        # Already a datetime, move it over
                        new_dict[key_name] = d[key_name]
                elif str.upper(row['type']) == 'INTEGER':
                    # convert string numbers to integers
                    if isinstance(d[key_name], str):
                        try:
                            new_dict[key_name] = int(remove_punctuation(d[key_name]))
                        except:
                            logger.error("Unable to parse {} item {} into integer format".format(key_name, d[key_name]))
                            pass
                            # new_dict[key_name] = d[key_name]
                    else:
                        new_dict[key_name] = d[key_name]
                else:
                    new_dict[key_name] = d[key_name]
        else:
            keys_deleted.append(key_name)

    if len(keys_deleted) > 0:
        logger.debug(
            "Cleaned dict according to schema. Did not find {} keys: {}".format(len(keys_deleted), keys_deleted))

    set_orig = set(d.keys())
    set_new = set(new_dict.keys())
    set_removed = set_orig - set_new

    if len(set_removed) > 0:
        logger.debug(
            "Cleaned dict according to schema. Did not include {} keys: {}".format(len(set_removed), set_removed))

    return new_dict
