import datetime
import os
import uuid
from distutils.util import strtobool

import backoff
import google.auth
import humanfriendly
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage

from datatools.log import getLogger
from datatools.utils import chunks, remove_punctuation

logger = getLogger()

TIMEOUT = 600

#https://cloud.google.com/bigquery/pricing
GOOGLE_PRICE_PER_BYTE = 5 / 10E12  # $5 per tb.

class GCloud:
    def __init__(self, project_id=None, GOOGLE_JSON_KEY=None):
        if GOOGLE_JSON_KEY:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_JSON_KEY

        self.project_id = project_id
        self.bq_client = None
        self.gcs_client = None

        self.bq_get_clients(project_id=self.project_id)

    def bq_get_clients(self, project_id=None):
        credentials, project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Make clients.
        self.bq_client = bigquery.Client(
            credentials=credentials,
            project=project_id,
        )

        self.gcs_client = storage.Client(
            credentials=credentials,
            project=project_id
        )

    from google.api_core import exceptions
    @backoff.on_exception(backoff.expo, (exceptions.GoogleAPICallError, exceptions.ClientError), max_tries=5)
    def upload(self, uri, data):

        logger.debug(f'Uploading file {uri}.')

        with self.gcs_client.open(uri, 'w', content_type='text/plain') as gcs_file:
            gcs_file.write(data)

        logger.debug(f'Successfully uploaded file {uri} with {len(data)} lines written.')

        return True

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
            "Query stats: Ran in {} seconds, cache hit: {}, billed {}, approx cost ${}.".format(time_taken, cache_hit, bytes_billed, approx_cost))

        if do_not_return_results:
            return True
        else:
            # job.result() blocks until the query has finished.
            results_df = job.result().to_dataframe()
            return results_df


    def upload_rows(self, schema, rows, destination, backup_file_name=None, ensure_schema_compliance=False,
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
                if backup_file_name:
                    logger.increment_run_summary('Failed rows saved to disk', len(chunk))
                    save_file_full = '{}.{}'.format(backup_file_name, index)
                    logger.error(
                        "Failed to upload rows! Saving {} rows to newline delimited JSON file ({}) for later upload.".format(
                            len(rows), save_file_full))

                    try:
                        os.makedirs(os.path.dirname(save_file_full), exist_ok=True)
                    except FileNotFoundError:
                        pass  # We get here if we are saving to a file within the cwd without a full path

                    try:
                        df = pd.DataFrame.from_dict(chunk)
                        df = self.nan_ints(df, convert_strings=True)

                        # TODO: this file naming format does not work on windows.
                        df.to_json(save_file_full, orient="records", lines=True, force_ascii=False)
                        str_error += "Saved {} rows to newline delimited JSON file ({}) for later upload.\n\n".format(
                            len(rows), save_file_full)
                    except Exception as e:
                        str_error += "Unable to save backup file {}: {}\n\n".format(save_file_full, str(e)[:200])

                else:
                    str_error += "No backup save file configured.\n\n"

                message_body = f"Exception pushing to BigQuery table {destination}, chunk {index}.\n\n"
                message_body += str_error

                logger.send_exception(
                    message_body=message_body,
                    subject=f"Error inserting rows to Google Bigquery! Table: {destination}")
                logger.debug("First three rows:")
                logger.debug(bq_rows[:3])

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
                    elif hasattr(d[key], 'dtype'):
                        d[key] = np.asscalar(d[key])
                    elif isinstance(d[key], dict):
                        d[key] = self.scrub_serializable(d[key])
                    elif isinstance(d[key], list):
                        d[key] = [self.scrub_serializable(x) for x in d[key]]
                    elif isinstance(d[key], datetime.date) or isinstance(d[key], datetime.datetime):
                        # ensure dates are stored as strings in ISO format for uploading
                        d[key] = d[key].isoformat()
                    elif isinstance(d[key], uuid.UUID):
                        # if the obj is uuid, we simply return the value of uuid
                        d[key] = d[key].hex
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
