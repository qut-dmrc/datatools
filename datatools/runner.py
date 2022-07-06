import datetime

import inspect
import sys

import os
import pandas as pd

from datatools.gcloud import GCloud
from datatools.log import getLogger

class ProcessingFinished(Exception):
    """Jobs done."""
    pass

class FatalError(Exception):
    ## Something has gone horribly wrong and the process must terminate.
    pass

class Delay(Exception):
    ## Wait before running again
    pass

RUNS_SCHEMA = """
[{"name":"runtime","type":"TIMESTAMP","mode":"REQUIRED"},{"name":"name","type":"STRING","mode":"REQUIRED"},{"name":"arguments","type":"STRING","mode":"NULLABLE"},{"name":"task","type":"STRING","mode":"NULLABLE"},{"name":"meta","type":"STRING","mode":"NULLABLE","description":"A serialised representation of a dict with extra fields"},{"name":"successful","type":"BOOLEAN","mode":"NULLABLE"}]
"""
RUNS_TABLE = "observatory-158104.scrapers.runs"
#####
#
# decorators
#
#####

# Run the function only if sufficient time has elapsed
def only_run(func):
    def inner(*args, **kwargs):
        min_time = datetime.timedelta(seconds=35)
        name = os.path.basename(sys.argv[0])
        task = inspect.stack()[1].function  # calling method
        calling_obj = args[0]

        arguments = [ str(a) for a in args[1:] ]
        for k, v in kwargs.items():
            arguments.append(f'{k}={v}')
        arguments = ",".join(arguments)

        last_run_time = calling_obj._last_run_time(name=name, arguments=arguments, task=task)
        if last_run_time:
            elapsed = datetime.datetime.utcnow() - last_run_time
            if elapsed > min_time:
                # do not run, not enough time has elapsed
                raise Delay(f'Elapsed time for {name} {task} {args} is only {elapsed}, not running.')

        result = func(*args[1:], **kwargs)

        # log successful run since we didn't get an error.
        calling_obj._log_run(name, arguments, task, successful=True)

        return result

    return inner

class Runner:
    def __init__(self, **kwargs):
        self.results = pd.DataFrame()
        self.gCloud = GCloud()
        self.logger = getLogger()
        self.dataset = kwargs.get('dataset')
        self.schema = kwargs.get('schema')

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save out if we have results still
        if self.results:
            save_path = self.save_results()
            self.logger.error(f'Runner emergency save triggered. Saved to: {save_path}.')

    def save_results(self, **params):
        params['destination'] = self.dataset
        params['schema'] = self.schema
        params['data'] = params.get('data', params.get('results', self.results))
        return self.gCloud.save(**params)

    def _last_run_time(self, name, arguments, task):
        sql = f"""
            SELECT runtime, r.name, arguments, task, meta, successful 
            FROM `{RUNS_TABLE}` r
            
            WHERE DATE(runtime) > DATETIME_SUB(CURRENT_DATETIME, INTERVAL 7 day)
            AND r.name = '{name}' AND arguments = '{arguments}' AND task = '{task}' 
            ORDER BY runtime DESC LIMIT 1
            """

        df = self.gCloud.run_query(sql)
        last_run_time = None
        try:
            last_run_time = pd.to_datetime(df.iloc[0]['runtime'].value)
        except IndexError as e:
            # no data
            pass

        return last_run_time

    def _log_run(self, name, arguments, task, **kwargs):
        # log a successful update
        run_id = self.logger.run_id

        record = dict(name=name, arguments=str(arguments), task=task, runtime=datetime.datetime.utcnow().isoformat(), run_id=run_id)
        record.update(kwargs)
        self.gCloud.upload_rows(schema=RUNS_SCHEMA, rows=[record], destination=RUNS_TABLE)

