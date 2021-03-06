import datetime
import inspect
import itertools
import logging
import os
import platform
import psutil
import socket
import sys
import threading
import timeit
from datetime import timedelta
from traceback import format_exc
import humanfriendly
from requests import post
import coloredlogs
import google.cloud.logging  # Don't conflict with standard logging
from google.cloud.logging.handlers import CloudLoggingHandler


EXCEPTION_LIMIT = 25 # If we reach this many exceptions, something is clearly wrong, and we should stop the scraper.
EXCEPTION_TIMEOUT = 4 * 60 * 60 # Expire exceptions after four hours
_LOGGER_NAME = 'DMRCLogger'
MIN_MINUTES_BETWEEN_EMAILS=5

class DMRCLogger(logging.Logger):
    def __init__(self, name):
        super(DMRCLogger, self).__init__(name)
        self.run_summary = {
            'summary_log_messages': [],
            'summary_counts': {},
        }
        # Create a unique identifier for this run
        node_name = platform.uname().node
        username = psutil.Process().username()
        run_time = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.run_id = f'{run_time}-{username}-{node_name}'

        self.exceptions = []
        self.last_email = None
        self.already_setup = False

        self.short_name = _LOGGER_NAME

        # Keeps track of the last log time of the given token.
        self._log_timer_per_token = {}

        # Counter to keep track of number of log entries per token.
        self._log_counter_per_token = {}

        self.mailgun_config = dict()

    def _get_next_log_count_per_token(self, token):
        """Wrapper for _log_counter_per_token. Thread-safe.
        Args:
          token: The token for which to look up the count.
        Returns:
          The number of times this function has been called with
          *token* as an argument (starting at 0).
        """
        # Can't use a defaultdict because defaultdict isn't atomic, whereas
        # setdefault is.
        return next(self._log_counter_per_token.setdefault(token, itertools.count()))

    def log_every_n(self, msg=None, n=100, level=logging.INFO, *args):
        """Logs 'msg % args' at level 'level' once per 'n' times.
        Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
        Not threadsafe.
        Args:
          level: int, the logging level at which to log.
          msg: str, the message to be logged.
          n: int, the number of times this should be called before it is logged.
          *args: The args to be substituted into the msg.
        """
        count = self._get_next_log_count_per_token(self.findCaller())
        self.log_if(level, msg, not (count % n), *args)

    def _seconds_have_elapsed(self, token, num_seconds):
        """Tests if 'num_seconds' have passed since 'token' was requested.
        Not strictly thread-safe - may log with the wrong frequency if called
        concurrently from multiple threads. Accuracy depends on resolution of
        'timeit.default_timer()'.
        Always returns True on the first call for a given 'token'.
        Args:
          token: The token for which to look up the count.
          num_seconds: The number of seconds to test for.
        Returns:
          Whether it has been >= 'num_seconds' since 'token' was last requested.
        """
        now = timeit.default_timer()
        then = self._log_timer_per_token.get(token, None)
        if then is None or (now - then) >= num_seconds:
            self._log_timer_per_token[token] = now
            return True
        else:
            return False

    def log_if(self, level, msg, condition, *args):
        """Logs 'msg % args' at level 'level' only if condition is fulfilled."""
        if condition:
            self.log(level, msg, *args)

    def log_every_n_seconds(self, msg=None, n_seconds=60, level=logging.INFO, *args):
        """Logs 'msg % args' at level 'level' iff 'n_seconds' elapsed since last call.
        Logs the first call, logs subsequent calls if 'n' seconds have elapsed since
        the last logging call from the same call site (file + line). Not thread-safe.
        Args:
          level: int, the absl logging level at which to log.
          msg: str, the message to be logged.
          n_seconds: float or int, seconds which should elapse before logging again.
          *args: The args to be substituted into the msg.
        """
        should_log = self._seconds_have_elapsed(self.findCaller(), n_seconds)
        self.log_if(level, msg, should_log, *args)


    def init_run_summary(self):
        self.run_summary = {
            'summary_log_messages': [],
            'summary_counts': {},
        }

    def increment_run_summary(self, variable_name, value=1):
        self.run_summary['summary_counts'][variable_name] = self.run_summary['summary_counts'].get(variable_name, 0) + value

    def log_run_summary(self, summary_message, module_name=None, **kwargs):
        if module_name:
            summary_message = "[{}] {}\n".format(module_name, summary_message)
        for key, value in kwargs.items():
            summary_message += f' {str(key)}: {str(value)}'
        self.run_summary['summary_log_messages'].append(summary_message)
        self.info(summary_message)

    def get_log_summary(self, reset_summary=False):
        message_body = ""

        for line in self.run_summary['summary_log_messages']:
            message_body += str(line) + "\n"

        for key, value in self.run_summary['summary_counts'].items():
            if key == 'BigQuery Bytes Billed':
                message_body += "{key}: {value}\n".format(key=key, value=humanfriendly.format_size(value))
            else:
                message_body += "{key}: {value}\n".format(key=key, value=value)

        for handlerobj in self.handlers:
            if isinstance(handlerobj, CountsHandler):
                counts = handlerobj.get_counts()
                message_body += "\n\nLog messages:\n"
                for key, value in counts.items():
                    message_body += "{key} messages: {value}\n".format(key=key, value=value)
                if reset_summary:
                    handlerobj.reset_counts()

        if reset_summary:
            self.init_run_summary()

        return message_body

    def print_run_summary(self, subject=None, log_file=None, reset_summary=True, send_email=False):
        message_body = self.get_log_summary(reset_summary=reset_summary)

        self.info('\n\n')
        self.info('----------------------')
        self.info(subject)
        self.info(message_body)
        self.info('----------------------')

        if send_email:
            self.send_update_mail(subject, message_body)

        if reset_summary:
            self.init_run_summary()  # Not sure why the run summary is not being reset in the get_log_summary() function.


    def send_exception(self, module_name=None, subject="(unspecified)", message_body=None, print_exc_info=True):
        self.exceptions.append({'msg': subject, 'time': datetime.datetime.utcnow()})

        num_exceptions = len([exc for exc in self.exceptions if
                           exc['time'] > (datetime.datetime.utcnow() - datetime.timedelta(seconds=EXCEPTION_TIMEOUT))])

        if num_exceptions > EXCEPTION_LIMIT:
            # just quit -- something's majorly wrong.
            subject = f"{self.short_name} Fatal errors -- too many exceptions."
            body_msg = f"We hit more than {EXCEPTION_LIMIT} errors in this run. Run has been terminated.\n\nLast error was:\n\n" + message_body
            self.send_update_mail(subject, message_body)
            self.error(body_msg)
            sys.exit()

        if message_body:
            self.exception(message_body, exc_info=print_exc_info)

        if self.last_email:
            if (self.last_email-datetime.datetime.utcnow())/timedelta(minutes=1)<MIN_MINUTES_BETWEEN_EMAILS:
                return False

        self.last_email = datetime.datetime.utcnow()
        if message_body:
            message_body = message_body[:10000]

        try:
            hostname = socket.gethostname()
            full_address = socket.gethostbyname_ex(hostname)
            message_body = f'Unexpected Error, please check your instance {full_address}. {message_body}\n{format_exc()}'
            subject = "[{}] Unexpected Error: {}".format(module_name, subject)
            self.send_mail({
                'subject': subject,
                'text': message_body
            })
        except Exception as e:
            self.error("Unable to send exception mail: {}".format(e))

        return True

    def send_update_mail(self, subject, message):
        try:
            hostname = socket.gethostname()
            full_address = socket.gethostbyname_ex(hostname)
            self.send_mail({
                "subject": subject,
                "text": str(message) + "\n" +
                        "From {host}.".format(
                            host=full_address
                        )
            }
            )
        except Exception as e:
            self.exception("Unable to send error mail: {}".format(e), exc_info=True)

    def send_mail(self, message_dict):

        data = {
            "from": "DMRC Python Data Tools <%s>" % self.mailgun_config['mailgun_smtp_login'],
            "to": self.mailgun_config['notify_email']
        }

        data.update(message_dict)
        self.info(f"Sending email to {self.mailgun_config['notify_email']}.")
        return post(self.mailgun_config['mailgun_api_base_url'], auth=self.mailgun_config['mailgun_auth'], data=data)

class CountsHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        self.counts = {
        }
        self._countLock = threading.Lock()

        super(CountsHandler, self).__init__(level)

    def emit(self, record):
        self._countLock.acquire()
        self.counts[record.levelname] = self.counts.get(record.levelname, 0) + 1
        self._countLock.release()
        pass

    def get_counts(self):
        return self.counts

    def reset_counts(self):
        self.counts = {}


def getLogger():
    if not _LOGGER_NAME in logging.root.manager.loggerDict:
        # logger not yet initialised.
        original_logger_class = logging.getLoggerClass()

        # set up our custom logging class
        logging.setLoggerClass(DMRCLogger)
        logger = logging.getLogger(_LOGGER_NAME)

        # restore the default logging class
        logging.setLoggerClass(original_logger_class)

    return logging.getLogger(_LOGGER_NAME)


def setup_logging(name=None, verbose=False, job="Unknown"):
    logger = getLogger()

    if logger and logger.already_setup:
        logger.warning(f"setup_logging() called but logger is already initialised. Called by {inspect.stack()[1]}")
        return logger
    if not verbose:
        # Quieten other loggers down a bit (particularly requests and google api client)
        for logger_str in logging.Logger.manager.loggerDict:
            try:
                logging.getLogger(logger_str).setLevel(logging.WARNING)
            except:
                pass
    else:
        logger.debug('Set loglevel to DEBUG.')
        for logger_str in logging.Logger.manager.loggerDict:
            try:
                logging.getLogger(logger_str).setLevel(logging.INFO)
            except:
                pass

    logger = getLogger()

    console_format = "%(asctime)s %(hostname)s [%(filename)-20.20s:%(lineno)-4.4s - %(funcName)-20.20s() [%(threadName)-12.12s] [%(levelname)-8.8s]  %(message)s"
    # try some new formats
    console_format = '%(asctime)s %(hostname)s %(name)s %(filename).20s[%(lineno)4d] %(levelname)s %(message)s'
    if not verbose:
        coloredlogs.install(level='INFO', logger=logger, reconfigure=True, fmt=console_format)
    else:
        coloredlogs.install(level='DEBUG', logger=logger, reconfigure=True, fmt=console_format)

    # Add logger to count number of errors
    countsHandler = CountsHandler()
    logger.addHandler(countsHandler)

    # Import in this function because it's not available until after initialisation
    from datatools.gcloud import GCloud
    gCloud = GCloud()

    node_name = platform.uname().node
    username = psutil.Process().username()
    script_name = os.path.basename(sys.argv[0])
    if not name or name is None:
        name = script_name

    labels = {
            "function_name": name,
            "logs": node_name,
            "user": username,
            }

    # Labels for cloud logger
    resource = google.cloud.logging.Resource(
        type="generic_task",
        labels={
            "project_id": 'dmrc-platforms',
            "location": 'us-central1',
            "namespace": name,
            "job": job,
            "task_id": logger.run_id
        },
    )

    cloudHandler = CloudLoggingHandler(gCloud.logging_client, resource=resource, name=name, labels=labels)

    if verbose:
        cloudHandler.setLevel(logging.DEBUG)
    else:
        cloudHandler.setLevel(logging.INFO)

    logger.addHandler(cloudHandler)
    logger.already_setup = True

    logger.info(f"Logging setup, ready for data collection, saving log to Google Cloud Logs ({resource}, {name}).")

    return logger

