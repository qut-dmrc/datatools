import uuid

import datetime
import logging
import time
import pytest

from datatools.log import setup_logging, CountsHandler
from datatools.gcloud import GCloud

import google.cloud.logging  # Don't conflict with standard logging

DEBUG_TEXT = "this should not show up in the log" + str(uuid.uuid1())
LOG_TEXT = "logging appears to be working" + str(uuid.uuid1())

@pytest.fixture (scope="session", autouse=True)
def logger():
    logobj = setup_logging(verbose=False)
    yield logobj

    logobj.info('Tearing test logger down.')

def test_error(capsys, logger):
    logger.error(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert 'error' in str.lower(captured.err)

def test_warning(capsys, logger):
    logger.warning(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert 'warning' in str.lower(captured.err)

def test_debug(capsys, logger):
    logger.debug(DEBUG_TEXT)
    captured = capsys.readouterr()
    assert DEBUG_TEXT not in captured.err
    assert 'debug' not in str.lower(captured.err)

def test_info(capsys, logger):
    logger.info(LOG_TEXT)
    captured = capsys.readouterr()
    assert LOG_TEXT in captured.err
    assert 'info' in str.lower(captured.err)

def test_cloud_logger_info():
    # sleep 5 seconds to allow the last class to write to the cloud servide
    time.sleep(5)
    gc = GCloud()
    entries = gc.logging_client.list_entries(order_by=google.cloud.logging.DESCENDING, max_results=100)
    for entry in entries:
        if LOG_TEXT in str(entry.payload):
            return True

    raise IOError(f'Info message not found in log: {LOG_TEXT}')

def test_cloud_loger_debug():
    gc = GCloud()
    entries = gc.logging_client.list_entries(order_by=google.cloud.logging.DESCENDING, max_results=100)
    for entry in entries:
        if DEBUG_TEXT in str(entry.payload):
            raise IOError(f'Debug message found in log: {LOG_TEXT}')


def test_zzz01_send_exception_email(logger):
    try:
        raise Exception("Test exception")
    except Exception as e:
        message_sent = logger.send_exception(module_name="test_logging.py", subject="Test exception email",
                              message_body=f"Send date: {datetime.datetime.utcnow().isoformat()}")
        assert message_sent
    try:
        raise Exception("Test exception")
    except Exception as e:
        message_sent = logger.send_exception(module_name="test_logging.py", subject="Test exception email",
                                                  message_body=f"You should not have received this message. It was sent as a test, and "
                                                               f"the logger should not send more than one message every five minutes. "
                                                               f"Send date: {datetime.datetime.utcnow().isoformat()}")
        assert not message_sent

def test_zzz01_warning_counts(logger):
    logger.warning("test: logging appears to be working.")
    counts = {}
    for handlerobj in logger.handlers:
        if isinstance(handlerobj, CountsHandler):
            counts = handlerobj.get_counts()
            break

    assert counts['WARNING'] == 2
    assert counts['ERROR'] == 3

    # should also keep the count from the info calls above
    assert counts['INFO'] == 2

def test_zzz02_summary_increment(logger):
    logger.increment_run_summary('Test rows saved', 500)
    summary = logger.get_log_summary()
    assert "Test rows saved: 500\n" in summary

    # Check that summary contains errors above (Warning: relies on tests running in alphabetical order.)
    assert "WARNING messages: 2\n" in summary

def test_zzz03_test_log_n(logger):
    for i in range(0, 20):
        logger.log_every_n(f"test log {i}", level=logging.INFO, n=10)

    counts = {}
    for handlerobj in logger.handlers:
        if isinstance(handlerobj, CountsHandler):
            counts = handlerobj.get_counts()
            break

    assert counts['INFO'] == 3
    assert counts['WARNING'] == 2
    assert counts['ERROR'] == 3
