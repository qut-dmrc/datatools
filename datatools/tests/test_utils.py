import requests
from nose.tools import assert_equal, assert_not_equal, assert_raises, assert_true

from utils import fetch_with_backoff


class TestUtils(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """This method is run once for each class before any tests are run"""
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fetch_with_backoff(self):
        url = "http://invalidURL"
        s = requests.session()
        assert_raises(IOError, fetch_with_backoff, s, url, wait_expo_max=4, max_tries=3)



