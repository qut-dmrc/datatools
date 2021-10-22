import unittest


### Google cloud authentication
GOOGLE_JSON_KEY = "e:/src/nicanalysis-c0df8a860fb.json"

class MyTestCase(unittest.TestCase):
    def test_imports(self):
        from datatools.log import getLogger
        from datatools.gcloud import GCloud

        logger = getLogger()
        logger.debug('Logger test debug')

        gc = GCloud(GOOGLE_JSON_KEY=GOOGLE_JSON_KEY)
        df = gc.run_query('SELECT True')
        self.assertEqual(True, df.iloc[0, 0])


if __name__ == '__main__':
    unittest.main()
