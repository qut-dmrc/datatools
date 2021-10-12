import unittest


class MyTestCase(unittest.TestCase):
    def test_imports(self):
        from datatools.log import getLogger
        from datatools.gcloud import GCloud

        logger = getLogger()
        logger.debug('Logger test debug')

        gc = GCloud()
        df = gc.run_query('SELECT True')
        self.assertEqual(True, df.loc[0,0])


if __name__ == '__main__':
    unittest.main()
