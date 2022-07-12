######
# This module fetches the full text SCOTUS dataset from the PACER project
#
# Using data from https://www.courtlistener.com/api/bulk-info/
# Downloaded from https://www.courtlistener.com/api/bulk-data/courts/all.tar.gz
#
# The dataset has been uploaded by Nic Suzor to GCS in May 2022:
# gs://platform_datasets_hot/judicial/scotus_pacer.json
#
######

import gcsfs
import re
from datatools.log import setup_logging
from datatools.gcloud import GCloud
import pandas as pd
from lxml import etree

logger = setup_logging()

DATA_SCOTUS = 'gs://dmrc-platforms/judicial/scotus_pacer.json'


class SCOTUS(pd.DataFrame):
    def fetch(self):
        # init cloud client
        GCloud()

        # fetch from storage
        df = pd.read_json(DATA_SCOTUS, orient='records', lines=True)

        try:
            import mapply
            mapply.init(n_workers=-1, chunk_size=1)
            df['judgments'] = df['html'].mapply(self.extract_judgments)
        except ImportError:
            logger.info('This would go a lot faster if you installed mapply (pip install mapply)')
            df['judgments'] = df['html'].apply(self.extract_judgments)

        # next, we want to expand the judgments by joining the column to the dataframe
        cases = df.judgments.explode().fillna({}).to_frame()
        cases = cases['judgments'].apply(pd.Series, dtype='object')

        # join the case decisions back up to the main dataframe
        df = pd.concat([df, cases], axis=1)

        return df

    @staticmethod
    def _judge_regex():
        # return a compiled regex that can be used to check whether a line is likely to be
        # introducing a judicial opinion
        patterns = ['^.{0,50}(opinion.{10}justice|certiorari|per curiam|decree|delivered the opinion).{0,30}$',
                    'justice.{1,30}(delivered|statement|the opinion|dissent|subscribe|judgment|concur|decision|joins).{0,30}$',
                    '^.{0,10}justice.{0,20}$',
                    '.*justice.*(dissent|concur).{0,10}$'
                    ]
        return re.compile('|'.join(patterns), re.IGNORECASE)

    def extract_judgments(self, html_text):
        if not html_text or not isinstance(html_text, str):
            return None

        html_text = str.lower(html_text)

        htmlparser = etree.HTMLParser()

        # try and build a xml tree from the html text
        tree = etree.fromstring(html_text, htmlparser)

        judgments = []

        if tree is not None:

            # first, find all the lines about justices.
            # This is a useful cheatsheet for xpath selectors: https://devhints.io/xpath
            judges = tree.xpath('//p[contains(text(),"justice")]')
            judges = [j.text for j in judges]

            for i, judge in enumerate(judges):
                # For each potential judgment

                try:
                    # first, make sure it is likely an opinion introduction
                    pattern = self._judge_regex()
                    if not pattern.search(judge):
                        continue
                except TypeError as e:
                    logger.debug(f'Could not parse {judge} with regex: {e}')

                # get all the paragraphs within elements that are numbered divs
                # by counting only parapraph after each Justice is introduced
                decision = tree.xpath(
                    f'//div[contains(@class,"num") and count(preceding::p[contains(text(),"justice")])={i + 1}]/p')
                paragraphs = [d.text for d in decision]

                # Create a dictionary with the judge's identifying text and all their sentences
                judgment = dict(judge=judge, paragraphs=paragraphs)
                judgments.append(judgment)

            if not judgments:
                # This expression gets all the numbered paragraphs available.
                decision = tree.xpath('//div[contains(@class,"num")]')
                paragraphs = [d.text for d in decision]

                judge = tree.xpath(
                    '//p[contains(text(),"certiorari") or contains(text(),"curiam") or contains(text(),"decree")]')
                if judge:
                    judge = judge[0].text
                else:
                    judge = 'unknown'

                judgment = dict(judge=judge, paragraphs=paragraphs)
                judgments.append(judgment)

            return judgments

        else:
            return None
