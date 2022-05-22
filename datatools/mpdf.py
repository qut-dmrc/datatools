"""
    Class to preprocess in parallel a dataset broken into a queue of dataframe chunks.

    Expects:
        input_df: Pandas Dataframe
        func: a function to process the df and return a dataframe
                optionally, do not pass a function and overwrite process()

    Returns:
        output_df: joined resulting dataframes
"""
import time
import multiprocessing as mp

import math

from log import getLogger
import pandas as pd


class DFProcessor():
    def __init__(self, concurrency=8, batch_size=None):
        self.concurrency = concurrency
        self.batch_size = batch_size
        self.func = self.process
        self.logger = getLogger()

    def process_mp(self, df, func=None):
        if func:
            self.func = func

        if not self.batch_size:
            self.batch_size = int(math.ceil(df.shape[0] / self.concurrency))
        list_df = [df[i:i + self.batch_size] for i in range(0, df.shape[0], self.batch_size)]
        num_chunks = len(list_df)
        self.logger.debug(f'Chunked input dataframe into {num_chunks} chunks.')

        with mp.Pool(processes=self.concurrency) as pool:
            results = pool.map(self.time_wrapper, list_df)

        df = pd.concat(results)

        self.logger.info(f'Finished processing dataframe.')

        return df

    def time_wrapper(self, df):
        t0 = time.perf_counter()
        df = self.func(df)
        time_taken = time.perf_counter() - t0
        self.logger.debug(f"Preprocessor finished dataframe chunk in {time_taken}.")

        return df

    def process(self, df):
        """ Overwrite this method. Do something and return a dataframe. """
        return df
