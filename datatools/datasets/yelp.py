import pandas as pd


def BlackOwnedBusinesses():
    ### This dataset was scraped from yelp in July 2022
    # URL: 'https://www.yelp.com/collection/FXDOTfw9y7LhSA_VZbpAmg/Black-Owned-Businesses-Collection'

    df = pd.read_json('gs://dmrc-platforms/yelp/Black_owned_businesses.json', orient='records', lines=True)
    df['link'] = 'https://yelp.com' + df['link']
    return df