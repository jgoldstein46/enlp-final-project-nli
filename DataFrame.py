import pandas as pd

# wrapper class for NLI dataframe

class DataFrame:
    def __init__(self, df):
        self.df = df
        self.english_df = None
        self.create_english()

    def create_english(self):
        self.english_df = self.df[self.df['lang_abv'] == 'en']