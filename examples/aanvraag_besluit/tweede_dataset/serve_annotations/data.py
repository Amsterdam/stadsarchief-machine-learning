import csv

import pandas as pd

# SRC_FILE = '../src/alle_aanvragen_en_besluiten_na_1980_HK-annotated.csv'
SRC_FILE = '../src/alle_aanvragen_en_besluiten_na_1980_HK-annotated.csv'
TARGET_FILE = '../src/alle_aanvragen_en_besluiten_na_1980_HK-annotated.csv'


class AnnotationData(object):

    def load_source(self):
        csv_path = SRC_FILE
        self.df = pd.read_csv(csv_path)
        return self.df

    def save(self):
        self.df.to_csv(TARGET_FILE,
                       index=False,
                       quoting=csv.QUOTE_ALL,
                       sep=',',
                       line_terminator='\r\n',  # windows style
                       encoding='utf-8')

    def set_row_type(self, index, value):
        print('inserting: ' + str(value))
        self.df.at[index, 'document_type'] = value
        print(self.df.iloc[index])

    def get_json(self):
        return self.df.to_json(orient='records')

# data = AnnotationData()
# data.load_source()
# data.save()

