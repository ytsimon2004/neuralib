from pathlib import Path

from neuralib.util.csv import csv_header


def test_quotes_csv():
    fields = ['idx', 'note', 'note2']
    literal_a = ['hello',
                 "what's happen, bro?"]
    literal_b = ['yo',
                 ""]

    with csv_header(Path('test.csv'), fields, quotes_header=['note', 'note2']) as csv:
        for i in range(2):
            csv(i, literal_a[i], literal_b[i])


if __name__ == '__main__':
    test_quotes_csv()
