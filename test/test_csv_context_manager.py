import unittest
from pathlib import Path
from typing import ClassVar

import polars as pl
import polars.testing

from neuralib.util.csv import csv_header


class TestCSVContextManager(unittest.TestCase):
    FILE: ClassVar[Path] = Path('.test.csv')

    def test_basic_usage(self):
        fields = ['neuron_id', 'variable_a', 'variable_b']
        total_neurons = 3
        val_a = [v * 0.5 for v in range(total_neurons)]
        val_b = [v * 2 for v in range(total_neurons)]

        with csv_header(self.FILE, fields) as csv:
            for neuron in range(total_neurons):
                csv(neuron, val_a[neuron], val_b[neuron])

        df = pl.read_csv(self.FILE)

        ans = pl.DataFrame(
            {
                f'{fields[0]}': pl.Series([i for i in range(total_neurons)]),
                f'{fields[1]}': pl.Series(val_a),
                f'{fields[2]}': pl.Series(val_b)
            }
        )

        pl.testing.assert_frame_equal(df, ans)

        self.FILE.unlink()

    def test_double_quotes_usage(self):
        fields = ['neuron_id', 'variable_a', 'variable_b']
        total_neurons = 3
        val_a = [v * 0.5 for v in range(total_neurons)]
        val_b = ['0.1,0.5,0.8', '0.7,0.7', '0.8,0.8']

        with csv_header(self.FILE, fields, quotes_header='variable_b') as csv:
            for neuron in range(total_neurons):
                csv(neuron, val_a[neuron], val_b[neuron])

        df = pl.read_csv(self.FILE)

        ans = pl.DataFrame(
            {
                f'{fields[0]}': pl.Series([i for i in range(total_neurons)]),
                f'{fields[1]}': pl.Series(val_a),
                f'{fields[2]}': pl.Series(val_b)
            }
        )

        pl.testing.assert_frame_equal(df, ans)

        self.FILE.unlink()


if __name__ == '__main__':
    unittest.main()
