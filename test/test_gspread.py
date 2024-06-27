import unittest

import gspread
import numpy as np

from neuralib.tools.gspread import *

# replace the following for testing
CFG_FILE = ...  # config filepath for gspread token (i.e., service_account.json)
EMAIL_ADDRESS = ...  # your email address


class TestSpreadSheet(unittest.TestCase):
    sh: GoogleSpreadSheet

    @classmethod
    def setUpClass(cls):
        """create test spreadsheet"""
        gc = gspread.service_account(filename=CFG_FILE)
        sh = gc.create('test_gspread')
        sh.share(EMAIL_ADDRESS, perm_type='user', role='writer')
        sh.add_worksheet('A', rows=10, cols=20)
        sh.add_worksheet('B', rows=20, cols=30)
        sh.del_worksheet(sh.worksheet('Sheet1'))

        #
        cls.sh = GoogleSpreadSheet('test_gspread', service_account_path=CFG_FILE)

    def test_len(self):
        self.assertEqual(len(self.sh), 2)

    # noinspection PyTypeChecker
    def test_contain(self):
        self.assertIn('A', self.sh)
        self.assertNotIn('C', self.sh)

    def test_title(self):
        self.assertEqual('test_gspread', self.sh.title)

    def test_worksheet_list(self):
        self.assertListEqual(['A', 'B'], self.sh.worksheet_list)

    @classmethod
    def tearDownClass(cls):
        cls.sh._client.del_spreadsheet(cls.sh._sheet.id)


class TestWorkSheetCust(unittest.TestCase):
    ws: GoogleWorkSheet

    @classmethod
    def setUpClass(cls):
        cls.ws = GoogleWorkSheet.of('Test_YWAnalysis', 'apcls_tac', service_account_path=CFG_FILE)

    def test_title(self):
        self.assertEqual('apcls_tac', self.ws.title)

    def test_headers(self):
        expect = [
            'Data', 'region', 'pair_wise_group', 'dark end', 'ch2_num', 'depth', 'n_planes',
            'loc_MA', 'loc_MP', 'loc_LP', 'loc_LA', 'rotation', 'Notes', 'TODO', 'Gernal Notes'
        ]
        self.assertListEqual(self.ws.headers, expect)

    def test_index_list(self):
        expect = ['210315_YW006', '210401_YW006', '210402_YW006', '210409_YW006', '210402_YW008', '210407_YW008',
                  '210409_YW008', '210416_YW008', '210604_YW010', '210610_YW010', '#210513_YW017', '210514_YW017',
                  '210519_YW017', '211202_YW022', '211209_YW022', '211203_YW032', '211208_YW032', '211202_YW033',
                  '211208_YW033', '221018_YW048', '221019_YW048', '#TODO', '# BACKUP', '211029_YW022', '## TBA',
                  '210514_YW018', '210518_YW018', '210526_YW018', '####', '220518_YW040', '220520_YW040',
                  '220526_YW040']
        self.assertListEqual(self.ws.index_value, expect)

    def test_get_range_value(self):
        region_a1 = 'B3:B5'
        region_exp = ['aRSC', 'pRSC', 'pRSC']
        self.assertListEqual(self.ws.get_range_value(region_a1), region_exp)

    def test_get_range_value_2d(self):
        loc_a1 = 'H2:K3'
        loc_exp = ['0;-1.17', '0;-2.17', '-0.67;-2.17', '-0.67;-1.17', '0;-1.17', '0;-2.17', '-0.67;-2.17',
                   '-0.67;-1.17']
        self.assertListEqual(self.ws.get_range_value(loc_a1), loc_exp)

    def test_get_value(self):
        exp = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '-5.1',
               '-6.9']
        self.assertListEqual(self.ws.values('rotation'), exp)

    def test_row(self):
        f = self.ws._row
        self.assertEqual(f(0), 2)
        self.assertEqual(f('210315_YW006'), 2)
        self.assertListEqual(f(['210604_YW010', '210610_YW010']), [10, 11])
        np.testing.assert_array_equal(f(np.array([1, 3])), np.array([3, 5]))

    def test_col(self):
        self.assertEqual(self.ws._col('pair_wise_group'), 3)

    def test_get_cells(self):
        v = self.ws.get_cell(data='# test', head='pair_wise_group')
        self.assertEqual(v, '0.5')

        f = self.ws.get_cell(data='# test', head='pair_wise_group', value_render_option='FORMULA')
        self.assertEqual(f, '=100/200')


if __name__ == '__main__':
    unittest.main()
