import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

from neuralib.stimpy import PyVlog


class TestRiglogParser(unittest.TestCase):
    RIGLOG_CONTENT = """
# Version 1.4
# VStim git commit hash: 6e12a4e1.4
# 07-03-2024 - 17:26:41
# Filename: run00_onep_retino_circling_squares.log
# CODES: vstim=10
# VLOG HEADER:code,presentTime,iStim,iTrial,iFrame,blank,contrast,posx,posy,apx,apy,indicatorFlag
# RIG VERSION: 0.3
# RIG GIT COMMIT HASH: af97b40
# CODES: screen=0,imaging=1,position=2,lick=3,reward=4,lap=5,cam1=6,cam2=7,cam3=8,act0=21,act1=22,opto=15
# RIG CSV: code,time received,duino time,value
# STARTED EXPERIMENT
6,608,600.0,18

7,608,600.0,18

8,608,600.0,5

6,639,633.0,19

7,639,633.0,19

6,671,666.0,20

7,671,666.0,20

6,701,700.0,21

7,701,700.0,21

8,717,720.0,6

6,733,733.0,22

7,733,733.0,22

6,763,766.0,23

7,763,766.0,23

6,796,800.0,24

7,796,800.0,24

6,826,833.0,25

7,842,833.0,25

10,720.3551391139627,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

8,842,840.0,7

10,724.6931836707518,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,728.7487314315513,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,735.326271969825,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,739.6519687026739,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,743.7469694996253,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,748.0548971798271,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

6,873,866.0,26

7,873,866.0,26

10,752.329696319066,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,756.518360809423,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,763.2964802905917,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,769.7168104350567,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,773.909691371955,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,778.0492653837427,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

6,904,899.0,27

10,782.3400263441727,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

7,904,900.0,27

10,786.5672406041995,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,790.7001888379455,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,797.2147850086913,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,801.5992097789422,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,805.7806462747976,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,809.9467231659219,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,814.3350630998611,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

6,935,933.0,28

7,935,933.0,28

10,818.6709994915873,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,822.846713825129,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,829.4317836407572,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,833.8439159560949,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,838.2590599358082,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

8,951,959.0,8

10,844.2679926520213,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

6,967,966.0,29

7,967,966.0,29

10,848.4771368093789,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,854.6173793729395,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,860.7702710432932,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,865.0613331701607,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,869.1367580322549,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,873.2332647778094,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

10,877.3144119186327,1,1,-1,1,1,0.0,0.0,50.0,0.0,0

6,997,999.0,30

7,997,999.0,30

# STOPPED EXPERIMENT
    """

    FILEPATH: ClassVar[Path] = Path('.test.log')
    MISSING = np.array([])
    LOG: PyVlog

    @classmethod
    def setUpClass(cls):
        with open(cls.FILEPATH, 'w') as f:
            f.write(cls.RIGLOG_CONTENT)

        cls.LOG = PyVlog(cls.FILEPATH)

    @classmethod
    def tearDownClass(cls):
        cls.FILEPATH.unlink()
        cls.FILEPATH.with_stem('.test_log').with_suffix('.npy').unlink()

    def test_source_version(self):
        self.assertEqual(self.LOG.version, 'pyvstim')

    def test_version(self):
        self.assertEqual(self.LOG.log_config['version'], 0.3)

    def test_commit_hash(self):
        self.assertEqual(self.LOG.log_config['commit_hash'], 'af97b40')

    def test_codes(self):
        res = {
            'screen': 0,
            'imaging': 1,
            'position': 2,
            'lick': 3,
            'reward': 4,
            'lap': 5,
            'cam1': 6,
            'cam2': 7,
            'cam3': 8,
            'act0': 21,
            'act1': 22,
            'opto': 15
        }
        self.assertDictEqual(self.LOG.log_config['codes'], res)

    def test_csv_fields(self):
        res = ('code', 'time received', 'duino time', 'value')
        self.assertTupleEqual(self.LOG.log_config['fields'], res)


if __name__ == '__main__':
    unittest.main()
