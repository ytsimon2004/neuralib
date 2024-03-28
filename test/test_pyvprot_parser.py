import os
import unittest

import numpy as np
from pyvstim import getStimuliTimesFromLog

from neuralib.stimpy.stimpy_pyv import PyVProtocol, StimlogPyVStim, PyVlog
from rscvp.port.wfield_manual import find_nloops_from_prot


class TestPyvProt(unittest.TestCase):
    pyvprot = PyVProtocol
    stim: StimlogPyVStim

    @classmethod
    def setUpClass(cls) -> None:
        f = '/Users/simon/code/Analysis/1p/210302_YW006_1P_YW/run00_onep_retino_circling_squares.log'
        cls.t, cls.pars, cls.opt = getStimuliTimesFromLog(f)

        cls.pyvprot = PyVlog(os.path.dirname(f)).get_prot_file()
        cls.stim = PyVlog(os.path.dirname(f)).stimlog_data()

    def test_pars_header(self):
        self.assertSetEqual(set(self.pars.columns), set(self.pyvprot.visual_stimuli_dataframe.columns))

    def test_pars(self):
        self.assertEqual(self.pars['n'].item(), self.pyvprot['n'])
        self.assertEqual(self.pars['dur'].item(), self.pyvprot['dur'])
        self.assertEqual(self.pars['len'].item(), self.pyvprot['len'])
        self.assertEqual(self.pars['pad'].item(), self.pyvprot['pad'])
        self.assertEqual(self.pars['c'].item(), self.pyvprot['c'])
        self.assertEqual(self.pars['width'].item(), self.pyvprot['width'])
        self.assertEqual(self.pars['xc'].item(), self.pyvprot['xc'])
        self.assertEqual(self.pars['yc'].item(), self.pyvprot['yc'])
        self.assertEqual(self.pars['apwidth'].item(), self.pyvprot['apwidth'])
        self.assertEqual(self.pars['apheight'].item(), self.pyvprot['apheight'])
        self.assertEqual(self.pars['apxc'].item(), self.pyvprot['apxc'])
        self.assertEqual(self.pars['apyc'].item(), self.pyvprot['apyc'])


    def test_stim_time(self):
        f = np.testing.assert_allclose
        # iStim
        f(self.t[:, 0], self.stim.get_time_profile().i_stim)

        # iTrial
        f(self.t[:, 1], self.stim.get_time_profile().i_trial)
        #
        # starttime
        f(self.t[:, 2], self.stim.get_time_profile().get_time_interval()[:, 0])
        #
        # endtime
        f(self.t[:, 3], self.stim.get_time_profile().get_time_interval()[:, 1])

    def test_nloops_from_prot(self):
        self.assertEqual(find_nloops_from_prot(self.pars), self.pyvprot.get_loops_expr().n_cycles)


if __name__ == '__main__':
    unittest.main()
