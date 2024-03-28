"""
Test if customized parsing method show the same result from Joao's labcam repo
"""
import unittest
from typing import Any

import numpy as np
from pyvstim import parseVStimLog

from neurolib.stimpy.camlog import LabCamlog
from neurolib.stimpy.stimpy_pyv import PyVlog
from rscvp.port.wfield_manual import parse_camlog, get_camera_time  # TODO


class TestCamlogParse(unittest.TestCase):
    cam_log_file = ...
    pyv_log_file = ...

    @classmethod
    def setUpClass(cls) -> None:
        cls.prev_camlog: dict[str, Any] = parse_camlog(cls.cam_log_file)[0]
        cls.camlog = LabCamlog.load(cls.cam_log_file)

    def test_camlog(self):
        np.testing.assert_equal(self.prev_camlog['frame_id'].to_numpy(), self.camlog.frame_id)
        np.testing.assert_equal(self.prev_camlog['timestamp'].to_numpy(), self.camlog.timestamp)

    def test_camlog_time(self):
        """need sync between pvstim log and camlog file"""
        # old
        plog = parseVStimLog(self.pyv_log_file)[0]
        t1 = get_camera_time([self.cam_log_file], plog)
        # new
        pvlog = PyVlog(self.pyv_log_file.parent)
        t2 = LabCamlog.load(self.cam_log_file).get_camera_time(pvlog)

        np.testing.assert_allclose(t1, t2)


if __name__ == '__main__':
    unittest.main()
