"""
Test if customized parsing method show the same result from Joao's pyvstim repo
"""
import unittest
from typing import Any

import numpy as np
from pyvstim.utils import parseVStimLog

from neuralib.stimpy.stimpy_pyv import PyVlog


class TestPvStimParse(unittest.TestCase):
    prev_data: dict[str, Any]
    pyvlog: PyVlog

    @classmethod
    def setUpClass(cls) -> None:
        log_file = ...
        cls.prev_data: dict[str, Any] = parseVStimLog(log_file)[0]
        cls.pyvlog = PyVlog(log_file.parent)

        import matplotlib.pyplot as plt
        t = cls.pyvlog.stimlog_data().v_present_time
        ind = cls.pyvlog.stimlog_data().v_indicator_flag
        apx = cls.pyvlog.stimlog_data().v_ap_x
        plt.plot(t, ind, t, apx)
        plt.show()

    def debug_plot(self, old, new):
        import matplotlib.pyplot as plt
        plt.plot(old, label='old', alpha=0.5)
        plt.plot(new, label='new', alpha=0.5)
        plt.legend()
        plt.show()

    def print_diff(self, old, new):
        if np.array_equal(old, new):
            print('pass')
        else:
            indices = np.where(old != new)
            different_values_old = old[indices]
            different_values_new = new[indices]
            print("Indices of differences:", indices)
            print("Different values (old):", different_values_old)
            print("Different values (new):", different_values_new)
            print("value subtraction:", different_values_new - different_values_old)

    def test_main_log(self):

        def get_prev_data(code: str, header: str) -> np.ndarray:
            return self.prev_data[code][header].to_numpy()

        at = np.testing.assert_allclose  # rtol=1e-07

        at(get_prev_data('vstim', 'presentTime'), self.pyvlog.stimlog_data().v_present_time)
        at(get_prev_data('vstim', 'iStim'), self.pyvlog.stimlog_data().v_stim)
        at(get_prev_data('vstim', 'iTrial'), self.pyvlog.stimlog_data().v_trial)
        at(get_prev_data('vstim', 'iFrame'), self.pyvlog.stimlog_data().v_frame)
        at(get_prev_data('vstim', 'blank'), self.pyvlog.stimlog_data().v_blank)
        at(get_prev_data('vstim', 'contrast'), self.pyvlog.stimlog_data().v_contrast)
        at(get_prev_data('vstim', 'posx'), self.pyvlog.stimlog_data().v_pos_x)
        at(get_prev_data('vstim', 'posy'), self.pyvlog.stimlog_data().v_pos_y)
        at(get_prev_data('vstim', 'apx'), self.pyvlog.stimlog_data().v_ap_x)
        at(get_prev_data('vstim', 'apy'), self.pyvlog.stimlog_data().v_ap_y)
        at(get_prev_data('vstim', 'indicatorFlag'), self.pyvlog.stimlog_data().v_indicator_flag)
        at(get_prev_data('vstim', 'duinotime') / 1000, self.pyvlog.stimlog_data().v_duino_time)

        at(get_prev_data('screen', 'duinotime') / 1000, self.pyvlog.screen_event.time)
        at(get_prev_data('screen', 'value'), self.pyvlog.screen_event.value)

        at(get_prev_data('position', 'duinotime') / 1000, self.pyvlog.position_event.time)
        at(get_prev_data('position', 'value'), self.pyvlog.position_event.value)

        at(get_prev_data('lick', 'duinotime') / 1000, self.pyvlog.lick_event.time)
        at(get_prev_data('lick', 'value'), self.pyvlog.lick_event.value)

        at(get_prev_data('reward', 'duinotime') / 1000, self.pyvlog.reward_event.time)
        at(get_prev_data('reward', 'value'), self.pyvlog.reward_event.value)

        at(get_prev_data('lap', 'duinotime') / 1000, self.pyvlog.lap_event.time)
        at(get_prev_data('lap', 'value'), self.pyvlog.lap_event.value)

        at(get_prev_data('cam1', 'duinotime') / 1000, self.pyvlog.camera_event['facecam'].time)
        at(get_prev_data('cam1', 'value'), self.pyvlog.camera_event['facecam'].value)

        at(get_prev_data('cam2', 'duinotime') / 1000, self.pyvlog.camera_event['eyecam'].time)
        at(get_prev_data('cam2', 'value'), self.pyvlog.camera_event['eyecam'].value)

        at(get_prev_data('cam3', 'duinotime') / 1000, self.pyvlog.camera_event['1P_cam'].time)
        at(get_prev_data('cam3', 'value'), self.pyvlog.camera_event['1P_cam'].value)


if __name__ == '__main__':
    unittest.main()
