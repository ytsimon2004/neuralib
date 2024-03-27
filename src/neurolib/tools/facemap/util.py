import numpy as np

__all__ = ['sync_facemap_rig']


def sync_facemap_rig(track_res: np.ndarray, camera_time: np.ndarray) -> np.ndarray:
    """
    Sync the frame number in facemap tracking result & rig camera event pulse
    :param track_res: (nF, nK) | (nF,)
    :param camera_time: (nFc, ) number of camera's frame
    :return:
    """
    n_track_data = track_res.shape[0]
    n_cam_pulse = len(camera_time)
    if n_track_data > n_cam_pulse:
        diff = n_track_data - n_cam_pulse
        return track_res[:-diff]
    raise NotImplementedError('')
