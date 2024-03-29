from neuralib.atlas.brainrender.core import BrainReconstructor

brender = BrainReconstructor()
brender.main((['--source=allen_mouse_25um', '--region=VISal,VISam,VISl,VISli,VISp,VISpl,VISpm,VISpor'], []))
