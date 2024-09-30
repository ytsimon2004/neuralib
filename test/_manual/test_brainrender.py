from neuralib.atlas.brainrender import RoisReconstructor, ProbeReconstructor
from neuralib.io.dataset import google_drive_file


def test_brainrender_rois():
    with google_drive_file('1cf2r3kcqjENBQMe8YzBQvol8tZgscN4J') as file:
        class Test(RoisReconstructor):
            csv_file = file

        Test().main()


def test_brainrender_npx2():
    with google_drive_file('1fRvMNHhGgh5KP3CgGm6CMFth1qIAmwfh') as file:
        class Test(ProbeReconstructor):
            csv_file = file
            implant_depth = 3000
            plane_type = 'sagittal'
            regions = ('ENT',)
            hemisphere = 'left'

        Test().main()


if __name__ == '__main__':
    test_brainrender_npx2()
