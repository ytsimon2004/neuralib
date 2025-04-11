from unittest.mock import patch

from neuralib.morpho.swc import SwcNode, SwcFile, SwcPlotOptions


def test_identifier_flags():
    node = SwcNode(1, 1, 0, 0, 0, 1.0, -1)
    assert node.is_soma
    assert not node.is_axon
    assert node.identifier_name == 'soma'
    assert node.point.tolist() == [0, 0, 0]


def test_load_swc_from_mock_file(tmp_path):
    swc_content = """
    1 1 0 0 0 2.0 -1
    2 3 1 0 0 1.0 1
    3 4 0 1 0 1.0 1
    """
    swc_path = tmp_path / "test.swc"
    swc_path.write_text(swc_content, encoding='Big5')

    swc = SwcFile.load(swc_path)
    assert len(swc.node) == 3
    assert swc[1].is_soma
    assert swc['dendrite'].node[0].is_basal_dendrite or swc['dendrite'].node[0].is_apical_dendrite


@patch("matplotlib.pyplot.show", return_value=None)
@patch("vedo.Plotter.show", return_value=None)
def test_cli_run_2d_and_3d(mock_vedo, mock_plt, tmp_path, monkeypatch):
    swc_path = tmp_path / "test_complex.swc"
    swc_content = """
    1 1 0.0 0.0 0.0 5.0 -1
    2 2 10.0 0.0 0.0 1.0 1
    3 2 20.0 0.0 0.0 1.0 2
    4 3 -5.0 -5.0 0.0 1.5 1
    5 3 -10.0 -10.0 0.0 1.2 4
    6 3 -15.0 -10.0 0.0 1.0 5
    7 4 0.0 5.0 5.0 1.5 1
    8 4 0.0 10.0 10.0 1.0 7
    9 4 0.0 15.0 15.0 0.8 8
    """
    swc_path.write_text(swc_content, encoding="Big5")

    # 2D
    monkeypatch.setattr("sys.argv", ["swc", str(swc_path), "--radius", "--2d"])
    SwcPlotOptions().main()

    # 3D
    monkeypatch.setattr("sys.argv", ["swc", str(swc_path), "--radius"])
    SwcPlotOptions().main()

    swc_path.unlink()
