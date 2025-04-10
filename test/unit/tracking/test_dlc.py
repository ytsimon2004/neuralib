import pytest
from polars.polars import ColumnNotFoundError
from polars.testing import assert_frame_equal

from neuralib.io.dataset import load_example_dlc_h5, load_example_dlc_csv
from neuralib.tracking import DeepLabCutDataFrame


@pytest.fixture()
def h5_data():
    return load_example_dlc_h5(cached=True)


@pytest.fixture()
def csv_data():
    return load_example_dlc_csv(cached=True)


def test_parse(h5_data: DeepLabCutDataFrame, csv_data: DeepLabCutDataFrame):
    h5_df = h5_data.dataframe()
    csv_df = csv_data.dataframe().select(h5_df.columns)
    assert_frame_equal(h5_df, csv_df)


def test_default_filtered(csv_data: DeepLabCutDataFrame):
    assert not csv_data.default_filtered


def test_model_config(csv_data: DeepLabCutDataFrame):
    assert isinstance(csv_data.model_config, dict)


def test_list_joint(csv_data: DeepLabCutDataFrame):
    assert csv_data.joints == [
        'Nose', 'EarL', 'EarR', 'Neck', 'PawFL', 'PawFR',
        'Body', 'PawBL', 'PawBR', 'TailBase', 'TailMid', 'TailEnd'
    ]


def test_get_joints(csv_data: DeepLabCutDataFrame):
    with pytest.raises(ColumnNotFoundError):
        csv_data.get_joint('nose')

    assert csv_data.get_joint('Nose').columns == ['Nose_x', 'Nose_y', 'Nose_likelihood']
