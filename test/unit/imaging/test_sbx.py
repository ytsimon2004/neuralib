from neuralib.io.dataset import load_example_scanbox


def test_load_sbx():
    sbx = load_example_scanbox()
    sbx.asdict()
