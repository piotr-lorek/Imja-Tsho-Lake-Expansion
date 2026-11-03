import numpy as np

from glacial_lake.indices import mndwi, ndwi, normalized_difference


def test_normalized_difference_values():
    a = np.array([0.3, 0.1, 0.2])
    b = np.array([0.1, 0.3, 0.2])
    np.testing.assert_allclose(normalized_difference(a, b), [0.5, -0.5, 0.0])


def test_normalized_difference_is_zero_where_undefined():
    zeros = np.zeros(3)
    result = normalized_difference(zeros, zeros)
    np.testing.assert_array_equal(result, 0.0)
    assert np.isfinite(result).all()


def test_normalized_difference_is_bounded():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-1, 1, (2, 1000))  # includes non-physical negative values
    result = normalized_difference(a, b)
    assert result.min() >= -1.0 and result.max() <= 1.0


def test_index_band_choice():
    bands = {"B03": np.array([0.3]), "B08": np.array([0.1]), "B11": np.array([0.0])}
    np.testing.assert_allclose(ndwi(bands), [0.5])  # green vs NIR
    np.testing.assert_allclose(mndwi(bands), [1.0])  # green vs SWIR
