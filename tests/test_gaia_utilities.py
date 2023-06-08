from lensedquasarsurveyor.gaia_utilities import get_positionangles_separations

def test_get_positionangles_separations():
    # Set a known target with known results
    ra, dec = 44.8899, -23.6338
    expected_separation = 2.92 
    expected_position_angle = 81.1 
    result = get_positionangles_separations(ra, dec, searchradiusarcsec=3)
    
    assert result is not None, "Result is None"
    assert result['name'] == "J0259-2338", f"Expected name '{name}', but got '{result['name']}'"
    assert abs(result['separation'] - expected_separation) < 0.05, f"Expected separation {expected_separation}, but got {result['separation']}"
    assert abs(result['position_angle'] - expected_position_angle) < 0.5, f"Expected position angle {expected_position_angle}, but got {result['position_angle']}"

    for magnitude in result['magnitudes']:
        assert 'ra' in magnitude
        assert 'dec' in magnitude
        assert 'g' in magnitude
        assert 'bp' in magnitude
        assert 'rp' in magnitude
