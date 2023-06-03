from lensedquasarsutilities.legacysurvey_utilities import download_legacy_survey_cutout, create_weighted_stack


def test_download_legacy_survey():
    # J 2122-1621
    filename = download_legacy_survey_cutout(320.6075, -16.357, 100)

    stack, noisemap = create_weighted_stack(filename, 'g')

