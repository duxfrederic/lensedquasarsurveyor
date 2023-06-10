supported_surveys = ['legacysurvey', 'panstarrs', 'hsc']
band_header_keyword = {
   'legacysurvey': 'BAND',
   'panstarrs': 'FPA.FILTER',
   'hsc': 'FILTER'
}
limit_psf_star_magnitude = {
    'legacysurvey': 16.8,
    'panstarrs': 16.3,
    'hsc': 18.4
}

if not set(band_header_keyword.keys()) == set(supported_surveys):
    raise AssertionError('Not all surveys have their band keyword info')
if not set(limit_psf_star_magnitude.keys()) == set(supported_surveys):
    raise AssertionError('Not all surveys have their limit mag star for psf')