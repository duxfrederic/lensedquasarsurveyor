supported_surveys = ['legacysurvey', 'panstarrs', 'hsc']
band_header_keyword = {
   'legacysurvey': 'BAND',
   'panstarrs': 'FPA.FILTER',
   'hsc': 'FILTER'
}

if not set(band_header_keyword.keys()) == set(supported_surveys):
    raise AssertionError('Not all surveys have their band keyword info')