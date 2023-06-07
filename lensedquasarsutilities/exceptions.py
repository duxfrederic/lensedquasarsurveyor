

class HSCCredentialsNotInEnvironment(Exception):
    def __init__(self, message="Add to your environment: your HSC username `HSCUSERNAME` and password `HSCPASSWORD`"):
        self.message = message
        super().__init__(self.message)


class HSCNoData(Exception):
    # here pass band, ra, dec info as well as which HSC re_run was queried
    def __init__(self, message="No HSC data found at your coordinates"):
        self.message = message
        super().__init__(self.message)
