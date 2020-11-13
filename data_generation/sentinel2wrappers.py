class Sentinel2Tile:
    def __init__(self):
        self.id = None
        self.safes = []
        self.epsgs = []
        self.corner = {'x': None, 'y': None}

    def __str__(self):
        return 'Sentinel2Tile:{' \
               f'   ID: {self.id},' \
               f'   SAFES: {self.safes},' \
               f'   EPSG\'S: {self.epsgs}' \
               '}'


class Sentinel2Safe:
    def __init__(self):
        self.corners = None
        self.s2_path = None
        self.date = None
        self.time = None
        self.epsg = None
        self.tidal_elevation = None
