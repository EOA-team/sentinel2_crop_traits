'''
Sensors currently supported for RTM inversion
'''

import pandas as pd

from copy import deepcopy
from pathlib import Path

class Sensors(object):
    """
    Class of classes of sensors
    """

    class Sentinel2:
        """
        defines properties shared by all members of the Sentinel2 platform
        """
        
        num_bands = 13
        band_names = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10',
            'B11', 'B12'
        ]
        central_wvls = [
            443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190
        ]
        band_names_long = [
            'COSTAL-AEROSOL', 'BLUE', 'GREEN', 'RED', 'REDEDGE1',
            'REDEDGE2', 'REDEDGE3', 'NIR', 'NARROW-NIR', 'WATERVAPOUR',
            'SWIR-CIRRUS', 'SWIR1', 'SWIR2'
        ]

        @classmethod
        def _read_srf_from_xls(cls, fpath_srf_xls: Path, sheet_name: str) -> pd.DataFrame:
            """
            Reads spectral response function from XLS document
    
            :param fpath_srf_xls:
                file-path to the xlsx document from ESA containing the SRF values per
                S2 band
            :param sheet_name:
                name of the sheet with the SRF values
            :returns:
                SRF values per wavelength [nm] and S2 band as DataFrame
            """
            srf = pd.read_excel(fpath_srf_xls, sheet_name=sheet_name)
            # rename columns to match the band names of S2
            new_cols = ['wvl'] + deepcopy(cls.band_names)
            srf.columns = new_cols
            return srf

    class Sentinel2A(Sentinel2):
        """
        defines Sentinel2A-MSI
        """
        def __init__(self):
            self.name = 'Sentinel2A-MSI'
            self.band_widths = [
                21, 66, 36, 31, 15, 15, 20, 106, 21, 20, 31, 91, 175
            ]

        def read_srf_from_xls(self, fpath_srf_xls: Path):
            """
            Reads spectral response function from XLS document
    
            :param fpath_srf_xls:
                file-path to the xlsx document from ESA containing the SRF values per
                S2A band
            :returns:
                SRF values per wavelength [nm] and S2A band as DataFrame
            """
            return self._read_srf_from_xls(fpath_srf_xls=fpath_srf_xls, sheet_name='Spectral Responses (S2A)')

    class Sentinel2B(Sentinel2):
        """
        defines Sentinel2B-MSI
        """
        def __init__(self):
            self.name = 'Sentinel2B-MSI'
            self.band_widths = [
                21, 66, 36, 31, 16, 15, 20, 106, 22, 21, 30, 94, 185
            ]

        def read_srf_from_xls(self, fpath_srf_xls: Path):
            """
            Reads spectral response function from XLS document
    
            :param fpath_srf_xls:
                file-path to the xlsx document from ESA containing the SRF values per
                S2A band
            :returns:
                SRF values per wavelength [nm] and S2A band as DataFrame
            """
            return self._read_srf_from_xls(fpath_srf_xls=fpath_srf_xls, sheet_name='Spectral Responses (S2B)')

    class Landsat8:
        """
        defines Landsat8-OLI
        """
        name = 'LANDSAT8-OLI'
        num_bands = 9
        # order of wvl of bands is a bit weird but it's the official ordering
        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']
        central_wvls = [440, 480, 560, 655, 865, 1610, 2200, 590, 1370]

    class PlanetSuperDove:
        """
        defines PlanetScope SuperDove
        """
        name = 'PS_Superdove'
        num_bands = 8
        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
        band_names_long = [
            'coastal-blue', 'blue', 'green_i', 'green', 'yellow', 'red', 'red-edge', 'NIR'
        ]
        central_wvls = [
            443, 490, 531, 565, 610, 665, 705, 865
        ]
        band_widths = [
            20, 50, 36, 36, 20, 31, 15, 40
        ]
