from functools import lru_cache
from typing import Tuple, Union

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.ipac.irsa import Irsa

import numpy as np

from .core import ProperMotionCorrectionMeta, InteractSkyCatalogProvider


def _to_lc_url(oid, data_release, format):
    # see: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
    if isinstance(oid, (int, np.int64, str)):
        url = f"https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID={oid}&COLLECTION=ztf_dr{data_release}&FORMAT={format}"
        return url
    else:
        return [_to_lc_url(a_oid, data_release, format) for a_oid in oid]

        # OPEN: consider memoize the result, as astroquery v0.47 does not support caching for Irsa


@lru_cache
def _query_cone_region(ra_deg, dec_deg, radius_arcsec, catalog) -> Table:
    # Note: memoize the result: astroquery (as of v0.47) does not support caching for Irsa
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    return Irsa.query_region(coord, catalog=catalog, spatial="Cone", radius=radius_arcsec * u.arcsec)


class ZTFInteractSkyCatalogProvider(InteractSkyCatalogProvider):
    """
    Provide Zwicky Transient Facility Archive data to
    `TargetPixelFile.interact_sky() <lightkurve.TargetPixelFile.interact_sky>`.

    More information: https://irsa.ipac.caltech.edu/Missions/ztf.html

    https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html

    The class is used by ``interact_sky()`` internally. The behavior can
    be customized by supplying a dictionary of keyword parameters to
    `catalogs` parameter of ``interact_sky()``. The keyword parameters are
    then used to customize the parameters passed to the constructor here.

    Parameters
    ----------
    coord: `~astropy.coordinates.SkyCoord`
        the coordinate of the target.

    radius: float or `~astropy.units.Quantity`
        the cone search radius, in arc seconds if the value is float.

    magnitude_limit: float
        A value to limit the results in based on ZTF medianmag.

    scatter_kwargs: dict
        keyword arguments passed to bokeh's ``figure.scatter()``
        function to plot the stars.

    ngoodobsrel_min: int, optional
        minimum number of good observations (``ngoodobsrel``)
        required for an object to be included in result.
        Default is 100

    filtercode: str, optional
        If specified, only include objects of the specified `filtercode`,
        which can be one of ``zg``, ``zr``,  ``zi``.

    data_release: int, optional
        The ZTF data release to be used. The default is 21.

    """

    # OPEN: support BAD_CATFLAGS_MASK parameter in the ZTF LC URL

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        # query: ZTF-specific
        ngoodobsrel_min: int = 100,
        filtercode: str = None,
        data_release: int = 23,
        # for ZTF LC URL
        lc_format: str = "csv",
        scatter_kwargs: dict = None,
    ) -> None:
        if scatter_kwargs is None:
            scatter_kwargs = dict(
                marker="square",
                fill_alpha=0.3,
                line_color=None,
                selection_color="firebrick",
                nonselection_fill_alpha=0.3,
                nonselection_line_color=None,
                nonselection_line_alpha=1.0,
                fill_color="firebrick",
                hover_fill_color="firebrick",
                hover_alpha=0.9,
                hover_line_color="white",
            )
        super().__init__(coord, radius, magnitude_limit, scatter_kwargs)
        # ZTF-specific query criteria
        self.ngoodobsrel_min = ngoodobsrel_min
        self.filtercode = filtercode
        self.data_release = data_release
        # for ZTF LC URL
        self.lc_format = lc_format
        # ZTF-specific
        self.cols_for_source = [  # extra columns to be included in bokeh data source
            "oid",
            "filtercode",
            "ngoodobsrel",  # num. of good observations in the release
            "refmag",
            "refmagerr",
            "medianmag",
            "medmagerr",
            "maxmag",
            "minmag",
            "astrometricrms",  # Root Mean Squared deviation in epochal positions relative to object RA,Dec
        ]

    @property
    def label(self) -> str:
        return "ZTF"

    def query_catalog(self) -> Table:
        # Optimization: ZTF coverage is about north of -30deg dec.
        # So skip query IRSA if the target is way too south and return an empty result table
        #
        # https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/ztf_release_notes_latest
        if self.coord.dec < -35 * u.deg:
            empty_data = dict(RA=[], DEC=[], magForSize=[])
            for c in self.cols_for_source:
                empty_data[c] = []
            return Table(data=empty_data)

        catalog = f"ztf_objects_dr{self.data_release}"
        rs = _query_cone_region(
            self.coord.ra.to(u.deg).value, self.coord.dec.to(u.deg).value, self.radius.to(u.arcsec).value, catalog
        )

        if self.magnitude_limit is not None:
            # column medianmag better reflects observed mag, but it could be 0
            rs = rs[(rs["medianmag"] < self.magnitude_limit) & (rs["medianmag"] != 0.0)]
        if self.ngoodobsrel_min is not None:
            rs = rs[rs["ngoodobsrel"] >= self.ngoodobsrel_min]
        if self.filtercode is not None:
            rs = rs[rs["filtercode"] == self.filtercode]

        # use standardized names for the required columns (across different catalogs)
        #
        # RA / DEC : coordinate in degree, proper motion corrected if possible
        rs.rename_column("ra", "RA")
        rs.rename_column("dec", "DEC")

        # magForSize: use a constant size, to avoid dots too small for dim ones
        rs["magForSize"] = np.full(len(rs), 15.0)  # use np.full() for empty rs edge case

        return rs

    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        # No PM correction can be done, as PM is not available in ZTF
        return None

    def get_tooltips(self) -> list:
        return [
            ("ZTF oid", "@oid"),
            ('Separation (")', "@separation{0.00}"),
            ("filter", "@filtercode"),
            ("num. good obs.", "@ngoodobsrel"),
            ("median mag", "@medianmag"),
            ("RA", "@ra{0,0.00000000}"),
            ("DEC", "@dec{0,0.00000000}"),
            ("column", "@x{0.0}"),
            ("row", "@y{0.0}"),
        ]

    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        ztf_url = _to_lc_url(data["oid"], self.data_release, self.lc_format)  # the csv data
        sand_url = f"https://ztf.snad.space/view/{data['oid']}"  # a web viewer
        oid_html = (
            f"""{data['oid']} (<a href="{ztf_url}" title="csv data" target="_blank">LC</a>"""
            f""",&emsp;<a href="{sand_url}" title="web viewer" target="_blank">sand</a>)"""
        )
        return {
            "ZTF OID": oid_html,
            'Separation (")': f"{data['separation']:.2f}",
            "filter": data["filtercode"],
            "num. good obs.": data["ngoodobsrel"],
            "Reference Mag": f"{data['refmag']:.3f} <span class='error'>{data['refmagerr']:.3f}<span>",
            "Median Mag": f"{data['medianmag']:.3f} <span class='error'>{data['medmagerr']:.3f}<span>",
            "Max Mag": f"{data['maxmag']:.3f}",
            "Min Mag": f"{data['minmag']:.3f}",
            "RA": f"{data['ra']:.8f}",
            "DEC": f"{data['dec']:.8f}",
            'astrometric RMS (")': f"{data['astrometricrms'] * 3600:.4f}",
            "column": f"{data['x']:.1f}",
            "row": f"{data['y']:.1f}",
        }, None
