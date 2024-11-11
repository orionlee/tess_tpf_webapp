from typing import Tuple

import urllib

import astroquery.vizier as vizier

from lightkurve.interact_sky_providers.gaia_tic import GaiaDR3TICInteractSkyCatalogProvider


class ExtendedGaiaDR3TICInteractSkyCatalogProvider(GaiaDR3TICInteractSkyCatalogProvider):
    """Custom extension to standard Gaia DR3 TIC provider."""

    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        key_vals, extra_rows = super().get_detail_view(data)

        vizier_server = vizier.conf.server

        # link to Gaia DR3. Cross-match with known variable objects (Gavras+, 2023)
        # Note: coordinate search (15" radius) is used instead of matching Gaia source
        # for cases that a TIC has no Gaia source, or cases the xmatch result is slightly off.
        coordStrEncoded = urllib.parse.quote_plus(f"{data['ra']} {data['dec']}")
        gaia_xmatch_var_url = f"https://{vizier_server}/viz-bin/VizieR-4?-ref=VIZ65ea51f497bf&-to=-4b&-from=-3&-this=-4&%2F%2Fsource=J%2FA%2BA%2F674%2FA22%2Fcatalog&%2F%2Ftables=J%2FA%2BA%2F674%2FA22%2Fcatalog&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&-out.add=_r&-out.add=_p&%2F%2Foutaddvalue=default&-sort=_r&-order=I&-oc.form=sexa&-out.src=J%2FA%2BA%2F674%2FA22%2Fcatalog&-nav=cat%3AJ%2FA%2BA%2F674%2FA22%26tab%3A%7BJ%2FA%2BA%2F674%2FA22%2Fcatalog%7D%26key%3Asource%3DJ%2FA%2BA%2F674%2FA22%2Fcatalog%26pos%3A{coordStrEncoded}%28+15+arcsec+J2000%29%26HTTPPRM%3A&-c={coordStrEncoded}&-c.eq=J2000&-c.r=+15&-c.u=arcsec&-c.geom=r&-source=&-source=J%2FA%2BA%2F674%2FA22%2Fcatalog&-out=GaiaDR3&-out=ONames&-out=RAJ2000&-out=DEJ2000&-out=Omags&-out=psuperclass&psuperclass=%21%3D%2CCST%2CAGN%2CGALAXY&-out=pvarTypes&-out=VarTypes&-out=VarTypesOri&-out=AltVarTypesOri&-out=pPer&-out=OPers&-out=ORefEpoch&-out=Cats&-out=Sel&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET"
        extra_rows.append(f'<a href="{gaia_xmatch_var_url}" target="_blank">Gaia DR3 XMatch Variables by coordinate</a> [2023A&A...674A..22G]')

        # link to Gaia DR3 Stellar Variability catalog (based on photometric dispersions)
        # Note: coordinate search is used instead of Gaia source,
        #       as it could be helpful for users to review the variability of stars in the vicinity.
        gaia_dispersions_url = f"https://{vizier_server}/viz-bin/VizieR-4?-out.add=_r&-out.add=_p&%2F%2Foutaddvalue=default&-sort=_r&-order=I&-c={coordStrEncoded}&-c.eq=J2000&-c.r=+30&-c.u=arcsec&-c.geom=r&-source=&-out.src=J%2FA%2BA%2F677%2FA137%2Fcatalog&-source=J%2FA%2BA%2F677%2FA137%2Fcatalog&-bmark=GET"
        extra_rows.append(f'<a href="{gaia_dispersions_url}" target="_blank">Gaia DR3 Stellar Variability by coordinate</a> [2023A&A...677A.137M]')

        return key_vals, extra_rows
