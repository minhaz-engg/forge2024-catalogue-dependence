# Data

All inputs are public. Download the two catalogues into `data/raw/` (about 1 MB in total):

| Put here | Download from | Licence |
|---|---|---|
| `data/raw/FORGE2024_Niemz2024_surface_catalog.csv` | Niemz et al. (2024), Zenodo, https://doi.org/10.5281/zenodo.12817881 (the event catalogue CSV, renamed) | see record |
| `data/raw/FORGE2024_Zhu2026_phasepick_catalog.csv` | Zhu et al. (2026), Geothermal Data Repository, https://gdr.openei.org/submissions/1817 (`Utah_FORGE_2024_Auto_Pick_catalog_2026.1.23.csv`, renamed) | CC BY 4.0 |

`data/stage_times.csv` (included) gives pump-on (`t_on`), shut-in (`t_off`), clean volume (`V`, m³) and the end of the analysis window (`t_end`) of each stage, in seconds since 2024-04-03 00:00 UTC. It was derived from the Calfrac treatment records of England (2024), https://doi.org/10.15121/2371032 (CC BY 4.0): pump-on and shut-in are the first and last samples with clean rate above 5 bbl/min, with UTC = local time + 6 h.
