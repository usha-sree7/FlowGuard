"""Dataset registry. Every field below was checked against the actual files on 2026-10-25
(see docs/PAPER_DATASET_AUDIT.md). Items that could not be verified are marked as such."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    paper_title: str
    paper_ref: str
    doi: str
    repo: str
    repo_commit: str
    licence: str
    location: str
    period_claimed: str
    resolution: str
    step_hours: float
    files: tuple[RemoteFile, ...]
    column_mapping: dict
    rain_sensors: tuple[str, ...]
    label_info: dict
    limitations: tuple[str, ...]
    unverified: tuple[str, ...] = field(default_factory=tuple)
    level_bounds_m: tuple[float, float] = (0.0, 10.0)
    rain_unit: str = "gauge units; unit not documented in the source"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["files"] = [asdict(f) for f in self.files]
        return d


BENGALAS = DatasetSpec(
    id="bengalas",
    name="Bengalas River (Petropolis, RJ, Brazil) - UAE-ML",
    paper_title="Uncertainty-Aware Machine Learning Models for Flash Flood Prediction",
    paper_ref="Soares, J.A.J.P.; Soares, A.K.S.; Satolo, L.F.; Scofield, G.B.; Stephany, S.; Santos, L.B.L. "
    "Transactions in GIS 30(2), 2026",
    doi="10.1111/tgis.70254",
    repo="https://github.com/jaqueline-soares/UAE-ML",
    repo_commit="dfb6ee046667ad7287e2630cc004b2be9aa5347b",
    licence="MIT (repository LICENSE, Copyright (c) 2026 the paper authors)",
    location="Bengalas River watershed, Petropolis, RJ, Brazil; INEA hydrological station code 221342310 "
    "(lon -42.5201, lat -22.2285, from HydrologicalStation.dbf)",
    period_claimed="2019-2023 (file name); verified first/last timestamps 2019-01-04 14:30 to 2023-12-31 16:00",
    resolution="30 min (70,760 of 72,010 consecutive steps are 30 min; the rest are longer gaps - see quality report)",
    step_hours=0.5,
    files=(
        RemoteFile(
            name="dataset.zip",
            url="https://github.com/jaqueline-soares/UAE-ML/raw/main/data/dataset.zip",
            sha256="19ef08b739f4ed4dd094996636ce912289aa40c5ddcf6a83617089c35acef6d4",
            size=41520920,
        ),
    ),
    column_mapping={
        "source_file": "dataset.zip :: bacia_rio_bengalas_2019_2023_out00h30m.csv",
        "datahora": "timestamp (timezone not stated in repo - treated as naive local time)",
        "nivel_conselheiro": "level [m] - water level at the hydrological station (unit inferred: the authors' "
        "code compares it with mean+2sd 'flood threshold'; values 0.50-4.44)",
        "chuva_conselheiro": "rain__gauge [mm per 30 min, UNVERIFIED unit] - station rain gauge",
        "precip0..precip503": "504 weather-radar grid cells (504 records in WeatherRadarCoverage_Grid.dbf). "
        "Aggregated to rain__radar = cell mean [unit not stated in repo; bias-adjusted to the gauge on the train split]",
        "out00h30m": "next-step water level (verified == nivel_conselheiro at t+30min for all 70,760 consecutive pairs). "
        "Not used as a model input.",
    },
    rain_sensors=("gauge", "radar"),
    label_info={
        "true_flood_labels": False,
        "kind": "proxy",
        "description": "No observed flood/no-flood labels in the repository. The authors' uncertainty-analysis code "
        "defines a statistical 'flood threshold' = mean + 2*std of nivel_conselheiro (computed by them on the full "
        "dataset). HydroSentinel AI recomputes it on the chronological TRAIN split only and uses it as the High "
        "threshold of a 5-level PROXY label on the FUTURE water level (Normal / Low / Moderate / High / Critical). "
        "Low (mean+0.5sd), Moderate (mean+1sd) and Critical (mean+3sd) are project extensions, not from the paper.",
    },
    limitations=(
        "Proxy labels only: exceedance of a statistical level threshold, not observed flood impacts.",
        "Units of rain gauge and radar columns are not documented in the repository.",
        "Timezone not stated.",
        "1,250 timestamp gaps longer than 30 min; rise rate is not computed across gaps.",
        "Single water-level sensor: level faults cannot be cross-checked against a redundant level sensor.",
    ),
    unverified=(
        "Paper full text (Wiley returned HTTP 403). Authors, title, Transactions in GIS 30(2), article e70254 confirmed via the Crossref record of the DOI (2026-10-26).",
        "Brief states '>73,000 tuples'; the 30-min file actually has 72,011 rows (69,356 in the 6-h file).",
    ),
    level_bounds_m=(0.0, 10.0),
)

MALWATHU = DatasetSpec(
    id="malwathu",
    name="Malwathu Oya (Sri Lanka) - Malwathu_Oya_RR",
    paper_title="Water level prediction using soft computing techniques: A case study in the Malwathu Oya, Sri Lanka",
    paper_ref="Rathnayake, N.; Rathnayake, U.; Dang, T.L.; Hoshino, Y. PLOS ONE 18(4): e0282847, 2023",
    doi="10.1371/journal.pone.0282847",
    repo="https://github.com/namalhappy/Malwathu_Oya_RR",
    repo_commit="b4f6bfea42de79f6267cf02352cfc7cff01a4541",
    licence="MIT (repository LICENSE, Copyright (c) 2023 namalhappy)",
    location="Malwathu Oya basin, Sri Lanka; water level at Kappachchiya; rain gauges Murunkan, Pavattakulam, "
    "Nachchiduva, Vavuniya, Mannar, Anuradhapura",
    period_claimed="2005-2018 per paper (years only). The CSV has NO date column; HydroSentinel AI uses a relative day index.",
    resolution="daily (per paper; files contain lag-structured rows, no timestamps)",
    step_hours=24.0,
    files=(
        RemoteFile(
            name="malwathu_1_day.csv",
            url="https://raw.githubusercontent.com/namalhappy/Malwathu_Oya_RR/main/malwathu_1_day.csv",
            sha256="3edcc4d6b09b9cbf4223cbcb74cfcb5662357a916c774899c1b790875351ec49",
            size=201634,
        ),
    ),
    column_mapping={
        "source_file": "malwathu_1_day.csv (4,764 rows x 14 cols; row i holds day i-1 lags and day i values)",
        "<Station>-1": "previous-day value; verified equal to <Station> on the previous row for all 6 rain gauges",
        "Murrunkan, Pavattakulam, Nachchiduva, Vavniya, Mannr, Apura": "rain__<station> [unit: paper says cm; "
        "magnitudes (max 417.97/day) suggest mm - UNVERIFIED, kept as-is]",
        "Kappachichiya": "level [m] (paper: water level in metres at Kappachchiya). The lag column "
        "Kappachichiya-1 is exactly 100x the target (cm) - the loader uses the metre-scaled column.",
        "series reconstruction": "day 0 from row-0 lag columns, days 1..4764 from current columns -> 4,765 days, "
        "matching the paper's 4,765 samples.",
    },
    rain_sensors=("Murrunkan", "Pavattakulam", "Nachchiduva", "Vavniya", "Mannr", "Apura"),
    label_info={
        "true_flood_labels": False,
        "kind": "proxy",
        "description": "Paper reports no numeric flood/alert water levels. HydroSentinel AI applies the same statistical "
        "rule as the Bengalas proxy (train-split mean + k*sd of level, k=0.5,1,2,3) and labels it as a PROXY. "
        "It is used for a generalisation check only.",
    },
    limitations=(
        "No timestamps in the files; only years 2005-2018 are stated in the paper.",
        "Daily resolution: flash-flood dynamics within a day are not observable.",
        "Rain unit ambiguity (paper says cm; values look like mm).",
        "Proxy labels only; paper gives no alert levels.",
    ),
    unverified=(
        "Data-provider agency not stated in the paper.",
        "Exact start/end dates.",
    ),
    level_bounds_m=(0.0, 15.0),
)

NWDP_LEVEL_DS = "https://nwdp.nwic.gov.in/dataset/68600163-c5a0-4327-aa1a-fa7157b86cce"
NWDP_RAIN_DS = "https://nwdp.nwic.gov.in/dataset/792f0818-bedf-42ab-a119-dc7002ceb312"

# Checked on 2026-10-29 (see docs/INDIAN_DATA_AUDIT.md). Of the 66 Godavari-basin telemetry level stations, Chindnar was
# chosen because it has a rain gauge at the same coordinates, natural (unregulated) monsoon rises and few stuck values.
CWC_CHINDNAR = DatasetSpec(
    id="cwc_chindnar",
    name="Indravati River at Chindnar (Chhattisgarh, India) - CWC telemetry",
    paper_title="River Water Level (Telemetry - Hourly) and Rainfall (Telemetry - Hourly), Central Water Commission",
    paper_ref="Central Water Commission (CWC), published by the National Water Informatics Centre on the National "
    "Water Data Portal (NWDP), Ministry of Jal Shakti, Government of India",
    doi="",
    repo="https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc",
    repo_commit="files downloaded 2026-10-29 (NWDP 'Last Update' 2026-10-29)",
    licence="Not stated on the dataset pages (portal footer: (c) NWIC, All Rights Reserved; National Water Data "
    "Policy applies). Raw files are NOT redistributed with this project; the importer downloads them from NWDP.",
    location="CWC telemetry station Chindnar, Indravati River (Godavari basin), Dantewada district, Chhattisgarh, "
    "India; lat 19.0831, lon 81.3000; gauge zero RL 327.15 m. Rain gauge at the same coordinates.",
    period_claimed="Resource '2021-2025'. Chindnar level 2021-01-01 to 2025-05-17 (see quality report)",
    resolution="hourly telemetry with gaps; off-hour readings (mostly hh:30) are moved to the next full hour",
    step_hours=1.0,
    files=(
        RemoteFile(
            name="rwl_tel_hr_cwc_006_2021_2025.csv",
            url=f"{NWDP_LEVEL_DS}/resource/4d139325-8371-422f-acef-d34ccec4300e/download/rwl_tel_hr_cwc_006_2021_2025.csv",
            sha256="675a55fc63aab786486ec9faaa7ef8073805ce986815d61c319f545965a0313a",
            size=185715692,
        ),
        RemoteFile(
            name="rainfall_tel_hr_cwc_cg_2021_2025.csv",
            url=f"{NWDP_RAIN_DS}/resource/8984d2b2-1275-4443-9bb6-087cf539fafb/download/rainfall_tel_hr_cwc_cg_2021_2025.csv",
            sha256="6338787c5beb798e10e8da43cdc3e684ee572a46dff590a0c85991e6a4fcf931",
            size=5850926,
        ),
    ),
    column_mapping={
        "source_files": "rwl_tel_hr_cwc_006_2021_2025.csv (Godavari basin, all 66 stations) and "
        "rainfall_tel_hr_cwc_cg_2021_2025.csv (Chhattisgarh, all stations); rows with Station == 'Chindnar' used",
        "Data Acquisition Time": "timestamp, format DD-MM-YYYY HH:MM (timezone not stated; assumed IST)",
        "River Water Level Telemetry Hourly (meter)": "level [m above mean sea level; gauge zero RL 327.15 m]",
        "Telemetry Hourly Rainfall (mm)": "rain__gauge [mm per hour]. The file holds ONLY non-zero records "
        "(minimum 0.1 mm). An hour with a level reading but no rain record is set to 0 mm (assumption).",
    },
    rain_sensors=("gauge",),
    label_info={
        "true_flood_labels": False,
        "kind": "proxy",
        "description": "No flood events or official warning / danger levels are included in the NWDP files. The same "
        "PROXY rule as for Bengalas is used: train-split mean + k*sd of the level (k = 0.5, 1, 2, 3); Alert = level "
        "above mean + 2 sd. This is a statistical exceedance, not an observed flood and not a CWC warning level.",
    },
    limitations=(
        "Proxy labels only; CWC warning / danger levels for Chindnar were not available in the downloaded files.",
        "Rain file contains non-zero records only, so a missing rain hour cannot be told apart from a dry hour.",
        "Single point rain gauge for a large basin: local rain explains little of the level change "
        "(correlation of past-24 h rain with next-24 h rise about 0.08).",
        "Many gaps; 1 h and 3 h horizons only (30 min is not representable).",
        "Level values outside 327.15-350 m and rain above 100 mm/h are removed as implausible (counts in the "
        "quality report).",
    ),
    unverified=(
        "Units: the column names say metre and mm; no separate data dictionary was found.",
        "Timezone of 'Data Acquisition Time'.",
        "Licence / reuse terms for the NWDP files.",
    ),
    level_bounds_m=(327.15, 350.0),
    rain_unit="mm (from the column name 'Telemetry Hourly Rainfall (mm)')",
)

DATASETS: dict[str, DatasetSpec] = {d.id: d for d in (BENGALAS, MALWATHU, CWC_CHINDNAR)}

# Candidates audited but NOT integrated (see docs/PAPER_DATASET_AUDIT.md).
NOT_INTEGRATED = [
    {"name": "FlowDB", "ref": "arXiv:2012.11154 (Godfried et al., 2020)",
     "status": "Paper verified (hourly river flow + precipitation, flash-flood event subset). No download link on the "
     "arXiv page; not integrated."},
    {"name": "LamaH-CE", "ref": "arXiv:2601.09336 is NOT the LamaH-CE data paper; it is Bertoli et al. 2026, "
     "'A Hybrid Machine Learning Framework for Improved Short-Term Peak-Flow Forecasting', which USES LamaH-CE "
     "(857 catchments, 6-h). Data download not verified here.", "status": "Not integrated; reference needs correction."},
    {"name": "'ANNEX' benchmark", "ref": "arXiv:2001.06414 is Okuno, Ikeuchi & Aihara 2020, 'Practical data-driven "
     "flood forecasting based on dynamical systems theory: Case studies from Japan'. The abstract page does not "
     "mention an 'ANNEX' benchmark or a download link.", "status": "Not integrated; needs manual verification."},
]
