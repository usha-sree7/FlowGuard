"""Verified references (single source for the dashboard and REFERENCES.md, see scripts/write_references.py).

Every DOI below was resolved through the Crossref REST API (api.crossref.org/works/<doi>) on 2026-10-26 and the
authors / year / title / venue were copied from that record. `supports` says which design decision the work is
cited for; none of them is cited as evidence that our framework works.
"""
ACCESSED = "2026-10-26"

REFERENCES = [
    # ---- datasets
    {"id": "soares2026", "topic": "Dataset (primary)", "authors": "Soares, J.A.J.P.; Soares, A.K.S.; Satolo, L.F.; "
     "Scofield, G.B.; Stephany, S.; Santos, L.B.L.", "year": 2026,
     "title": "Uncertainty-Aware Machine Learning Models for Flash Flood Prediction", "venue": "Transactions in GIS 30(2), e70254",
     "doi": "10.1111/tgis.70254", "url": "https://github.com/jaqueline-soares/UAE-ML",
     "supports": "Source of the Bengalas River 30-min dataset (MIT-licensed repository) and of the mean + 2 sd "
                 "statistical 'flood threshold' rule used for the High proxy threshold."},
    {"id": "rathnayake2023", "topic": "Dataset (secondary)", "authors": "Rathnayake, N.; Rathnayake, U.; Dang, T.L.; Hoshino, Y.",
     "year": 2023, "title": "Water level prediction using soft computing techniques: A case study in the Malwathu Oya, Sri Lanka",
     "venue": "PLOS ONE 18(4), e0282847", "doi": "10.1371/journal.pone.0282847",
     "url": "https://github.com/namalhappy/Malwathu_Oya_RR",
     "supports": "Source of the Malwathu Oya daily dataset, used as a generalisation check."},
    # ---- fuzzy logic
    {"id": "zadeh1965", "topic": "Fuzzy logic", "authors": "Zadeh, L.A.", "year": 1965, "title": "Fuzzy sets",
     "venue": "Information and Control 8(3), 338-353", "doi": "10.1016/S0019-9958(65)90241-X", "url": None,
     "supports": "Membership functions and linguistic terms (IT2FIS, PSO-tuned IT2FIS, ANFIS)."},
    {"id": "mamdani1975", "topic": "Fuzzy logic", "authors": "Mamdani, E.H.; Assilian, S.", "year": 1975,
     "title": "An experiment in linguistic synthesis with a fuzzy logic controller",
     "venue": "International Journal of Man-Machine Studies 7(1), 1-13", "doi": "10.1016/S0020-7373(75)80002-2", "url": None,
     "supports": "IF-THEN rule base with linguistic consequents (the 80 expert rules of the IT2FIS)."},
    {"id": "liang2000", "topic": "Interval Type-2 FIS", "authors": "Liang, Q.; Mendel, J.M.", "year": 2000,
     "title": "Interval type-2 fuzzy logic systems: theory and design", "venue": "IEEE Transactions on Fuzzy Systems 8(5), 535-550",
     "doi": "10.1109/91.873577", "url": None,
     "supports": "IT2 FLS structure: interval firing strengths, interval (non-singleton) inputs (IT2FIS; interval inputs in the reliability layer)."},
    {"id": "karnik2001", "topic": "Interval Type-2 FIS", "authors": "Karnik, N.N.; Mendel, J.M.", "year": 2001,
     "title": "Centroid of a type-2 fuzzy set", "venue": "Information Sciences 132(1-4), 195-220",
     "doi": "10.1016/S0020-0255(01)00069-X", "url": None,
     "supports": "Karnik-Mendel type reduction giving the interval [yl, yr] (IT2FIS, PSO-tuned IT2FIS)."},
    {"id": "mendel2002", "topic": "Interval Type-2 FIS", "authors": "Mendel, J.M.; John, R.I.B.", "year": 2002,
     "title": "Type-2 fuzzy sets made simple", "venue": "IEEE Transactions on Fuzzy Systems 10(2), 117-127",
     "doi": "10.1109/91.995115", "url": None,
     "supports": "Footprint of uncertainty and embedded Type-1 sets (Type-1 vs Type-2 explanation)."},
    {"id": "mendel2017", "topic": "Interval Type-2 FIS", "authors": "Mendel, J.M.", "year": 2017,
     "title": "Uncertain Rule-Based Fuzzy Systems: Introduction and New Directions (2nd ed.)", "venue": "Springer (book)",
     "doi": "10.1007/978-3-319-51370-6", "url": None,
     "supports": "Textbook reference for IT2 rule-based systems and uncertainty in measurements."},
    # ---- neural pattern recognition
    {"id": "kohonen1990", "topic": "SOM", "authors": "Kohonen, T.", "year": 1990, "title": "The self-organizing map",
     "venue": "Proceedings of the IEEE 78(9), 1464-1480", "doi": "10.1109/5.58325", "url": None,
     "supports": "SOM algorithm: competitive learning with a shrinking neighbourhood (batch variant used here)."},
    {"id": "kalteh2008", "topic": "SOM in hydrology", "authors": "Kalteh, A.M.; Hjorth, P.; Berndtsson, R.", "year": 2008,
     "title": "Review of the self-organizing map (SOM) approach in water resources: Analysis, modelling and application",
     "venue": "Environmental Modelling & Software 23(7), 835-845", "doi": "10.1016/j.envsoft.2007.10.001", "url": None,
     "supports": "SOM is an established tool for clustering hydrological states (not a new invention here)."},
    # ---- sensors, fusion, evaluation
    {"id": "hill2010", "topic": "Sensor fault / anomaly detection", "authors": "Hill, D.J.; Minsker, B.S.", "year": 2010,
     "title": "Anomaly detection in streaming environmental sensor data: A data-driven modeling approach",
     "venue": "Environmental Modelling & Software 25(9), 1014-1022", "doi": "10.1016/j.envsoft.2009.08.010", "url": None,
     "supports": "Causal, streaming anomaly checks on environmental sensors (reliability layer)."},
    {"id": "khaleghi2013", "topic": "Sensor fusion", "authors": "Khaleghi, B.; Khamis, A.; Karray, F.O.; Razavi, S.N.",
     "year": 2013, "title": "Multisensor data fusion: A review of the state-of-the-art",
     "venue": "Information Fusion 14(1), 28-44", "doi": "10.1016/j.inffus.2011.08.001", "url": None,
     "supports": "Data imperfection (outliers, conflict, uncertainty) and reliability-weighted fusion concepts (reliability layer)."},
    {"id": "mosavi2018", "topic": "Flood forecasting", "authors": "Mosavi, A.; Ozturk, P.; Chau, K.-w.", "year": 2018,
     "title": "Flood Prediction Using Machine Learning Models: Literature Review", "venue": "Water 10(11), 1536",
     "doi": "10.3390/w10111536", "url": None,
     "supports": "Context: data-driven flood prediction; lead-time dependence of skill."},
    {"id": "jolliffe2011", "topic": "Alert evaluation", "authors": "Jolliffe, I.T.; Stephenson, D.B. (eds.)", "year": 2011,
     "title": "Forecast Verification: A Practitioner's Guide in Atmospheric Science (2nd ed.)", "venue": "Wiley (book)",
     "doi": "10.1002/9781119960003", "url": None,
     "supports": "Contingency-table verification (hits, misses, false alarms) behind precision / recall / FPR."},
    {"id": "cohen1960", "topic": "Evaluation", "authors": "Cohen, J.", "year": 1960,
     "title": "A Coefficient of Agreement for Nominal Scales", "venue": "Educational and Psychological Measurement 20(1), 37-46",
     "doi": "10.1177/001316446002000104", "url": None,
     "supports": "Cohen's kappa: agreement of the 5-level alert with the proxy label beyond chance."},
    {"id": "kunsch1989", "topic": "Evaluation", "authors": "Kunsch, H.R.", "year": 1989,
     "title": "The Jackknife and the Bootstrap for General Stationary Observations",
     "venue": "The Annals of Statistics 17(3), 1217-1241", "doi": "10.1214/aos/1176347265", "url": None,
     "supports": "Moving-block bootstrap for the 95 % confidence intervals of alert F1 on autocorrelated test data."},
    # ---- neuro-fuzzy and optimisation
    {"id": "jang1993", "topic": "ANFIS", "authors": "Jang, J.-S.R.", "year": 1993,
     "title": "ANFIS: adaptive-network-based fuzzy inference system",
     "venue": "IEEE Transactions on Systems, Man, and Cybernetics 23(3), 665-685", "doi": "10.1109/21.256541", "url": None,
     "supports": "ANFIS architecture (5 layers, first-order TSK rules) and hybrid least-squares / gradient learning."},
    {"id": "kennedy1995", "topic": "PSO", "authors": "Kennedy, J.; Eberhart, R.", "year": 1995,
     "title": "Particle swarm optimization", "venue": "Proceedings of ICNN'95 - International Conference on Neural Networks, vol. 4, 1942-1948",
     "doi": "10.1109/ICNN.1995.488968", "url": None,
     "supports": "Particle swarm optimisation used to tune the IT2FIS membership functions."},
    {"id": "shi1998", "topic": "PSO", "authors": "Shi, Y.; Eberhart, R.", "year": 1998,
     "title": "A modified particle swarm optimizer",
     "venue": "1998 IEEE International Conference on Evolutionary Computation Proceedings, 69-73",
     "doi": "10.1109/ICEC.1998.699146", "url": None,
     "supports": "Inertia weight (w = 0.7 here) in the PSO velocity update."},
]

DATA_SOURCES_CHECKED = [
    {"name": "India Meteorological Department (IMD)", "url": "https://mausam.imd.gov.in/", "reachable": True,
     "status": "Rainfall only (station / gridded, daily). No co-located river level. NOT used."},
    {"name": "Central Water Commission (CWC)", "url": "https://cwc.gov.in/", "reachable": True,
     "status": "Stage is observed hourly only in the monsoon (3 readings/day otherwise); archived data is served "
               "through India-WRIS to registered users. NOT used."},
    {"name": "CWC Flood Forecasting System", "url": "https://ffs.india-water.gov.in/", "reachable": True,
     "status": "Live hydrographs covering roughly the last 72 h; no bulk historical download found. NOT used."},
    {"name": "National Water Data Portal (NWIC)", "url": "https://nwdp.nwic.gov.in/", "reachable": True,
     "status": "Lists daily CWC river discharge and daily IMD rainfall; no sub-daily river level found. NOT used."},
    {"name": "India-WRIS", "url": "https://indiawris.gov.in/wris/", "reachable": True,
     "status": "Page loads only as a script application (timed out on the first check); downloads need "
               "registration. Could not verify a sub-daily level + rain archive. NOT used."},
    {"name": "CAMELS-IND (Mangukiya et al., ESSD 17, 461-491, 2025, doi:10.5194/essd-17-461-2025)",
     "url": "https://doi.org/10.5281/zenodo.14005378", "reachable": True,
     "status": "Verified open Indian dataset (472 catchments, 1980-2020), but DAILY streamflow + forcings and no "
               "river stage: cannot support the 30 min / 1 h / 3 h experiments. Best candidate for a future daily "
               "Indian generalisation check. NOT used."},
    {"name": "INEA (Instituto Estadual do Ambiente, RJ, Brazil)", "url": None, "reachable": None,
     "status": "Station code 221342310 appears in the UAE-ML repository files; the data were obtained from the "
               "authors' repository, not directly from INEA. Original agency provenance not independently verified."},
]
