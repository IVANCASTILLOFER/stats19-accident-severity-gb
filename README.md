# Road Accident Severity in Great Britain (STATS19, 2005–2025)

A multivariate analysis of what is associated with the severity of road accidents (slight, serious, fatal) in Great Britain, using the official police-reported data from the UK Department for Transport. The work compares a model **with** vehicle and driver data (A) against one **without** it (B), validates everything forward in time, and shows how the models age when the world (and the recording system) changes.

It was developed as the final project of a postgraduate diploma in data analysis. **It is not suitable for real road-safety decisions.**

## Questions

1. Which variables (road, environment, time, place, vehicles, drivers) are most associated with accident severity?
2. How much do vehicle and driver data add (model A vs model B)?
3. Does a model trained on the past keep working in later years, including the pandemic, and can it be corrected?

## Data

| Source | Content | Records (2005–2025) |
|---|---|---|
| DfT *Road Safety Data* – collisions | One row per accident: place, time, road, severity | 2,892,427 |
| DfT *Road Safety Data* – vehicles | One row per vehicle: type and driver characteristics | 5,301,126 |
| DfT data guide (xlsx) | Official code tables used to decode every label | – |

Source: <https://www.gov.uk/government/statistical-data-sets/road-safety-open-data> (Open Government Licence v3.0). STATS19 covers Great Britain (England, Wales, Scotland), not Northern Ireland.

The data (about 3.4 GB) is **not included** in this repository. Download the three files into `data/oficial_dft/`:

- `dft-road-casualty-statistics-collision-1979-latest-published-year.csv`
- `dft-road-casualty-statistics-vehicle-1979-latest-published-year.csv`
- `data-guide-2025.xlsx`

## Approach

- **Time-based split instead of a random one:** train 2005–2014, validation 2015, test 2016–2017, final hold-out 2018–2019, stress test 2020–2021 (pandemic), and later years 2022–2025.
- **Data-leakage control:** variables that contain the answer or are not known at the time of the estimate (for example, number of casualties) are excluded.
- **Imbalanced classes:** fatal accidents are about 1.5 % of the total, so accuracy is misleading. The project uses macro-F1, average precision and calibration metrics.
- **Models compared:** logistic regression (reference), XGBoost and LightGBM, explained with SHAP.
- **Spatial view:** accidents aggregated into cells of about 5 km, with checks on how stable the zones are across periods and on regions the model never saw.
- **Recording-system change:** since around 2016 many police forces classify severity by injury instead of by the officer's judgement. The analysis measures this effect by comparing forces that changed with forces that did not.

## Main findings (hold-out 2018–2019)

- Vehicle and driver data carry the most weight (31 % of model A's importance). With a motorcycle involved, 34 % of accidents are serious or fatal, against 18 % without one. Model B cannot use this and scores lower (macro-F1 0.41 vs 0.44).
- The model is useful to **prioritise**, not to detect with certainty: reviewing the top 5 % of accidents by risk captures 34 % of fatal accidents (chance would capture 5 %), but only about 1 in 10 alerts is truly fatal.
- Over the years the model keeps ranking cases well but underestimates the percentages: for 2025 it estimates 18.8 % serious or fatal accidents when the real figure was 26.2 %. Part of this comes from the change in recording; the rest is an increase that also appears without that change.
- Recalibrating the probabilities cuts the calibration error from 0.071 to 0.008, and retraining on recent years improves the ranking. Doing both works best.
- Knowing the place helps little, and only where there is already history.

## Limitations

- It estimates the severity of accidents that already happened. It does not forecast accidents or measure travel risk (there are no exposure data such as traffic or distance travelled).
- Importance values and proportions are associations, not causes.
- Severity is the officially recorded one, and its criterion changed during the period.
- Results are specific to Great Britain; what transfers to other contexts is the method of analysis and validation, not the estimated values.

## Repository structure

| Folder | Content |
|---|---|
| `src/` | Data (`dft.py`, `data.py`, `vehicles.py`), features (`features.py`), training and comparison (`train.py`, `ablation.py`, `final_models.py`, `model_comparison.py`), calibration (`calibration.py`, `recalibrate.py`), evaluation (`evaluate.py`, `holdout_eval.py`), spatial analysis (`geo.py`, `spatial_stability.py`, `spatial_holdout.py`) and app data (`prepare_app_data.py`) |
| `notebooks/` | `Proyecto1_Proceso_Completo.ipynb` (and `.html`): the complete process with outputs. Written in Spanish. |
| `models/` | Model A (with vehicles), model B (without vehicles) and the geographic feature object |
| `reports/` | Result tables (CSV) and figures |
| `app/` | Streamlit app with the zone map and the model estimates |
| `tests/` | App tests |

## Run it

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux / macOS
```

1. Put the three DfT files in `data/oficial_dft/` (see **Data**).
2. Build the processed tables: `python src/dft.py`, then `python src/prepare_app_data.py` (creates the aggregates the app uses).
3. Open the notebook (`notebooks/Proyecto1_Proceso_Completo.ipynb`). Its outputs are saved, so it can be read without running it.
4. Optional app: `streamlit run app/streamlit_app.py`

Tested with Python 3.12 on Windows 11.

## License

Code under the [MIT License](LICENSE). The data belongs to the UK Department for Transport and is published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Author

Iván Ricardo Castillo Fernández
