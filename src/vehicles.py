"""Vehículos agregados por accidente (fuente oficial del DfT).

La construcción vive en src/dft.py (build_vehicles). Solo se usan atributos previos al choque
(tipo de vehículo, edad del conductor, antigüedad y cilindrada); se excluyen derrape, punto de impacto,
objeto golpeado y maniobra porque describen cómo ocurrió el choque (fuga de información).
Los códigos de tipo de vehículo anteriores a 2005 (106 = moto > 125 cc, 108 = taxi, 109 = coche) se
interpretan con la guía oficial.
"""
import pandas as pd

from dft import VEHICLE_COLUMNS, VEHICLES_DFT, build_vehicles

__all__ = ["VEHICLE_COLUMNS", "load_vehicle_agg"]


def load_vehicle_agg() -> pd.DataFrame:
    if not VEHICLES_DFT.exists():
        return build_vehicles()
    return pd.read_parquet(VEHICLES_DFT)
