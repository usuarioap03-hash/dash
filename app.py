import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError

# ⚠️ OJO: aquí cambiamos "dashboard" por "dash"
from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px


# =========================
# Configuración DB (Render)
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")  # p.ej: postgresql://user:pass@host:5432/dbname
if not DATABASE_URL:
    print("ERROR: Falta la variable de entorno DATABASE_URL", file=sys.stderr)


def get_engine() -> Engine:
    # Recomendado: pool pequeño para Render
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=2,
    )


QUERY = text("SELECT * FROM vista_resumen")
PANAMA_TZ = pytz.timezone("America/Panama")

# -----------------------------------------------------
# Helpers para parsear fechas, horas, puntos, ciclos...
# -----------------------------------------------------
def _parse_fecha(fecha_col: pd.Series) -> pd.Series:
    out = pd.to_datetime(fecha_col, errors="coerce", dayfirst=True)
    return out.dt.date


def _parse_hora(h: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(h):
        return h.dt.time
    parsed = pd.to_datetime(h, errors="coerce", format="%I:%M:%S %p")
    nulls = parsed.isna()
    if nulls.any():
        parsed.loc[nulls] = pd.to_datetime(h[nulls], errors="coerce", infer_datetime_format=True)
    return parsed.dt.time


def _detect_punto_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.lower().startswith("punto")]


def _ensure_cycle_total(df: pd.DataFrame, punto_cols: list) -> pd.Series:
    if "ciclo_total" in df.columns:
        return pd.to_numeric(df["ciclo_total"], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def cargar_datos() -> pd.DataFrame:
    try:
        engine = get_engine()
        df = pd.read_sql(QUERY, engine)

        if "ciclo" in df.columns and "ciclo_numero" not in df.columns:
            df = df.rename(columns={"ciclo": "ciclo_numero"})

        if "fecha" in df.columns:
            df["fecha"] = _parse_fecha(df["fecha"])

        punto_cols = _detect_punto_cols(df)
        for c in punto_cols:
            df[c] = _parse_hora(df[c])

        df["ciclo_total"] = _ensure_cycle_total(df, punto_cols)

        sort_cols = [c for c in ["sesion_id", "ciclo_numero"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        return df
    except (SQLAlchemyError, OperationalError) as e:
        print(f"[DB ERROR] {e}", file=sys.stderr)
        return pd.DataFrame()


# =========================
#   App Dash (responsive)
# =========================
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, title="QRLogix Dashboard")
server = app.server  # para gunicorn


# ========== Layout inicial ==========
df0 = cargar_datos()

def kpi_card(title, value, id_value=None):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted small"),
            html.H4(id=id_value) if id_value else html.H4(value)
        ]),
        className="shadow-sm rounded-3"
    )

placas_opts = sorted([p for p in df0["placa"].dropna().unique()]) if "placa" in df0.columns else []
min_fecha = df0["fecha"].min() if "fecha" in df0.columns and not df0.empty else None
max_fecha = df0["fecha"].max() if "fecha" in df0.columns and not df0.empty else None

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H3("QRLogix • Dashboard de Escaneos"),
        dcc.Dropdown(
            id="filtro-placa",
            options=[{"label": p, "value": p} for p in placas_opts],
            multi=True,
            placeholder="Todas"
        ),
        dcc.DatePickerRange(
            id="filtro-fechas",
            start_date=min_fecha,
            end_date=max_fecha,
            display_format="DD/MM/YYYY"
        ),
        dcc.Graph(id="grafico-puntos"),
    ]
)

# =========================
# Callbacks
# =========================
@app.callback(
    Output("grafico-puntos", "figure"),
    Input("filtro-placa", "value"),
    Input("filtro-fechas", "start_date"),
    Input("filtro-fechas", "end_date"),
)
def actualizar(placas_sel, start_date, end_date):
    df = cargar_datos()
    if df.empty:
        return px.line(title="Sin datos")
    if placas_sel:
        df = df[df["placa"].isin(placas_sel)]
    if start_date:
        df = df[df["fecha"] >= pd.to_datetime(start_date).date()]
    if end_date:
        df = df[df["fecha"] <= pd.to_datetime(end_date).date()]

    punto_cols = _detect_punto_cols(df)
    if not punto_cols:
        return px.bar(title="Sin columnas de punto")

    melt = df.melt(id_vars=["placa", "fecha"], value_vars=punto_cols,
                   var_name="punto", value_name="hora")
    resumen = melt.groupby("punto", as_index=False).agg(registros=("hora", lambda s: s.notna().sum()))
    fig = px.bar(resumen, x="punto", y="registros", title="Registros por punto")
    return fig


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=True)