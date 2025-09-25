import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError

import dash
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

QUERY = text("""
    SELECT *
    FROM vista_resumen
""")

PANAMA_TZ = pytz.timezone("America/Panama")


def _parse_fecha(fecha_col: pd.Series) -> pd.Series:
    """
    Convierte una columna de fecha (posible texto 'DD-MM-YYYY' u otros) a datetime (fecha).
    """
    out = pd.to_datetime(fecha_col, errors="coerce", dayfirst=True)
    try:
        # Localizar a Panamá y devolver solo fecha
        return out.dt.tz_localize(PANAMA_TZ, nonexistent='NaT', ambiguous='NaT').dt.date
    except Exception:
        # Si ya estaba localizado o hay error, devolver solo la fecha
        return out.dt.date


def _parse_hora(h: pd.Series) -> pd.Series:
    """
    Convierte hora de texto (e.g. '2:43:23 pm') a datetime.time en TZ Panamá.
    Si ya viene como time/ts, intenta mantener.
    """
    if pd.api.types.is_datetime64_any_dtype(h):
        # Si viene timestamp, retornamos .dt.time
        return h.dt.tz_convert(PANAMA_TZ).dt.time

    # Intento robusto:
    parsed = pd.to_datetime(h, errors="coerce", format="%I:%M:%S %p")
    # Si no funcionó con ese formato, intento flexible:
    nulls = parsed.isna()
    if nulls.any():
        parsed.loc[nulls] = pd.to_datetime(h[nulls], errors="coerce", infer_datetime_format=True)

    return parsed.dt.time


def _detect_punto_cols(df: pd.DataFrame) -> list:
    """
    Detecta columnas tipo punto (punto1..puntoN).
    """
    return [c for c in df.columns if c.lower().startswith("punto")]


def _ensure_cycle_total(df: pd.DataFrame, punto_cols: list) -> pd.Series:
    """
    Si 'ciclo_total' (minutos) existe lo usa; si no, intenta calcular con
    la primera y última columna de 'puntoN' que tenga datos.
    Resultado en minutos (float).
    """
    if "ciclo_total" in df.columns:
        return pd.to_numeric(df["ciclo_total"], errors="coerce")

    if not punto_cols:
        return pd.Series(np.nan, index=df.index)

    # Tomar primera y última columna de puntos (por orden natural)
    punto_cols_sorted = sorted(
        punto_cols,
        key=lambda x: int(''.join([ch for ch in x if ch.isdigit()]) or 0)
    )
    first_col = punto_cols_sorted[0]
    last_col = punto_cols_sorted[-1]

    # Combinar fecha + hora para formar timestamps y calcular delta
    # Nota: necesitamos una fecha; usamos df['fecha'] si existe
    if "fecha" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    fecha_dt = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    # Parse horas
    first_h = pd.to_datetime(df[first_col], errors="coerce").dt.time
    last_h  = pd.to_datetime(df[last_col],  errors="coerce").dt.time

    # Construimos datetime con fecha + hora (naive) y localizamos a Panamá
    def build_ts(date_series, time_series):
        dt = pd.to_datetime(
            date_series.astype(str) + " " + pd.Series(time_series).astype(str),
            errors="coerce",
            infer_datetime_format=True
        )
        # Localizar a Panamá
        try:
            return dt.dt.tz_localize(PANAMA_TZ, nonexistent='NaT', ambiguous='NaT')
        except Exception:
            return dt  # fallback

    ts_first = build_ts(fecha_dt, first_h)
    ts_last  = build_ts(fecha_dt, last_h)

    delta = (ts_last - ts_first).dt.total_seconds() / 60.0
    return delta


def cargar_datos() -> pd.DataFrame:
    """
    Consulta la vista_resumen y normaliza tipos:
      - fecha -> date (Panamá)
      - horas de puntos -> time
      - ciclo_total -> minutos
    """
    try:
        engine = get_engine()
        df = pd.read_sql(QUERY, engine)

        # Normalizaciones mínimas
        # Renombrados opcionales para robustez (si algunas columnas se llaman diferente):
        # esperamos columnas: sesion_id, placa, fecha, ciclo_numero, device_cookie, punto1..puntoN, ciclo_total
        # Si 'ciclo' en lugar de 'ciclo_numero'
        if "ciclo" in df.columns and "ciclo_numero" not in df.columns:
            df = df.rename(columns={"ciclo": "ciclo_numero"})
        if "device_coockie" in df.columns:  # por si hay un typo previo
            df = df.rename(columns={"device_coockie": "device_cookie"})

        # Asegurar fecha
        if "fecha" in df.columns:
            df["fecha"] = _parse_fecha(df["fecha"])

        # Asegurar horas de los puntos
        punto_cols = _detect_punto_cols(df)
        for c in punto_cols:
            df[c] = _parse_hora(df[c])

        # Asegurar ciclo_total (minutos)
        df["ciclo_total"] = _ensure_cycle_total(df, punto_cols)

        # Ordenar por lo más reciente si hay 'sesion_id' y 'ciclo_numero'
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
app: Dash = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="QRLogix Dashboard")
server = app.server  # para gunicorn

# Carga inicial
df0 = cargar_datos()

def kpi_card(title, value, id_value=None):
    return dbc.Card(
        dbc.CardBody([
            html.div(title, className="text-muted small"),
            html.H4(id=id_value) if id_value else html.H4(value)
        ]),
        className="shadow-sm rounded-3"
    )

# Opciones de filtros iniciales
placas_opts = sorted([p for p in df0["placa"].dropna().unique()]) if "placa" in df0.columns else []
min_fecha = df0["fecha"].min() if "fecha" in df0.columns and not df0.empty else None
max_fecha = df0["fecha"].max() if "fecha" in df0.columns and not df0.empty else None

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Br(),
        dbc.Row([
            dbc.Col(html.H3("QRLogix • Dashboard de Escaneos", className="mb-0"), md=8),
            dbc.Col(html.Div(id="last-update", className="text-end text-muted"), md=4),
        ], align="center"),
        html.Hr(),

        # Filtros
        dbc.Row([
            dbc.Col([
                html.Label("Placa(s)"),
                dcc.Dropdown(
                    id="filtro-placa",
                    options=[{"label": p, "value": p} for p in placas_opts],
                    multi=True,
                    placeholder="Todas"
                ),
            ], md=4),
            dbc.Col([
                html.Label("Rango de fechas"),
                dcc.DatePickerRange(
                    id="filtro-fechas",
                    start_date=min_fecha,
                    end_date=max_fecha,
                    display_format="DD/MM/YYYY"
                ),
            ], md=5),
            dbc.Col([
                html.Label("Actualizar"),
                dcc.Interval(id="interval-refresh", interval=60_000, n_intervals=0),  # cada 60s
                dbc.Button("Actualizar ahora", id="btn-refresh", color="secondary", className="w-100")
            ], md=3),
        ], className="gy-3"),
        html.Br(),

        # KPIs
        dbc.Row([
            dbc.Col(kpi_card("Ciclos (filtrados)", "—", "kpi-ciclos"), md=3),
            dbc.Col(kpi_card("Escaneos (filtrados)", "—", "kpi-escaneos"), md=3),
            dbc.Col(kpi_card("Ciclo promedio (min)", "—", "kpi-ciclo-prom"), md=3),
            dbc.Col(kpi_card("Placas únicas", "—", "kpi-placas"), md=3),
        ], className="gy-3"),
        html.Br(),

        # Gráfico y tabla
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Tiempos por punto (promedio)", className="card-title"),
                        dcc.Graph(id="grafico-puntos")
                    ]),
                    className="shadow-sm rounded-3 h-100"
                )
            ], md=7),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Detalle (filtrado)", className="card-title"),
                        dash_table.DataTable(
                            id="tabla-detalle",
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": 12, "padding": "6px"},
                            style_header={"fontWeight": "bold"},
                        )
                    ]),
                    className="shadow-sm rounded-3 h-100"
                )
            ], md=5),
        ], className="gy-3"),

        html.Br()
    ],
)

# =========================
# Callbacks
# =========================
@app.callback(
    Output("last-update", "children"),
    Output("kpi-ciclos", "children"),
    Output("kpi-escaneos", "children"),
    Output("kpi-ciclo-prom", "children"),
    Output("kpi-placas", "children"),
    Output("grafico-puntos", "figure"),
    Output("tabla-detalle", "columns"),
    Output("tabla-detalle", "data"),
    Input("interval-refresh", "n_intervals"),
    Input("btn-refresh", "n_clicks"),
    State("filtro-placa", "value"),
    State("filtro-fechas", "start_date"),
    State("filtro-fechas", "end_date"),
    prevent_initial_call=False
)
def actualizar(_, __, placas_sel, start_date, end_date):
    df = cargar_datos()
    if df.empty:
        fig = px.line(title="Sin datos")
        return (
            "Última actualización: sin conexión",
            "—", "—", "—", "—",
            fig, [], []
        )

    # Filtrado por placa(s)
    if placas_sel:
        df = df[df["placa"].isin(placas_sel)]

    # Filtrado por fechas (si existen)
    if "fecha" in df.columns:
        if start_date:
            sd = pd.to_datetime(start_date).date()
            df = df[df["fecha"] >= sd]
        if end_date:
            ed = pd.to_datetime(end_date).date()
            df = df[df["fecha"] <= ed]

    # KPIs
    total_ciclos = df["ciclo_numero"].nunique() if "ciclo_numero" in df.columns else df.shape[0]
    # Escaneos: cuenta de columnas punto* no nulas por fila, sumadas
    punto_cols = _detect_punto_cols(df)
    escaneos = int(df[punto_cols].notna().sum().sum()) if punto_cols else 0

    ciclo_prom = np.nan
    if "ciclo_total" in df.columns:
        ciclo_prom = round(pd.to_numeric(df["ciclo_total"], errors="coerce").mean(), 2)
    kpi_ciclo_prom = "—" if np.isnan(ciclo_prom) else ciclo_prom

    placas_unicas = df["placa"].nunique() if "placa" in df.columns else 0

    # Gráfico: tiempos promedio por punto (en minutos desde el primer punto si hay 'ciclo_total', de lo contrario count)
    # Para algo rápido: porcentaje de filas con dato por punto
    if punto_cols:
        melt = df.melt(id_vars=[c for c in ["sesion_id", "placa", "fecha", "ciclo_numero"] if c in df.columns],
                       value_vars=punto_cols, var_name="punto", value_name="hora")
        resumen = melt.groupby("punto", as_index=False).agg(
            registros=("hora", lambda s: s.notna().sum())
        )
        fig = px.bar(resumen, x="punto", y="registros", title="Registros por punto (conteo)")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    else:
        fig = px.bar(title="Sin columnas de punto")

    # Tabla
    # Mostramos columnas clave primero si existen:
    cols_prioridad = [c for c in ["sesion_id", "fecha", "ciclo_numero", "placa", "device_cookie"] if c in df.columns]
    cols_puntos = punto_cols
    cols_rest = [c for c in df.columns if c not in cols_prioridad + cols_puntos]
    cols_final = cols_prioridad + cols_puntos + cols_rest

    tabla_cols = [{"name": c, "id": c} for c in cols_final]
    tabla_data = df[cols_final].to_dict("records")

    # Última actualización (hora local Panamá)
    now_pa = datetime.now(PANAMA_TZ).strftime("%d/%m/%Y %I:%M:%S %p")
    last_update = f"Última actualización: {now_pa}"

    return (
        last_update,
        f"{total_ciclos:,}",
        f"{escaneos:,}",
        f"{kpi_ciclo_prom}",
        f"{placas_unicas:,}",
        fig,
        tabla_cols,
        tabla_data
    )


if __name__ == "__main__":
    # Para desarrollo local
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=True)