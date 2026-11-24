from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kexia_cancer.simulation.models import ResistanceParameters, SinglePopulationParameters, simulate_resistance, simulate_single

DATA = ROOT / "data" / "processed" / "breast_gdsc2_modeling.csv.gz"
CURVES = ROOT / "data" / "processed" / "breast_gdsc2_normalized_dose_response.csv.gz"
MODEL = ROOT / "artifacts" / "models" / "gdsc2_breast_ln_ic50.joblib"
EVALUATION = ROOT / "artifacts" / "reports" / "model_evaluation.json"
HYPOTHESES = ROOT / "artifacts" / "reports" / "hypotheses.json"
PHASE2_REPORTS = ROOT / "artifacts" / "reports"

st.set_page_config(page_title="Kexia Cancer Research AI", page_icon="🧬", layout="wide")
st.markdown(
    """
    <style>
    :root { --k-blue:#1864ab; --k-green:#2b8a3e; --k-gray:#495057; }
    .block-container {padding-top: 1.4rem;}
    [data-testid="stMetric"] {background:#f8fafc;border:1px solid #dbe4ff;border-radius:12px;padding:12px;}
    .k-note {background:#eef7ff;border-left:4px solid #1864ab;padding:.8rem 1rem;border-radius:4px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Kexia Cancer Research AI")
st.caption("Investigación preclínica computacional · No recomienda tratamientos ni demuestra curas")

if not DATA.exists():
    st.error("No existe el conjunto procesado. Ejecute `python scripts/prepare_breast_cancer.py`.")
    st.stop()


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA, low_memory=False)


@st.cache_data
def load_curves() -> pd.DataFrame:
    return pd.read_csv(CURVES, low_memory=False) if CURVES.exists() else pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL) if MODEL.exists() else None


data = load_data()
curves = load_curves()
artifact = load_model()
tabs = st.tabs([
    "Explorador de datos", "Motor de IA", "Simulación matemática", "Comparador",
    "Laboratorio de hipótesis", "Investigación y validación científica",
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelos de mama", data["SANGER_MODEL_ID"].nunique())
    c2.metric("Compuestos", data["DRUG_ID"].nunique())
    c3.metric("Observaciones ajustadas", f"{len(data):,}")
    c4.metric("RMSE mediana", f"{data['RMSE'].median():.3f}")
    st.markdown(
        '<div class="k-note"><b>Definiciones:</b> LN_IC50 = ln(IC50 en µM). AUC es el área fraccional del ajuste dentro del rango de concentraciones GDSC2; no es eficacia clínica.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        metric = st.selectbox("Distribución", ["LN_IC50", "AUC", "RMSE", "IC50_uM"])
        st.plotly_chart(px.histogram(data, x=metric, nbins=50, color_discrete_sequence=["#1864ab"]), use_container_width=True)
    with right:
        st.plotly_chart(
            px.scatter(data, x="LN_IC50", y="AUC", color="RMSE", hover_data=["CELL_LINE_NAME", "DRUG_NAME"], color_continuous_scale="Viridis"),
            use_container_width=True,
        )
    selected_cell = st.selectbox("Modelo celular", sorted(data["CELL_LINE_NAME"].dropna().unique()), key="explorer_cell")
    subset = data[data["CELL_LINE_NAME"] == selected_cell]
    feature_columns = [c for c in data if c.startswith("MUT_") or c.startswith("CN_")]
    st.dataframe(
        subset[["SANGER_MODEL_ID", "CELL_LINE_NAME", "DRUG_ID", "DRUG_NAME", "PUTATIVE_TARGET", "PATHWAY_NAME", "LN_IC50", "AUC", "RMSE"]].sort_values("LN_IC50"),
        use_container_width=True,
        hide_index=True,
    )
    if not curves.empty:
        selected_id = subset["SANGER_MODEL_ID"].iloc[0]
        available_ids = set(curves.loc[curves["SANGER_MODEL_ID"] == selected_id, "DRUG_ID"].astype(int))
        curve_drugs = subset[subset["DRUG_ID"].astype(int).isin(available_ids)][["DRUG_ID", "DRUG_NAME"]].drop_duplicates()
        if not curve_drugs.empty:
            curve_name = st.selectbox("Curva experimental normalizada", sorted(curve_drugs["DRUG_NAME"]), key="curve_drug")
            curve_id = int(curve_drugs.loc[curve_drugs["DRUG_NAME"] == curve_name, "DRUG_ID"].iloc[0])
            curve = curves[(curves["SANGER_MODEL_ID"] == selected_id) & (curves["DRUG_ID"].astype(int) == curve_id)].sort_values("CONC")
            fig_curve = go.Figure(
                go.Scatter(
                    x=curve["CONC"], y=curve["viability_mean"], mode="markers+lines",
                    error_y=dict(type="data", array=curve["viability_sd"].fillna(0)),
                    text=[f"n={n}" for n in curve["replicates"]], marker_color="#1864ab",
                )
            )
            fig_curve.update_xaxes(type="log", title="Concentración (µM)")
            fig_curve.update_yaxes(title="Viabilidad normalizada (sin recortar)")
            fig_curve.update_layout(title=f"{selected_cell} · {curve_name} · observación experimental raw normalizada")
            st.plotly_chart(fig_curve, use_container_width=True)
    with st.expander("Características moleculares verificadas disponibles"):
        st.dataframe(subset[["SANGER_MODEL_ID"] + feature_columns].drop_duplicates(), use_container_width=True, hide_index=True)

with tabs[1]:
    if artifact is None:
        st.warning("Modelo no entrenado. Ejecute `python scripts/train_models.py`.")
    else:
        cell_name = st.selectbox("Línea celular", sorted(data["CELL_LINE_NAME"].unique()), key="ai_cell")
        cell_rows = data[data["CELL_LINE_NAME"] == cell_name]
        drug_name = st.selectbox("Compuesto", sorted(cell_rows["DRUG_NAME"].unique()), key="ai_drug")
        row = cell_rows[cell_rows["DRUG_NAME"] == drug_name].iloc[[0]]
        pred = artifact.predict(row).iloc[0]
        obs = row.iloc[0]
        a, b, c = st.columns(3)
        a.metric("LN_IC50 predicho", f"{pred.prediction:.3f}")
        b.metric("Intervalo marginal 95%", f"[{pred.lower_95:.3f}, {pred.upper_95:.3f}]")
        c.metric("LN_IC50 experimental", f"{obs.LN_IC50:.3f}")
        st.caption("Predicción de IA · intervalo split-conformal. Observación experimental: ajuste GDSC2 de viabilidad a 72 h.")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Experimental", "IA"], y=[obs.LN_IC50, pred.prediction], marker_color=["#2b8a3e", "#1864ab"]))
        fig.add_trace(go.Scatter(x=["IA"], y=[pred.prediction], error_y=dict(type="data", array=[pred.upper_95-pred.prediction], arrayminus=[pred.prediction-pred.lower_95]), mode="markers", marker_color="#1864ab", showlegend=False))
        fig.update_layout(yaxis_title="LN_IC50 (ln µM)")
        st.plotly_chart(fig, use_container_width=True)
        if EVALUATION.exists():
            evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
            st.subheader("Validación sin fuga por protocolo")
            metric_rows = [{"protocolo": k, **v["metrics"], **v["split"]} for k, v in evaluation["protocols"].items()]
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        st.warning("LN_IC50 no se transforma en una tasa de muerte. La simulación dinámica exige parámetros adicionales explícitos.")

with tabs[2]:
    st.subheader("Simulación hipotética no calibrada")
    col1, col2, col3 = st.columns(3)
    with col1:
        n0 = st.number_input("Población inicial", min_value=0.0, value=100_000.0, format="%.0f")
        r = st.number_input("Proliferación r (h⁻¹)", min_value=0.0, value=0.03, format="%.4f")
        capacity = st.number_input("Capacidad K", min_value=1.0, value=10_000_000.0, format="%.0f")
    with col2:
        concentration = st.number_input("Concentración (µM)", min_value=0.0, value=1.0)
        kmax = st.number_input("kmax (h⁻¹)", min_value=0.0, value=0.04, format="%.4f")
        ec50 = st.number_input("EC50 dinámica hipotética (µM)", min_value=1e-6, value=1.0)
    with col3:
        hill = st.number_input("Coeficiente de Hill", min_value=0.05, value=1.0)
        duration = st.number_input("Duración (h)", min_value=1.0, value=240.0)
        resistant = st.toggle("Dos subpoblaciones resistentes")
    base = SinglePopulationParameters(n0, r, capacity, concentration, kmax, ec50, hill)
    untreated = simulate_single(SinglePopulationParameters(n0, r, capacity, 0.0, 0.0, ec50, hill), duration_hours=duration)
    untreated["escenario"] = "sin intervención"
    if resistant:
        resistant_fraction = st.slider("Fracción resistente inicial", 0.0, 1.0, 0.01)
        treated = simulate_resistance(ResistanceParameters(**base.__dict__, initial_resistant_fraction=resistant_fraction), duration_hours=duration)
    else:
        treated = simulate_single(base, duration_hours=duration)
    treated["escenario"] = "intervención hipotética"
    trajectories = pd.concat([untreated, treated], ignore_index=True)
    st.plotly_chart(px.line(trajectories, x="time_hours", y="population", color="escenario", labels={"population":"Población celular", "time_hours":"Tiempo (h)"}), use_container_width=True)
    if resistant:
        st.plotly_chart(px.line(treated, x="time_hours", y=["sensitive", "resistant"]), use_container_width=True)
    st.info("EC50 de esta función de tasa no equivale automáticamente al IC50 de viabilidad GDSC2. Este escenario no está calibrado con datos temporales.")

with tabs[3]:
    if artifact is None:
        st.warning("Entrene el modelo para habilitar comparaciones predictivas.")
    else:
        cell_name = st.selectbox("Línea celular para comparar", sorted(data["CELL_LINE_NAME"].unique()), key="compare_cell")
        rows = data[data["CELL_LINE_NAME"] == cell_name].copy()
        selections = st.multiselect("Compuestos", sorted(rows["DRUG_NAME"].unique()), max_selections=8)
        if selections:
            selected = rows[rows["DRUG_NAME"].isin(selections)].drop_duplicates("DRUG_NAME").copy()
            predictions = artifact.predict(selected)
            selected["LN_IC50_predicho"] = predictions["prediction"].to_numpy()
            selected["error_minus"] = predictions["prediction"].to_numpy() - predictions["lower_95"].to_numpy()
            selected["error_plus"] = predictions["upper_95"].to_numpy() - predictions["prediction"].to_numpy()
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Experimental", x=selected["DRUG_NAME"], y=selected["LN_IC50"], marker_color="#2b8a3e"))
            fig.add_trace(go.Bar(name="IA", x=selected["DRUG_NAME"], y=selected["LN_IC50_predicho"], marker_color="#1864ab", error_y=dict(type="data", array=selected["error_plus"], arrayminus=selected["error_minus"])))
            fig.update_layout(barmode="group", yaxis_title="LN_IC50 (ln µM)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(selected[["DRUG_NAME", "DRUG_ID", "MIN_CONC", "MAX_CONC", "LN_IC50", "LN_IC50_predicho", "AUC", "RMSE"]], use_container_width=True, hide_index=True)
            st.caption("Sólo se comparan cantidades LN_IC50 con LN_IC50. No se combinan AUC, tasas dinámicas o puntuaciones clínicas.")

with tabs[4]:
    st.write("Guarde hipótesis reproducibles con parámetros y notas. Los registros no son recomendaciones médicas.")
    hypothesis_name = st.text_input("Nombre del escenario")
    notes = st.text_area("Notas y supuestos")
    if st.button("Guardar escenario", type="primary"):
        existing = json.loads(HYPOTHESES.read_text(encoding="utf-8")) if HYPOTHESES.exists() else []
        record = {
            "name": hypothesis_name or f"scenario-{len(existing)+1}",
            "notes": notes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": base.__dict__ if "base" in locals() else {},
            "label": "Simulación hipotética no calibrada",
            "software": "Kexia Cancer Research AI 0.1.0",
        }
        existing.append(record)
        HYPOTHESES.parent.mkdir(parents=True, exist_ok=True)
        HYPOTHESES.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        st.success(f"Escenario guardado: {record['name']}")
    if HYPOTHESES.exists():
        existing = json.loads(HYPOTHESES.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame(existing), use_container_width=True, hide_index=True)
        st.download_button("Descargar registro JSON", HYPOTHESES.read_bytes(), "kexia_hypotheses.json", "application/json")

with tabs[5]:
    st.subheader("Investigación y validación científica")
    st.caption(
        "Auditoría reproducible de generalización. Verde = medición experimental; azul = predicción de IA; "
        "las simulaciones dinámicas conservan su etiqueta hipotética cuando no existe calibración temporal."
    )
    comparison_path = PHASE2_REPORTS / "phase2_model_comparison.csv"
    if not comparison_path.exists():
        st.info("Ejecute `python scripts/run_phase2_experiments.py` para generar la auditoría de Fase 2.")
    else:
        comparison = pd.read_csv(comparison_path)
        main_versions = comparison[
            ~comparison["variant"].str.startswith("ablate_")
            & comparison["variant"].ne("chemical_descriptors")
        ].copy()
        ablations = comparison[comparison["variant"].str.startswith("ablate_")].copy()
        st.markdown("#### Comparación entre versiones")
        protocol = st.selectbox("Protocolo de evaluación independiente", sorted(main_versions["protocol"].unique()))
        shown = main_versions[main_versions["protocol"] == protocol]
        st.plotly_chart(
            px.bar(shown, x="variant", y="r2", color="variant", hover_data=["mae", "rmse", "test_rows"],
                   labels={"variant": "Versión", "r2": "R² fuera de muestra"}),
            use_container_width=True,
        )
        st.dataframe(shown[["variant", "mae", "rmse", "r2", "pearson", "spearman", "train_rows", "test_rows"]],
                     use_container_width=True, hide_index=True)
        if not ablations.empty:
            st.markdown("#### Ablaciones")
            ablation_protocol = st.selectbox("Protocolo de ablación", sorted(ablations["protocol"].unique()))
            ablation_view = ablations[ablations["protocol"] == ablation_protocol]
            current_row = comparison[(comparison["protocol"] == ablation_protocol) & (comparison["variant"] == "current")]
            if not current_row.empty:
                ablation_view = pd.concat([current_row, ablation_view], ignore_index=True)
            st.plotly_chart(px.bar(ablation_view, x="variant", y="r2", color="variant"), use_container_width=True)

        repeated_path = PHASE2_REPORTS / "phase2_repeated_group_validation_summary.csv"
        if repeated_path.exists():
            st.markdown("#### Validación repetida por grupos")
            repeated = pd.read_csv(repeated_path)
            st.plotly_chart(px.bar(repeated, x="variant", y="r2_mean", color="protocol", barmode="group",
                                   error_y="r2_sd", hover_data=["mae_mean", "mae_sd", "repeats"]),
                            use_container_width=True)
        scaffold_path = PHASE2_REPORTS / "phase2_scaffold_evaluation.csv"
        if scaffold_path.exists():
            st.markdown("#### Generalización a andamiajes químicos no vistos")
            scaffold = pd.read_csv(scaffold_path)
            st.plotly_chart(px.bar(scaffold, x="variant", y="r2", color="variant",
                                   hover_data=["mae", "rmse", "test_rows"]), use_container_width=True)

        importance_path = PHASE2_REPORTS / "phase2_feature_importance.csv"
        if importance_path.exists():
            st.markdown("#### Importancia y estabilidad de características")
            importance = pd.read_csv(importance_path).head(20).sort_values("importance_mae_mean")
            st.plotly_chart(px.bar(importance, x="importance_mae_mean", y="feature", orientation="h",
                                   error_x="importance_mae_sd", labels={"importance_mae_mean": "Aumento de MAE al permutar"}),
                            use_container_width=True)
            st.caption("Importancia por permutación en células no vistas; las barras muestran variación entre permutaciones.")

        left, right = st.columns(2)
        cell_errors_path = PHASE2_REPORTS / "phase2_errors_by_cell.csv"
        compound_errors_path = PHASE2_REPORTS / "phase2_errors_by_compound.csv"
        with left:
            st.markdown("#### Errores por línea celular")
            if cell_errors_path.exists():
                cell_errors = pd.read_csv(cell_errors_path).sort_values("mae", ascending=False)
                st.dataframe(cell_errors, use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Errores por compuesto")
            if compound_errors_path.exists():
                compound_errors = pd.read_csv(compound_errors_path).sort_values("mae", ascending=False)
                st.dataframe(compound_errors.head(50), use_container_width=True, hide_index=True)

        interval_path = PHASE2_REPORTS / "phase2_interval_calibration.csv"
        if interval_path.exists():
            st.markdown("#### Calibración de incertidumbre")
            intervals = pd.read_csv(interval_path)
            st.plotly_chart(px.bar(intervals, x="protocol", y="empirical_coverage", color="protocol",
                                   hover_data=["half_width", "calibration_groups", "test_groups"]), use_container_width=True)
            st.dataframe(intervals, use_container_width=True, hide_index=True)

    math_path = PHASE2_REPORTS / "phase2_mathematical_audit.json"
    st.markdown("#### Curvas observadas, ajuste y simulación dinámica")
    if math_path.exists():
        math_audit = json.loads(math_path.read_text(encoding="utf-8"))
        hill = math_audit["hill_endpoint_models"]
        norm = math_audit["normalization"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pozos normalizados > 1", f"{100 * norm['fraction_above_one']:.2f}%")
        c2.metric("MAE de extrapolación Hill", f"{hill['extrapolation_mae']:.4f}" if hill["extrapolation_mae"] is not None else "N/D")
        c3.metric("EC50 dentro del rango", f"{100 * hill['fraction_ec50_inside_tested_range']:.1f}%" if hill["fraction_ec50_inside_tested_range"] is not None else "N/D")
        st.warning(hill["dynamic_validation_status"])
        extrapolation_path = PHASE2_REPORTS / "phase2_hill_extrapolation.csv"
        if extrapolation_path.exists():
            extrapolation = pd.read_csv(extrapolation_path)
            st.plotly_chart(px.scatter(extrapolation, x="observed_viability", y="predicted_viability",
                                       hover_data=["SANGER_MODEL_ID", "DRUG_ID", "held_out_concentration_uM"],
                                       labels={"observed_viability": "Experimental observado", "predicted_viability": "Curva Hill predicha"}),
                            use_container_width=True)
    else:
        st.info("Ejecute `python scripts/audit_mathematical_models.py` para mostrar el diagnóstico ampliado.")
    temporal_path = PHASE2_REPORTS / "phase2_temporal_validation.json"
    if temporal_path.exists():
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        st.markdown("#### Evidencia temporal independiente")
        t1, t2 = st.columns(2)
        t1.metric("Experimentos logísticos sin intervención", temporal["experiments"])
        t2.metric("NRMSE temporal mediana", f"{temporal['median_test_nrmse']:.3f}" if temporal["median_test_nrmse"] is not None else "N/D")
        st.caption(temporal["design"])
        st.warning(temporal["drug_dynamics_status"])
        temporal_predictions = PHASE2_REPORTS / "phase2_temporal_logistic_predictions.csv"
        if temporal_predictions.exists():
            time_predictions = pd.read_csv(temporal_predictions)
            cell = st.selectbox("Línea temporal", sorted(time_predictions["cell_line"].unique()), key="temporal_cell")
            time_view = time_predictions[time_predictions["cell_line"] == cell]
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=time_view["elapsed_hours"], y=time_view["observed_cell_count"],
                                        name="Experimental", mode="markers", marker_color="#2b8a3e"))
            figure.add_trace(go.Scatter(x=time_view["elapsed_hours"], y=time_view["predicted_cell_count"],
                                        name="Logística predicha", mode="lines", line_color="#1864ab"))
            figure.update_layout(xaxis_title="Tiempo (h)", yaxis_title="Conteo celular por pozo")
            st.plotly_chart(figure, use_container_width=True)

