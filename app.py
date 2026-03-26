'''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import config

st.set_page_config(page_title="Diabetes Health Dashboard", page_icon="🧬", layout="wide")


@st.cache_resource
def load_and_prepare():
    data = pd.read_csv(config.DATA_PATH)
    X = data.drop(config.TARGET_COLUMN, axis=1)
    from sklearn.model_selection import train_test_split
    y = data[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    model_path = "models/federated_model.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        fallback_path = "models/centralized_model.keras"
        if os.path.exists(fallback_path):
            model = tf.keras.models.load_model(fallback_path)
        else:
            model = None

    return scaler, model


@st.cache_resource
def load_eval_data():
    data = pd.read_csv(config.DATA_PATH)
    X = data.drop(config.TARGET_COLUMN, axis=1)
    y = data[config.TARGET_COLUMN].values.astype(int)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test, X.columns.tolist()


@st.cache_resource
def load_models_for_comparison():
    cen_path = "models/centralized_model.keras"
    fed_path = "models/federated_model.keras"

    cen_model = tf.keras.models.load_model(cen_path) if os.path.exists(cen_path) else None
    fed_model = tf.keras.models.load_model(fed_path) if os.path.exists(fed_path) else None
    return cen_model, fed_model


def compute_metrics_from_model(model, X_test_scaled, y_test, threshold=0.5):
    if model is None:
        return None

    out = model.predict(X_test_scaled, verbose=0).reshape(-1)

    if np.min(out) < 0 or np.max(out) > 1:
        probs = 1 / (1 + np.exp(-out))
    else:
        probs = out

    preds = (probs >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (y_test == 1)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def model_comparison_page():
    st.title("Centralized vs Federated Model Comparison")

    X_test_scaled, y_test, feature_cols = load_eval_data()
    cen_model, fed_model = load_models_for_comparison()

    if cen_model is None and fed_model is None:
        st.error("No models found. Expected: models/centralized_model.keras and/or models/federated_model.keras")
        return

    st.sidebar.markdown("### Comparison Settings")
    threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    cen_metrics = compute_metrics_from_model(cen_model, X_test_scaled, y_test, threshold=threshold)
    fed_metrics = compute_metrics_from_model(fed_model, X_test_scaled, y_test, threshold=threshold)

    rows = []
    if cen_metrics is not None:
        rows.append({
            "Model": "Centralized",
            "Accuracy": cen_metrics["accuracy"],
            "Precision": cen_metrics["precision"],
            "Recall": cen_metrics["recall"],
            "F1-Score": cen_metrics["f1"],
            "TP": cen_metrics["tp"],
            "TN": cen_metrics["tn"],
            "FP": cen_metrics["fp"],
            "FN": cen_metrics["fn"],
        })
    if fed_metrics is not None:
        rows.append({
            "Model": "Federated",
            "Accuracy": fed_metrics["accuracy"],
            "Precision": fed_metrics["precision"],
            "Recall": fed_metrics["recall"],
            "F1-Score": fed_metrics["f1"],
            "TP": fed_metrics["tp"],
            "TN": fed_metrics["tn"],
            "FP": fed_metrics["fp"],
            "FN": fed_metrics["fn"],
        })

    df = pd.DataFrame(rows).set_index("Model")

    st.subheader("Metrics Table")
    st.dataframe(df.style.format({
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
    }))

    st.subheader("Metric Comparison Chart")
    try:
        import plotly.graph_objects as go

        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
        fig = go.Figure()

        if cen_metrics is not None:
            fig.add_trace(go.Bar(
                name="Centralized",
                x=metrics_to_plot,
                y=[df.loc["Centralized", m] for m in metrics_to_plot]
            ))
        if fed_metrics is not None:
            fig.add_trace(go.Bar(
                name="Federated",
                x=metrics_to_plot,
                y=[df.loc["Federated", m] for m in metrics_to_plot]
            ))

        fig.update_layout(
            barmode="group",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Plotly chart could not be rendered. The table above still shows the full comparison.")

    st.subheader("Confusion Matrix Counts")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Centralized**")
        if cen_metrics is None:
            st.write("Model not found.")
        else:
            st.write(f"TP: {cen_metrics['tp']} | FP: {cen_metrics['fp']}")
            st.write(f"FN: {cen_metrics['fn']} | TN: {cen_metrics['tn']}")

    with c2:
        st.markdown("**Federated**")
        if fed_metrics is None:
            st.write("Model not found.")
        else:
            st.write(f"TP: {fed_metrics['tp']} | FP: {fed_metrics['fp']}")
            st.write(f"FN: {fed_metrics['fn']} | TN: {fed_metrics['tn']}")


def create_mock_scatter(glucose_val):
    import plotly.graph_objects as go
    np.random.seed(42)
    x = [f"{i}am" if i <= 12 else f"{i-12}pm" for i in range(1, 25)]
    y_base = np.random.normal(loc=glucose_val, scale=15, size=24)

    fig = go.Figure()

    for i in range(24):
        num_points = np.random.randint(1, 5)
        points = np.random.normal(loc=y_base[i], scale=5, size=num_points)
        fig.add_trace(go.Scatter(
            x=[x[i]] * num_points,
            y=points,
            mode='markers',
            marker=dict(size=np.random.uniform(8, 14, num_points), color='#00BFA5', opacity=0.7),
            hoverinfo='skip'
        ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                   tickfont=dict(color='#888', size=10), tickvals=[0, 4, 8, 12, 16, 20],
                   ticklen=0, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        height=180
    )
    return fig


def health_dashboard_page():
    scaler, model = load_and_prepare()

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

        html, body, p, div, h1, h2, h3, h4, h5, h6, span, label {
            font-family: 'Outfit', sans-serif;
        }

        .material-icons, .material-symbols-rounded {
            font-family: 'Material Symbols Rounded' !important;
        }

        .stApp {
            background-color: #F4F7F6;
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 191, 165, 0.05) 0%, transparent 50%),
                              radial-gradient(circle at 90% 80%, rgba(41, 98, 255, 0.05) 0%, transparent 50%);
        }

        [data-testid="stMainBlockContainer"] {
            padding: 2rem 3rem 1rem 3rem !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .element-container { margin-bottom: 0px !important; }

        .risk-alert {
            animation: pulse-border 2s infinite;
        }
        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); }
            100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
        }

        .stButton>button {
            border-radius: 12px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 191, 165, 0.3);
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 191, 165, 0.4);
        }
        </style>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html("""
    <script>
        const doc = window.parent.document;

        function enforceLayout() {
            const blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
            if (blocks.length >= 2) {
                blocks[0].style.height = 'calc(100vh - 220px)';
                blocks[0].style.marginBottom = '20px';

                blocks[0].querySelectorAll('div[data-testid="column"]').forEach((col) => {
                    col.style.height = 'calc(100vh - 220px)';
                    col.style.background = 'rgba(255, 255, 255, 0.95)';
                    col.style.backdropFilter = 'blur(20px)';
                    col.style.borderRadius = '24px';
                    col.style.padding = '24px';
                    col.style.boxShadow = '0 4px 24px rgba(0, 0, 0, 0.04)';
                    col.style.border = '1px solid white';
                    col.style.display = 'flex';
                    col.style.flexDirection = 'column';
                    col.style.justifyContent = 'space-between';
                });

                blocks[1].style.height = '100px';

                blocks[1].querySelectorAll('div[data-testid="column"]').forEach((col) => {
                    col.style.height = '100px';
                    col.style.background = 'rgba(255, 255, 255, 0.95)';
                    col.style.backdropFilter = 'blur(20px)';
                    col.style.borderRadius = '20px';
                    col.style.padding = '16px';
                    col.style.boxShadow = '0 4px 24px rgba(0, 0, 0, 0.04)';
                    col.style.border = '1px solid white';
                    col.style.display = 'flex';
                    col.style.justifyContent = 'center';
                });
            }
        }

        enforceLayout();
        setTimeout(enforceLayout, 500);
        setTimeout(enforceLayout, 1500);
    </script>
    """, height=0, width=0)

    st.sidebar.markdown("<h2 style='color: #111; margin-bottom: 20px;'>Patient Vitals</h2>", unsafe_allow_html=True)

    with st.sidebar.expander("Demographics & Vitals", expanded=True):
        gender = st.selectbox("Sex", ["Female", "Male"])
        age = st.slider("Age", 0, 100, 33)

        height = st.slider("Height (cm)", 100, 220, 170)
        weight = st.slider("Weight (kg)", 30, 150, 70)

        # BMI calculation
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)

        blood_pressure = st.slider("Blood Pressure", 0, 140, 70)


        #pregnancies = st.slider("Pregnancies", 0, 20, 1)
        if gender == "Male":
            pregnancies = 0
            st.slider("Pregnancies", 0, 20, 0, disabled=True)
        else:
            pregnancies = st.slider("Pregnancies", 0, 20, 0)


    with st.sidebar.expander("Lab Results", expanded=True):
        glucose = st.slider("Glucose Level", 0, 250, 100)
        insulin = st.slider("Insulin", 0, 900, 79)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
    ]], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    r1_col1, r1_col2, r1_col3 = st.columns([1, 1.5, 1], gap="medium")

    with r1_col1:
        st.markdown("<div style='font-size: 26px; font-weight: 800; color: #00BFA5; line-height: 1.1; margin-bottom: 8px;'>Health Overview</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 14px; color: #546E7A; line-height: 1.3; margin-bottom: 24px;'>AI-driven analysis of your current vitals. Identifying key risk factors for Type 2 Diabetes.</div>", unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background: rgba(0, 191, 165, 0.1); padding: 16px; border-radius: 12px; text-align:center; border: 1px solid rgba(0,191,165,0.2); margin-bottom:12px;">
                <div style="font-size:12px; color:#00BFA5; font-weight:700;">GLUCOSE LEVEL</div>
                <div style="font-size:32px; font-weight:800; color:#1C2833;">{glucose} <span style="font-size:14px; color:#78909C;">mg/dL</span></div>
            </div>

            <div style="background: rgba(41, 98, 255, 0.05); padding: 16px; border-radius: 12px; text-align:center; border: 1px solid rgba(41,98,255,0.1);">
                <div style="font-size:12px; color:#2962FF; font-weight:700;">BMI STATUS</div>
                <div style="font-size:32px; font-weight:800; color:#1C2833;">{bmi} <span style="font-size:14px; color:#78909C;">kg/m²</span></div>
            </div>
        """, unsafe_allow_html=True)

    with r1_col2:
        st.markdown("<div style='font-size: 14px; color: #5D6D7E; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;'>Glucose Variability Tracker</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px; color:#90A4AE; margin-bottom:12px;'>24H Scatter Simulation Based on Vitals</div>", unsafe_allow_html=True)
        fig_scatter = create_mock_scatter(glucose)
        fig_scatter.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

        st.markdown(f"""
            <div style="margin-top:20px; background: #FFEBEE; padding: 16px; border-radius: 12px; text-align:center; border: 1px solid rgba(229, 57, 53, 0.1);">
                <div style="font-size:12px; color:#E53935; font-weight:700;">BLOOD PRESSURE</div>
                <div style="font-size:32px; font-weight:800; color:#1C2833;">{blood_pressure} <span style="font-size:14px; color:#78909C;">mmHg</span></div>
            </div>
        """, unsafe_allow_html=True)

    with r1_col3:
        placeholder = st.empty()

        with placeholder.container():
            st.markdown("""
                <div style="text-align:center;">
                    <div style="font-size: 48px; margin-bottom: 8px;">🩺</div>
                    <div style="font-size: 22px; font-weight: 700; color: #1C2833;">Ready for Diagnosis</div>
                    <p style="color: #85929E; font-size: 14px; line-height:1.4; margin-top:8px; margin-bottom:24px;">Ensure all patient vitals in the sidebar are accurate prior to execution.</p>
                </div>
            """, unsafe_allow_html=True)
            run_diag = st.button("Run Risk Engine", type="primary")

        if run_diag and model is not None:
            input_scaled = scaler.transform(input_data)
            prediction_prob = model.predict(input_scaled)[0][0]

            with placeholder.container():
                if prediction_prob >= 0.7:
                    # HIGH RISK
                    st.markdown(f"""
                        <div class="risk-alert" style="padding: 24px; background: #FDEDEC; border: 2px solid #E74C3C; border-radius: 16px; text-align:center;">
                            <div style="color: #E74C3C; font-size:13px; font-weight:700; text-transform: uppercase;">Diagnosis Result</div>
                            <div style="font-size: 42px; font-weight: 900; color: #E74C3C;">HIGH RISK</div>
                            <div style="font-size: 20px; color: #C0392B; font-weight:700;">{(prediction_prob*100):.1f}% Prob.</div>
                            <div style="font-size:13px; color:#922B21;">Consult doctor immediately.</div>
                        </div>
                    """, unsafe_allow_html=True)

                elif prediction_prob >= 0.4:
                    # MEDIUM RISK
                    st.markdown(f"""
                        <div style="padding: 24px; background: #FFF8E1; border: 2px solid #F39C12; border-radius: 16px; text-align:center;">
                            <div style="color: #F39C12; font-size:13px; font-weight:700; text-transform: uppercase;">Diagnosis Result</div>
                            <div style="font-size: 42px; font-weight: 900; color: #F39C12;">MEDIUM RISK</div>
                            <div style="font-size: 20px; color: #D68910; font-weight:700;">{(prediction_prob*100):.1f}% Prob.</div>
                            <div style="font-size:13px; color:#AF601A;">Monitor lifestyle and recheck.</div>
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    # LOW RISK
                    st.markdown(f"""
                        <div style="padding: 24px; background: #E8F8F5; border: 2px solid #1ABC9C; border-radius: 16px; text-align:center;">
                            <div style="color: #1ABC9C; font-size:13px; font-weight:700; text-transform: uppercase;">Diagnosis Result</div>
                            <div style="font-size: 42px; font-weight: 900; color: #1ABC9C;">LOW RISK</div>
                            <div style="font-size: 20px; color: #117864; font-weight:700;">{(prediction_prob*100):.1f}% Prob.</div>
                            <div style="font-size:13px; color:#148F77;">Vitals within safe range.</div>
                        </div>
                    """, unsafe_allow_html=True)

    r2_cols = st.columns(4, gap="medium")

    metrics = [
        ("Insulin Level", insulin, "mu U/ml", "🧪", "#F3E5F5", "#8E24AA"),
        ("Pregnancies", pregnancies, "count", "🤰", "#E3F2FD", "#1976D2"),
        ("Skin Thickness", skin_thickness, "mm", "📏", "#FFF3E0", "#FF9800"),
        ("Pedigree Func", f"{dpf:.2f}", "", "🧬", "#E8EAF6", "#3949AB")
    ]

    for i, col in enumerate(r2_cols):
        bg, accent = metrics[i][4], metrics[i][5]
        with col:
            st.markdown(f"""
                <div style="display:flex; flex-direction:row; align-items:center;">
                    <div style="background:{bg}; width:48px; height:48px; min-width:48px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:24px; margin-right:16px;">
                        {metrics[i][3]}
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #78909C; font-weight: 700; text-transform: uppercase;">{metrics[i][0]}</div>
                        <div style="font-size: 22px; font-weight: 800; color: {accent}; line-height: 1.1;">{metrics[i][1]} <span style="font-size:12px; color:#90A4AE; font-weight:600;">{metrics[i][2]}</span></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


def main():
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Health Dashboard", "Model Comparison"],
        label_visibility="collapsed"
    )

    if page == "Health Dashboard":
        health_dashboard_page()
    else:
        model_comparison_page()


if __name__ == "__main__":
    main()
'''

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
import config

st.set_page_config(page_title="Diabetes Health Dashboard", page_icon="🧬", layout="wide")

SCALER_PATH = "models/scaler.pkl"


@st.cache_resource
def load_and_prepare():
    if not os.path.exists(SCALER_PATH):
        return None, None

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model_path = "models/federated_model.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        fallback_path = "models/centralized_model.keras"
        if os.path.exists(fallback_path):
            model = tf.keras.models.load_model(fallback_path)
        else:
            model = None

    return scaler, model


@st.cache_resource
def load_eval_data():
    test_path = os.path.join(config.DATASET_DIR, config.TEST_FILE)

    if not os.path.exists(test_path):
        return None, None, None

    test_df = pd.read_csv(test_path)

    if config.TARGET_COLUMN not in test_df.columns:
        return None, None, None

    X_test_scaled = test_df.drop(config.TARGET_COLUMN, axis=1).values.astype(np.float32)
    y_test = test_df[config.TARGET_COLUMN].values.astype(int)
    feature_cols = test_df.drop(config.TARGET_COLUMN, axis=1).columns.tolist()

    return X_test_scaled, y_test, feature_cols


@st.cache_resource
def load_models_for_comparison():
    cen_path = "models/centralized_model.keras"
    fed_path = "models/federated_model.keras"

    cen_model = tf.keras.models.load_model(cen_path) if os.path.exists(cen_path) else None
    fed_model = tf.keras.models.load_model(fed_path) if os.path.exists(fed_path) else None
    return cen_model, fed_model


def compute_metrics_from_model(model, X_test_scaled, y_test, threshold=0.5):
    if model is None:
        return None

    out = model.predict(X_test_scaled, verbose=0).reshape(-1)

    if np.min(out) < 0 or np.max(out) > 1:
        probs = 1 / (1 + np.exp(-out))
    else:
        probs = out

    preds = (probs >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (y_test == 1)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def model_comparison_page():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --bg-dark: #0B0F14;
            --accent-primary: #3ABEFF;
            --text-main: #FFFFFF;
        }
        
        html, body, p, div, h1, h2, h3, h4, h5, h6, span, label {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-main);
        }
        
        .stApp {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(58, 190, 255, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(10, 102, 194, 0.05) 0%, transparent 50%);
            background-attachment: fixed;
        }
        
        .metric-card {
            background: linear-gradient(145deg, rgba(25,30,40,0.6) 0%, rgba(15,18,25,0.8) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:32px; font-weight:800; color:#FFFFFF; margin-bottom:10px; letter-spacing: -0.5px;'>Model Performance Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:15px; color:#8E9BAE; margin-bottom:30px;'>Compare Centralized vs Federated predictive models across critical accuracy thresholds.</div>", unsafe_allow_html=True)

    X_test_scaled, y_test, feature_cols = load_eval_data()

    if X_test_scaled is None:
        st.error("Test set not found. Please run prepare_data.py first.")
        return

    cen_model, fed_model = load_models_for_comparison()

    if cen_model is None and fed_model is None:
        st.error("No models found. Expected: models/centralized_model.keras and/or models/federated_model.keras")
        return

    st.sidebar.markdown("### Comparison Settings")
    threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    cen_metrics = compute_metrics_from_model(cen_model, X_test_scaled, y_test, threshold=threshold)
    fed_metrics = compute_metrics_from_model(fed_model, X_test_scaled, y_test, threshold=threshold)

    rows = []
    if cen_metrics is not None:
        rows.append({
            "Model": "Centralized",
            "Accuracy": cen_metrics["accuracy"],
            "Precision": cen_metrics["precision"],
            "Recall": cen_metrics["recall"],
            "F1-Score": cen_metrics["f1"],
            "TP": cen_metrics["tp"],
            "TN": cen_metrics["tn"],
            "FP": cen_metrics["fp"],
            "FN": cen_metrics["fn"],
        })
    if fed_metrics is not None:
        rows.append({
            "Model": "Federated",
            "Accuracy": fed_metrics["accuracy"],
            "Precision": fed_metrics["precision"],
            "Recall": fed_metrics["recall"],
            "F1-Score": fed_metrics["f1"],
            "TP": fed_metrics["tp"],
            "TN": fed_metrics["tn"],
            "FP": fed_metrics["fp"],
            "FN": fed_metrics["fn"],
        })

    df = pd.DataFrame(rows).set_index("Model")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-bottom:15px; letter-spacing:0.5px;'>Metrics Table</div>", unsafe_allow_html=True)
        st.dataframe(df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
        }), use_container_width=True)

    with c2:
        st.markdown("<div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-bottom:15px; letter-spacing:0.5px;'>Comparison Chart</div>", unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
            fig = go.Figure()

            if cen_metrics is not None:
                fig.add_trace(go.Bar(
                    name="Centralized",
                    x=metrics_to_plot,
                    y=[df.loc["Centralized", m] for m in metrics_to_plot],
                    marker_color='#3ABEFF'
                ))
            if fed_metrics is not None:
                fig.add_trace(go.Bar(
                    name="Federated",
                    x=metrics_to_plot,
                    y=[df.loc["Federated", m] for m in metrics_to_plot],
                    marker_color='#8E9BAE'
                ))

            fig.update_layout(
                barmode="group",
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.05)', title_font=dict(color='#8E9BAE')),
                xaxis=dict(tickfont=dict(color='#8E9BAE')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#FFFFFF'), bgcolor='rgba(0,0,0,0)', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Plotly chart could not be rendered.")

    st.markdown("<div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-top:20px; margin-bottom:20px; letter-spacing:0.5px;'>Confusion Matrix Analysis</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="large")

    def render_cm_card(title, metrics, border_color):
        if metrics is None:
            return f"<div class='metric-card'>Model not found.</div>"
        return f'''
            <div class="metric-card" style="border-top: 2px solid {border_color};">
                <div style="font-size:14px; font-weight:700; color:#8E9BAE; letter-spacing:1px; margin-bottom:16px; text-transform:uppercase;">{title} Model</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                    <div><span style="color:#00E676; font-weight:800; font-size:24px;">{metrics['tp']}</span><br><span style="font-size:11px; color:#8E9BAE;">True Positive</span></div>
                    <div style="text-align:right;"><span style="color:#FF1744; font-weight:800; font-size:24px;">{metrics['fp']}</span><br><span style="font-size:11px; color:#8E9BAE;">False Positive</span></div>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <div><span style="color:#FF1744; font-weight:800; font-size:24px;">{metrics['fn']}</span><br><span style="font-size:11px; color:#8E9BAE;">False Negative</span></div>
                    <div style="text-align:right;"><span style="color:#00E676; font-weight:800; font-size:24px;">{metrics['tn']}</span><br><span style="font-size:11px; color:#8E9BAE;">True Negative</span></div>
                </div>
            </div>
        '''

    with m1:
        st.markdown(render_cm_card("Centralized", cen_metrics, "#3ABEFF"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_cm_card("Federated", fed_metrics, "#8E9BAE"), unsafe_allow_html=True)


def health_dashboard_page():
    scaler, model = load_and_prepare()

    if scaler is None:
        st.error("Scaler not found. Please run prepare_data.py first.")
        return

    if model is None:
        st.error("No trained model found. Please train and save a model first.")
        return

    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        :root {
            --bg-dark: #0B0F14;
            --bg-card: rgba(20, 25, 35, 0.6);
            --accent-primary: #3ABEFF;
            --accent-secondary: #0A66C2;
            --text-main: #FFFFFF;
            --text-muted: #8E9BAE;
        }

        html, body, p, div, h1, h2, h3, h4, h5, h6, span, label {
            font-family: 'Outfit', sans-serif;
            color: var(--text-main);
        }

        .stApp {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(58, 190, 255, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(10, 102, 194, 0.04) 0%, transparent 50%);
            background-attachment: fixed;
        }
        
        [data-testid="stMainBlockContainer"] {
            padding: 2rem 2.5rem !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        .element-container { margin-bottom: 0px !important; }
        
        div[data-testid="stNumberInput"] { margin-bottom: -15px !important; }
        .stSlider { padding-bottom: 0px !important; }
        .stSlider > div[data-testid="stWidgetLabel"] p, .stNumberInput > div[data-testid="stWidgetLabel"] p { 
            font-size: 13px !important; color: var(--text-muted) !important; font-weight: 500; letter-spacing: 0.3px;
        }

        .stButton>button {
            border-radius: 12px;
            padding: 14px 24px;
            font-size: 16px;
            font-weight: 700;
            background: var(--accent-primary);
            color: #0B0F14;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 20px;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(58, 190, 255, 0.4);
            color: #0B0F14;
        }

        .risk-pulse {
            animation: pulse-glow 2s infinite;
        }
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0.3); }
            70% { box-shadow: 0 0 0 15px rgba(255, 23, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 23, 68, 0); }
        }
        </style>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html("""
    <script>
        const doc = window.parent.document;
        function enforceLayout() {
            const cols = doc.querySelectorAll('div[data-testid="column"]');
            cols.forEach((col) => {
                col.style.background = 'linear-gradient(145deg, rgba(25,30,40,0.7) 0%, rgba(15,18,25,0.8) 100%)';
                col.style.backdropFilter = 'blur(20px)';
                col.style.webkitBackdropFilter = 'blur(20px)';
                col.style.border = '1px solid rgba(255, 255, 255, 0.05)';
                col.style.borderTop = '1px solid rgba(255, 255, 255, 0.1)';
                col.style.borderRadius = '24px';
                col.style.padding = '30px';
                col.style.height = 'calc(100vh - 64px)';
                col.style.overflow = 'hidden';
                col.style.boxShadow = '0 10px 30px rgba(0,0,0,0.5)';
                col.style.transition = 'all 0.3s ease';
            });
            const blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
            if (blocks.length > 0) blocks[0].style.gap = '24px';
        }
        enforceLayout();
        setTimeout(enforceLayout, 500);
    </script>
    """, height=0, width=0)

    col1, col2, col3 = st.columns([1.1, 1.8, 1.1], gap="medium")

    with col1:
        st.markdown("<div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-bottom:20px;'>Patient Vitals</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 0, 100, 33)
        with c2:
            gender = st.selectbox("Sex", ["Female", "Male"])
        
        c3, c4 = st.columns(2)
        with c3:
            height = st.number_input("Height (cm)", 100, 220, 170)
        with c4:
            weight = st.number_input("Weight (kg)", 30, 150, 70)
        
        bmi = round(weight / ((height / 100) ** 2), 2)
        
        glucose = st.slider("Glucose Level", 40, 250, st.session_state.get('glucose_val', 100), key='glucose_val')
        blood_pressure = st.slider("Blood Pressure", 0, 160, st.session_state.get('bp_val', 70), key='bp_val')
        
        max_insulin_allowed = min(900, int(glucose * 3.5 + 50))
        min_insulin_allowed = max(0, int(glucose * 0.1))
        ins_val = st.session_state.get('insulin_val', min_insulin_allowed + 20)
        if hasattr(ins_val, 'value'):
            ins_val = ins_val.value
        try:
            ins_val = max(min_insulin_allowed, min(int(ins_val), max_insulin_allowed))
        except:
            ins_val = min_insulin_allowed
            
        insulin = st.slider("Insulin", min_insulin_allowed, max_insulin_allowed, ins_val, key='insulin_val')
        
        max_skin_allowed = min(100, int(bmi * 1.8 + 10))
        min_skin_allowed = max(0, int(bmi * 0.3 - 5))
        skn_val = st.session_state.get('skin_val', 20)
        try:
            skn_val = max(min_skin_allowed, min(int(skn_val), max_skin_allowed))
        except:
            skn_val = min_skin_allowed
            
        skin_thickness = st.slider("Skin Thickness", min_skin_allowed, max_skin_allowed, skn_val, key='skin_val')
        dpf = st.slider("Pedigree Function", 0.0, 3.0, st.session_state.get('dpf_val', 0.5), key='dpf_val')
        pregnancies = 0 if gender == "Male" else st.slider("Pregnancies", 0, 20, st.session_state.get('preg_val', 0), key='preg_val')
        
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
        ]], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
        
        run_diag = st.button("PREDICT RISK")

    with col2:
        st.markdown("<div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-bottom:24px;'>Diagnostic Output</div>", unsafe_allow_html=True)
        pred_container = st.empty()
        insight_container = st.empty()
        
        if run_diag:
            input_scaled = scaler.transform(input_data)
            prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
            
            with pred_container.container():
                if prediction_prob >= 0.7:
                    risk_txt, color, bg, pulse = "HIGH RISK", "#FF1744", "rgba(255, 23, 68, 0.15)", "risk-pulse"
                elif prediction_prob >= 0.4:
                    risk_txt, color, bg, pulse = "MEDIUM RISK", "#FFD600", "rgba(255, 214, 0, 0.15)", ""
                else:
                    risk_txt, color, bg, pulse = "LOW RISK", "#00E676", "rgba(0, 230, 118, 0.15)", ""
                
                icon = f'<svg width="72" height="72" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
                
                st.markdown(f"""
                    <div class="{pulse}" style="background:{bg}; border:1px solid rgba(255,255,255,0.1); border-radius:20px; padding:40px; text-align:center; height:260px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                        <div style="margin-bottom:20px;">{icon}</div>
                        <div style="font-size:13px; font-weight:700; color:{color}; text-transform:uppercase; letter-spacing:2px; margin-bottom:8px;">Classification</div>
                        <div style="font-size:56px; font-weight:800; color:{color}; line-height:1;">{risk_txt}</div>
                        <div style="font-size:20px; font-weight:500; color:#FFFFFF; margin-top:12px;">{(prediction_prob*100):.1f}% AI Confidence</div>
                    </div>
                """, unsafe_allow_html=True)
                
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.history.insert(0, (ts, f"{(prediction_prob*100):.1f}%", risk_txt, color))
            if len(st.session_state.history) > 4:
                st.session_state.history.pop()
                
            with insight_container.container():
                st.markdown("<div style='font-size:16px; font-weight:700; color:#FFFFFF; margin-bottom:16px;'>Prediction Trend Analysis</div>", unsafe_allow_html=True)
                if len(st.session_state.history) > 1:
                    try:
                        import plotly.graph_objects as go
                        hist_rev = list(reversed(st.session_state.history))
                        x_vals = [h[0] for h in hist_rev]
                        y_vals = [float(h[1].strip('%')) for h in hist_rev]
                        marker_colors = [h[3] for h in hist_rev]
                        
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=x_vals, y=y_vals, mode='lines+markers',
                            line=dict(color='#5E81AC', width=2, shape='spline'),
                            marker=dict(color=marker_colors, size=12, line=dict(color='#0B0F14', width=2))
                        ))
                        fig_trend.update_layout(
                            height=180, margin=dict(l=0, r=0, t=5, b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.05)', ticksuffix='%', tickfont=dict(color='#8E9BAE', size=11)),
                            xaxis=dict(tickfont=dict(color='#8E9BAE', size=11), showgrid=False)
                        )
                        st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
                    except Exception:
                        pass
                else:
                    st.markdown("<div style='font-size:13px; color:#8E9BAE; padding-bottom:10px;'>Generate at least two predictions to view your historical trend graph.</div>", unsafe_allow_html=True)

                history_html = "".join([
                    f'<div style="font-size:13px; color:#8E9BAE; margin-bottom:12px; display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px;"><span>{h[0]}</span><span style="color:{h[3]}; font-weight:700;">{h[2]} ({h[1]})</span></div>'
                    for h in st.session_state.history
                ])
                st.markdown(f"""
                    <div style="margin-top:40px; display:flex; gap:40px;">
                        <div style="flex:1;">
                            <div style="font-size:16px; font-weight:700; color:#FFFFFF; margin-bottom:16px;">Clinical Insights</div>
                            <ul style="color:#8E9BAE; font-size:14px; line-height:1.8; padding-left:16px; margin:0;">
                                <li>{"Critical: Fasting glucose elevated." if glucose > 140 else "Stable: Glucose levels within bounds."}</li>
                                <li>{"Warning: BMI indicates risk factor." if bmi > 30 else "Stable: BMI threshold is nominal."}</li>
                                <li>{"Stable: BP normal." if blood_pressure < 80 else "Warning: Indications of hypertensive stress."}</li>
                            </ul>
                        </div>
                        <div style="flex:1;">
                            <div style="font-size:16px; font-weight:700; color:#FFFFFF; margin-bottom:16px;">Past Predictions</div>
                            {history_html}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            with pred_container.container():
                st.markdown("""
                    <div style="border:1px dashed rgba(255,255,255,0.2); background:rgba(255,255,255,0.02); border-radius:20px; padding:40px; text-align:center; height:260px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:20px;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                        <div style="font-size:15px; font-weight:500; color:#8E9BAE; text-transform:uppercase;">System Standby</div>
                        <div style="font-size:28px; font-weight:700; color:#FFFFFF; margin-top:8px;">Ready to Analyze</div>
                    </div>
                """, unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='font-size:20px; font-weight:700; color:#FFFFFF; margin-bottom:20px;'>Analysis Matrix</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius:14px; padding:18px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:11px; color:#8E9BAE; font-weight:700; letter-spacing:1px; margin-bottom:4px;">GLUCOSE LEVEL</div>
                    <div style="font-size:26px; font-weight:800; color:#FFFFFF; line-height:1;">{glucose} <span style="font-size:13px; color:#64748B; font-weight:500;">mg/dL</span></div>
                </div>
                <div style="background:{'rgba(255, 23, 68, 0.2)' if glucose > 140 else 'rgba(0, 230, 118, 0.2)'}; height:12px; width:12px; border-radius:50%; border:2px solid {'#FF1744' if glucose > 140 else '#00E676'}; box-shadow: 0 0 10px {'rgba(255,23,68,0.5)' if glucose > 140 else 'rgba(0,230,118,0.5)'};"></div>
            </div>
            
            <div style="background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius:14px; padding:18px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:11px; color:#8E9BAE; font-weight:700; letter-spacing:1px; margin-bottom:4px;">BODY MASS INDEX</div>
                    <div style="font-size:26px; font-weight:800; color:#FFFFFF; line-height:1;">{bmi} <span style="font-size:13px; color:#64748B; font-weight:500;">kg/m²</span></div>
                </div>
                <div style="background:{'rgba(255, 23, 68, 0.2)' if bmi > 30 else 'rgba(0, 230, 118, 0.2)'}; height:12px; width:12px; border-radius:50%; border:2px solid {'#FF1744' if bmi > 30 else '#00E676'}; box-shadow: 0 0 10px {'rgba(255,23,68,0.5)' if bmi > 30 else 'rgba(0,230,118,0.5)'};"></div>
            </div>
            
            <div style="background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius:14px; padding:18px; margin-bottom:32px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:11px; color:#8E9BAE; font-weight:700; letter-spacing:1px; margin-bottom:4px;">PATIENT AGE</div>
                    <div style="font-size:26px; font-weight:800; color:#FFFFFF; line-height:1;">{age} <span style="font-size:13px; color:#64748B; font-weight:500;">yrs</span></div>
                </div>
                <div style="background:rgba(58, 190, 255, 0.2); height:12px; width:12px; border-radius:50%; border:2px solid #3ABEFF; box-shadow: 0 0 10px rgba(58,190,255,0.5);"></div>
            </div>
        """, unsafe_allow_html=True)


def main():
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Health Dashboard", "Model Comparison"],
        label_visibility="collapsed"
    )

    if page == "Health Dashboard":
        health_dashboard_page()
    else:
        model_comparison_page()


if __name__ == "__main__":
    main()