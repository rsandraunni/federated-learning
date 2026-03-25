'''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import config

# Set page layout
st.set_page_config(page_title="Diabetes Health Dashboard", page_icon="🧬", layout="wide")

@st.cache_resource
def load_and_prepare():
    """Load the model and refit the scaler using the training approach."""
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
            x=[x[i]]*num_points, 
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

def main():
    scaler, model = load_and_prepare()

    # --- BENTO GRID LAYOUT ---
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
        
        /* Apply Outfit font selectively, avoiding Material Icons */
        html, body, p, div, h1, h2, h3, h4, h5, h6, span, label {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Protect Streamlit icons */
        .material-icons, .material-symbols-rounded {
            font-family: 'Material Symbols Rounded' !important;
        }

        /* General page background override */
        .stApp {
            background-color: #F4F7F6;
            background-image: radial-gradient(circle at 10% 20%, rgba(0, 191, 165, 0.05) 0%, transparent 50%),
                              radial-gradient(circle at 90% 80%, rgba(41, 98, 255, 0.05) 0%, transparent 50%);
        }
        
        /* Layout Padding Adjustments - 100% Viewport Height & Width */
        [data-testid="stMainBlockContainer"] {
            padding: 2rem 3rem 1rem 3rem !important;
            max-width: 100% !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        /* Hide Streamlit Header */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Streamlit elements spacer hack reduction */
        .element-container { margin-bottom: 0px !important; }

        /* Risk Display Overrides */
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
    
    # --- JAVASCRIPT LAYOUT ENFORCER ---
    import streamlit.components.v1 as components
    components.html("""
    <script>
        // Use Javascript to reliably force 100vh Bento Grid constraints on dynamic Streamlit elements
        const doc = window.parent.document;
        
        function enforceLayout() {
            const blocks = doc.querySelectorAll('[data-testid="stHorizontalBlock"]');
            if (blocks.length >= 2) {
                // Top Row Full Height Expansion - Explicit Height to bypass DOM wrapper inheritance
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
                
                // Bottom Row Compact Metrics Anchoring
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
        
        // Run immediately and after short delay to ensure elements are rendered
        enforceLayout();
        setTimeout(enforceLayout, 500);
        setTimeout(enforceLayout, 1500);
    </script>
    """, height=0, width=0)

    # --- Sidebar Inputs ---
    st.sidebar.markdown("<h2 style='color: #111; margin-bottom: 20px;'>Patient Vitals</h2>", unsafe_allow_html=True)
    
    with st.sidebar.expander("Demographics & Vitals", expanded=True):
        age = st.slider("Age", 21, 100, 33)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
        blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
        pregnancies = st.slider("Pregnancies", 0, 20, 1)

    with st.sidebar.expander("Lab Results", expanded=True):
        glucose = st.slider("Glucose Level", 0, 250, 100)
        insulin = st.slider("Insulin", 0, 900, 79)
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
    ]], columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    # --- BENTO GRID LAYOUT ---
    
    # ROW 1: Health Overview | Tracker | Risk Engine
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
        
        # Add Blood Pressure visual block manually to pack the space effectively
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
                 if prediction_prob >= 0.5:
                     st.markdown(f"""
                        <div class="risk-alert" style="padding: 24px; background: #FDEDEC; border: 2px solid #E74C3C; border-radius: 16px; text-align:center;">
                            <div style="color: #E74C3C; font-size:13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">Diagnosis Result</div>
                            <div style="font-size: 42px; font-weight: 900; color: #E74C3C; line-height:1.1; margin: 15px 0;">HIGH RISK</div>
                            <div style="font-size: 20px; color: #C0392B; font-weight:700; margin-bottom: 20px;">{(prediction_prob*100):.1f}% Prob.</div>
                            <div style="font-size:13px; color:#922B21; font-weight:600; background:rgba(231,76,60,0.1); padding:10px; border-radius:8px;">Consult endocrinologist immediately.</div>
                        </div>
                     """, unsafe_allow_html=True)
                 else:
                     st.markdown(f"""
                        <div style="padding: 24px; background: #E8F8F5; border: 2px solid #1ABC9C; border-radius: 16px; text-align:center;">
                            <div style="color: #1ABC9C; font-size:13px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">Diagnosis Result</div>
                            <div style="font-size: 42px; font-weight: 900; color: #1ABC9C; line-height:1.1; margin: 15px 0;">LOW RISK</div>
                            <div style="font-size: 20px; color: #117864; font-weight:700; margin-bottom: 20px;">{(prediction_prob*100):.1f}% Prob.</div>
                            <div style="font-size:13px; color:#148F77; font-weight:600; background:rgba(26,188,156,0.1); padding:10px; border-radius:8px;">Vitals inside safe tolerance.</div>
                        </div>
                     """, unsafe_allow_html=True)

    # ROW 2: Compact Secondary Metrics List
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

if __name__ == "__main__":
    main()

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