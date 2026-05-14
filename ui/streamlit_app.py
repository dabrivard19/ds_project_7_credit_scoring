
import requests
import streamlit as st

#API_URL = "http://127.0.0.1:8080"
API_URL = "https://dsproject7creditscoring-production.up.railway.app"

def get_form_config():
    response = requests.get(f"{API_URL}/form-config", timeout=10)
    response.raise_for_status()
    return response.json()["features"]


import plotly.graph_objects as go

def plot_client_score_gauge(client_proba, threshold=0.5, positive_label="Classe 1"):
    score_pct = client_proba * 100
    threshold_pct = threshold * 100
    y_pred = int(client_proba >= threshold)

    decision = positive_label if y_pred == 1 else "Classe 0"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_pct,
        number={
            "suffix": "%",
            "font": {"size": 36}
        },
        title={
            "text": (
                f"<b>Score client</b><br>"
                f"<span style='font-size:16px'>Décision : {decision}</span>"
            )
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickmode": "array",
                "tickvals": [0, threshold_pct, 100],
                "ticktext": ["0%", f"Seuil<br>{threshold_pct:.1f}%", "100%"],
            },
            "bar": {"color": "darkblue"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {
                    "range": [0, threshold_pct],
                    "color": "#B7E4C7"   # zone classe 0
                },
                {
                    "range": [threshold_pct, 100],
                    "color": "#F4A6A6"   # zone classe 1
                },
            ],
            "threshold": {
                "line": {"color": "red", "width": 5},
                "thickness": 0.8,
                "value": threshold_pct,
            },
        }
    ))

    fig.add_annotation(
        text=f"Probabilité client : <b>{score_pct:.1f}%</b><br>"
             f"Seuil modèle : <b>{threshold_pct:.1f}%</b>",
        x=0.5,
        y=0.05,
        showarrow=False,
        font={"size": 14}
    )

    fig.update_layout(
        height=380,
        margin=dict(t=80, b=40, l=30, r=30),
    )

    return fig


def render_input(feature: dict):
    label = feature["label"]
    help_text = feature.get("help")
    ftype = feature["type"]

    if ftype == "int":
        return st.number_input(
            label,
            min_value=int(feature.get("min", 0)),
            max_value=int(feature.get("max", 1000000)),
            value=int(feature.get("default", 0)),
            step=1,
            help=help_text,
        )

    if ftype == "float":
        return st.number_input(
            label,
            min_value=float(feature.get("min", 0.0)),
            value=float(feature.get("default", 0.0)),
            step=float(feature.get("step", 1.0)),
            help=help_text,
        )

    if ftype == "select":
        options = feature.get("options", [])
        default = feature.get("default")
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options=options, index=index, help=help_text)

    if ftype == "bool_yn":
        checked = feature.get("default", "N") == "Y"
        return "Y" if st.checkbox(label, value=checked, help=help_text) else "N"

    return st.text_input(label, value=str(feature.get("default", "")), help=help_text)


st.set_page_config(page_title="Scoring client", layout="centered")
st.title("Test de l'API de scoring")
st.caption("Version soutenance : 10 variables seulement, avec remplissage automatique du reste.")

try:
    form_config = get_form_config()
except Exception as exc:
    st.error(f"Impossible de charger la configuration depuis l'API : {exc}")
    st.stop()

with st.form("client_form"):
    values = {}
    for feature in form_config:
        values[feature["name"]] = render_input(feature)

    submitted = st.form_submit_button("Lancer la prédiction")

if submitted:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": values},
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        prediction = result.get("prediction")
        prediction_label = "Inconnu"
        probability = 0.0
        threshold = 0.0

        if prediction == 0:
            prediction_label = "Client non risqué (Classe 0)"
            probability = 1.0 - result.get("probability")
        else:            
            prediction_label = "Client à risque (Classe 1)"
            probability = result.get("probability")
        
        threshold = result.get("threshold")

        st.success("Prédiction calculée")
        st.metric("Classe prédite", prediction)
        st.metric("Probabilité", f"{probability:.4f}")
        st.metric("Threshold", f"{threshold:.4f}")

        with st.expander("Valeurs saisies"):
            st.json(values)

        # st.metric("Result", result)
        
        fig = plot_client_score_gauge(
            client_proba=probability,
            threshold=threshold,
            positive_label=prediction_label
        )

        # fig.show()
        st.plotly_chart(fig, use_container_width=True)

        # # ========================
        # # Importance locale
        # # ========================
        # import pandas as pd
        # st.subheader("Importance locale des variables")

        # local_importance = result.get("local_importance", [])

        # if local_importance and len(local_importance) > 0:
        #     local_df = pd.DataFrame(local_importance)

        #     st.dataframe(local_df)

        #     if "feature" in local_df.columns and "contribution" in local_df.columns:
        #         st.bar_chart(
        #             local_df.set_index("feature")["contribution"]
        #         )
        #     else:
        #         st.warning("Format inattendu des données d'importance locale.")

        # else:
        #     st.info("Importance locale non disponible pour ce modèle ou cette prédiction.")

        # # ========================
        # # Importance globale
        # # ========================
        # st.subheader("Importance globale des variables")

        # global_importance = result.get("global_importance", [])

        # if global_importance and len(global_importance) > 0:
        #     global_df = pd.DataFrame(global_importance)

        #     st.dataframe(global_df)

        #     if "feature" in global_df.columns and "importance" in global_df.columns:
        #         st.bar_chart(
        #             global_df.set_index("feature")["importance"]
        #         )
        #     else:
        #         st.warning("Format inattendu des données d'importance globale.")

        # else:
        #     st.info("Importance globale non disponible pour ce modèle.")


    except Exception as exc:
        st.error(f"Erreur lors de l'appel API : {exc}")




