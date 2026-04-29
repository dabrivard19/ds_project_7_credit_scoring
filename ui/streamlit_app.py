
import requests
import streamlit as st

#API_URL = "http://127.0.0.1:8080"
API_URL = "https://dsproject7creditscoring-production.up.railway.app"

def get_form_config():
    response = requests.get(f"{API_URL}/form-config", timeout=10)
    response.raise_for_status()
    return response.json()["features"]


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

        st.success("Prédiction calculée")
        st.metric("Classe prédite", result["prediction"])
        if result.get("probability") is not None:
            st.metric("Probabilité", f"{result['probability']:.4f}")

        with st.expander("Valeurs saisies"):
            st.json(values)

        # ========================
        # Importance locale
        # ========================
        import pandas as pd
        st.subheader("Importance locale des variables")

        local_importance = result.get("local_importance", [])

        if local_importance and len(local_importance) > 0:
            local_df = pd.DataFrame(local_importance)

            st.dataframe(local_df)

            if "feature" in local_df.columns and "contribution" in local_df.columns:
                st.bar_chart(
                    local_df.set_index("feature")["contribution"]
                )
            else:
                st.warning("Format inattendu des données d'importance locale.")

        else:
            st.info("Importance locale non disponible pour ce modèle ou cette prédiction.")

        # ========================
        # Importance globale
        # ========================
        st.subheader("Importance globale des variables")

        global_importance = result.get("global_importance", [])

        if global_importance and len(global_importance) > 0:
            global_df = pd.DataFrame(global_importance)

            st.dataframe(global_df)

            if "feature" in global_df.columns and "importance" in global_df.columns:
                st.bar_chart(
                    global_df.set_index("feature")["importance"]
                )
            else:
                st.warning("Format inattendu des données d'importance globale.")

        else:
            st.info("Importance globale non disponible pour ce modèle.")


    except Exception as exc:
        st.error(f"Erreur lors de l'appel API : {exc}")
