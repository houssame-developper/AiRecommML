import streamlit as st
import pandas as pd
from joblib import load

from tools.agent_api import agent_call
# Initialize session state ONCE
st.session_state.setdefault("submitted", False)


model = load("tools/recomm_model.joblib")

decode_pred = load("tools/LabelEncoder.joblib") 

# Helper function
def note_input(label):
    return st.number_input(
        label,
        min_value=0.0,
        max_value=20.0,
        step=0.25
    )

# BEFORE submit
if not st.session_state["submitted"]:
    st.header("Student Form")

    notes = {
    # Math & CS
    "note_algebre": note_input("Algebra"),
    "note_analyse": note_input("Analysis"),
    "note_probabilites_stats": note_input("Probabilities & Statistics"),
    "note_recherche_operationnelle": note_input("Operational Research"),
    "note_analyse_donnees": note_input("Data Analysis"),
    "note_algorithmique": note_input("Algorithms"),
    "note_programmation_c": note_input("C Programming"),
    "note_programmation_web": note_input("Web Programming"),
    "note_programmation_objet_cpp_java": note_input("OOP (C++ / Java)"),
    "note_structures_donnees": note_input("Data Structures"),
    "note_bases_de_donnees": note_input("Databases"),
    "note_systeme_exploitation": note_input("Operating Systems"),
    "note_architecture_ordinateurs": note_input("Computer Architecture"),
    "note_reseaux": note_input("Networks"),
    "note_genie_logiciel_agile": note_input("Software Engineering (Agile)"),
    "note_compilation": note_input("Compilation"),
    "note_microservices_jee": note_input("Microservices / JEE"),

    # Physics
    "note_mecanique_point_solide": note_input("Mechanics (Point & Solid)"),
    "note_thermodynamique": note_input("Thermodynamics"),
    "note_optique_geom_ondul": note_input("Optics (Geometric & Wave)"),
    "note_electricite_electromag": note_input("Electricity & Electromagnetism"),
    "note_mecanique_quantique": note_input("Quantum Mechanics"),
    "note_physique_nucleaire_stat": note_input("Nuclear & Statistical Physics"),
    "note_instrumentation_mesure": note_input("Instrumentation & Measurement"),
    "note_modelisation_simulation": note_input("Modeling & Simulation"),
    "note_python_physique": note_input("Python for Physics"),
    "note_electronique_num_analog": note_input("Electronics (Analog & Digital)"),

    # Biology
    "note_biologie_cellulaire_histologie": note_input("Cell Biology & Histology"),
    "note_bio_organismes_v_a": note_input("Organisms & Living Systems"),
    "note_ecologie_microbiologie": note_input("Ecology & Microbiology"),
    "note_biochimie_genetique": note_input("Biochemistry & Genetics"),
    "note_physiologie": note_input("Physiology"),

    # Chemistry & Geology
    "note_atomistique_liaison_chimique": note_input("Atomic Structure & Chemical Bonding"),
    "note_chimie_organique_solutions": note_input("Organic Chemistry & Solutions"),
    "note_cristallochimie": note_input("Crystallochemistry"),
    "note_geologie_generale": note_input("General Geology"),
    "note_geodynamique_int_ext": note_input("Internal & External Geodynamics"),
    "note_tectonique_petrologie": note_input("Tectonics & Petrology"),
    "note_sedimentologie_geochimie": note_input("Sedimentology & Geochemistry"),

    # Major
    "major": st.selectbox(
        "Select Major",
        ["Math-Info", "Physique", "BCG"]
    )
}


    if st.button("Submit"):
        st.session_state["submitted"] = True
        st.session_state["notes"] = notes
        st.rerun()

# AFTER submit
else:
    st.success("Form submitted successfully")

    df = pd.DataFrame([st.session_state["notes"]])
    st.dataframe(df)
    recomm = model.predict(df)
    recomm = decode_pred.inverse_transform(recomm)


    with st.container():
        st.subheader("Result")
        st.write("Recommandation:", recomm)
        placeholder = st.empty()  # مكان لتحديث النص تدريجيًا
        recomm_text = ""

        # استدعاء agent وتحديث Markdown أثناء التدفق
        for content in agent_call(data=st.session_state["notes"],recomm=recomm):
            recomm_text += content
            placeholder.markdown(
            f"""
            <div style="
                background-color: #0B0D0F;  /* رمادي فاتح */
                color: #EAEAEA;               /* نص داكن وواضح */
                padding: 15px;                /* مسافة داخلية */
                border-radius: 10px;          /* حواف مستديرة */
                border: 1px solid #b3b3b3;   /* حواف خارجية */
                margin-bottom: 15px;          /* مسافة خارجية بين الرسائل */
            ">
                {recomm_text}
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("Edit"):
        st.session_state["submitted"] = False
        st.rerun()
