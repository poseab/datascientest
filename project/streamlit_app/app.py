from collections import OrderedDict

import streamlit as st

# TODO : change TITLE, TEAM_MEMBERS and PROMOTION values in config.py.
import config

# TODO : you can (and should) rename and add tabs in the ./tabs folder, and import them here.
from tabs import A_intro, B_database, C_dataviz, D_preprocessing, E_machine_learning, F_bert, H_computer_vision,K_extraction, L_conclusion


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


# TODO: add new and/or renamed tab in this ordered dict by
# passing the name in the sidebar as key and the imported tab
# as value as follow :
TABS = OrderedDict(
    [
        (A_intro.sidebar_name, A_intro),
        (B_database.sidebar_name, B_database),
        (C_dataviz.sidebar_name, C_dataviz),
        (D_preprocessing.sidebar_name, D_preprocessing),
        (E_machine_learning.sidebar_name, E_machine_learning),
        (F_bert.sidebar_name, F_bert),
        (H_computer_vision.sidebar_name, H_computer_vision),
        (K_extraction.sidebar_name, K_extraction),
        (L_conclusion.sidebar_name, L_conclusion)
    ]
)


def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()
