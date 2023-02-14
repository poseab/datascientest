import streamlit as st

title = "BILAN & SUITE DU PROJET"
sidebar_name = "Conclusion"

def run():

    st.title(title)

    st.markdown("""A travers le projet, nous avons étudié de nombreux algorithmes (en traitement du langage et en computer
        vision) pour parvenir à répondre à notre problématique, à savoir classifier au mieux les pièces d'identité
        (cartes d'identité et passeports).
    """)
    st.markdown("""
        Compte tenu de notre jeu de données, l'algorithme ResNet50 semble le mieux répondre à cette
        problématique avec des précisions et rappels de l'ordre de 93%.\n
        Cependant, il faut noter que le jeu de données utilisé pour le projet n'était pas de qualité suffisante pour
        apporter une réponse complète au contexte énoncé en début de rapport.\n
        En effet, pour une étude plus pertinente, il faudrait un jeu de données avec:
    """)
    st.markdown("""
        * Plus de documents pour entraîner les modèles,\n
        * Une plus grande diversité dans les pièces d'identité,\n
        * Un scan de la page avec nom, prénom, photo pour les pièces d'identité,\n
        * Etc…\n
    """)
    st.markdown("""
        Un plus grand nombre de documents et des documents plus diversifiés permettrait de gagner en
        robustesse. Un meilleur cadrage des pièces d'identité avec le scan de la “bonne” page permettrait une
        meilleure océrisation et ainsi améliorerait considérablement les performances des algorithmes en
        traitement du langage. Des algorithmes tels que Logistic Regression ou BERT deviendraient alors de bons
        candidats pour répondre à la problématique.\n
        Ainsi, l'étape suivante serait de reprendre les algorithmes détaillés dans notre rapport dans un contexte
        entreprise avec des jeux de données de meilleure qualité.""")