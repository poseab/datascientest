import streamlit as st
import pandas as pd
from .helpers import ocr_predict, read_image, ocr_get_text, get_cv2_image_from_upload, translate_fr_to_en
from doctr.utils.visualization import visualize_page
import _pickle as cPickle

title = "Machine learning"
sidebar_name = "Machine learning"


def predictImage(image):
    with st.spinner("Ouverture..."):
        cv2image = get_cv2_image_from_upload(image)
        st.markdown("### Traitement de l'image")
        st.image(cv2image)
    with st.spinner("Extraction OCR en cours..."):
        document = ocr_predict(image, bin)
        text = ocr_get_text(document)
        fig = visualize_page(document.pages[0].export(), cv2image, interactive=False,add_labels=False)
        st.markdown("### Extraction OCR")
        st.pyplot(fig)
        st.markdown(f"Text : {text}")
    #st.markdown("### Detection langue")
    #code_lang = get_language_code_fasttext(text)
    #st.markdown(f"Langue détécté : {code_lang}")
    with st.spinner("Traduction en cours.."):
        text = translate_fr_to_en(text,False)
        st.markdown("### Traduction")
        st.markdown(f"Text traduit en anglais: {text}")
    
    with st.spinner("Prédiction en cours.."):

        # Vectorizer
        with open('../model/RF/vectorizer_RF', 'rb') as file_vec:
            vectorizer = cPickle.load(file_vec)
        t = vectorizer.transform([text])

        # Tokenizer
        with open('../model/RF/tokenizer_RF', 'rb') as file_tok:
            tokenizer = cPickle.load(file_tok)
        t = tokenizer.transform(t)

        # Model Rain Forest
        with open('../model/RF/model_RF', 'rb') as file_model:
            rf = cPickle.load(file_model)
        
        y_pred = rf.predict(t)
        pred =y_pred
        #pred = "toto"
        st.markdown("### Prédiction")
        st.markdown(f"Classifie en tant que **{pred}** ")
    
    return (document,text,pred)


def run():
    st.title(title)

    df = pd.read_csv('../data/data_with_meta.csv')
    st.markdown("""
                Une des principales disciplines de l'IA est le Machine Learning, aussi nommé l'apprentissage automatique.
                Il s'agit d'exploiter des données brutes, de les transformer en connaissances \n
                 et ce, de manière automatique afin de prendre de meilleures décisions d'affaires.
    """)
    with  st.expander('Modèles testés'):
        st.markdown("""
            * Random Forest\n
            * Régression Logique\n
            * SVM\n
            * Gradient Boosting
        """)
    st.markdown("""Pour chacun de ces modèles,comme certaines classes sont sous-représentées, nous avons rajouté
                   une fonction Random Over Sampler pour générer de nouveaux échantillons aléatoires.
                """)

    st.markdown("""
        Cette technique permet de rééquilibrer la distribution des classes pour un ensemble de données
        déséquilibré.\n
        Voici un tableau contenant les précisions globales obtenues pour les quatres modèles avant et après
        l'ajout de la fonction Random Over Sampler :
        """)

    st.markdown("### Résultats obtenus")

    st.image(
        "./assets/score_all.jpg"
    )

    with st.expander('Modèles retenu'):
        st.markdown("""
        * Random Forest
    """)

    st.markdown("---")
    st.markdown("### Résultats obtenus pour Random Forest")

    st.image(
        "./assets/random_f_score.jpg"
    )
    
    st.markdown("""
        On remarque une très bonne précision sur les passeports (1) 
        et pièces d'identité (0.95) .avec un rappel de 0.62 sur les passeports et 0.91
        sur les pièces d'identités.
    """)

    st.markdown("### Tester le model")

    uploaded_file = st.file_uploader("Choisir une image")
    if uploaded_file is not None:
        # To read file as bytes:
        image = uploaded_file.read()
        (document,text,predict) = predictImage(image)
