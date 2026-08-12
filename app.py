# import streamlit as st
# import requests
# from PIL import Image

# # Configuration de l'API
# API_URL = "https://api-inference.huggingface.co/models/Devarshi/Brain_Tumor_Classification"
# API_TOKEN = "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"  # Remplacez par votre vrai token API
# headers = {"Authorization": f"Bearer {API_TOKEN}"}

# def query(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()

# # Titre de l'application
# st.markdown(
#     """
#     <h3 style="text-align: center; color: #007BFF;">
#         🧠 Classification des tumeurs cérébrales 🧠
#     </h3>
#     <style>
#     .main {
#         background-color: #f0f8ff;
#     }
#     .stSidebar {
#         background-color: #f9f9f9;
#     }
#     .history-entry {
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 5px;
#     }
#     .no_tumor {
#         background-color: #d4edda; /* Vert pâle */
#     }
#     .meningioma_tumor {
#         background-color: #fff3cd; /* Jaune pâle */
#     }
#     .glioma_tumor {
#         background-color: #cce5ff; /* Bleu pâle */
#     }
#     .pituitary_tumor {
#         background-color: #f8d7da; /* Rouge pâle */
#     }
#     .classification-entry {
#         padding: 10px;
#         border-radius: 5px;
#         margin-bottom: 5px;
#         color: #fff;  /* Couleur du texte blanc pour un meilleur contraste */
#     }
#     .no_tumor_result {
#         background-color: #28a745; /* Vert */
#     }
#     .meningioma_tumor_result {
#         background-color: #ffc107; /* Jaune */
#     }
#     .glioma_tumor_result {
#         background-color: #007bff; /* Bleu */
#     }
#     .pituitary_tumor_result {
#         background-color: #dc3545; /* Rouge */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Ajout d'une colonne latérale pour l'historique de recherche
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Téléchargement de l'image
# uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# if uploaded_file is not None:
#     # Enregistrer le fichier téléchargé temporairement
#     filename = "temp_image" + uploaded_file.name
#     with open(filename, "wb") as temp_file:
#         temp_file.write(uploaded_file.getbuffer())

#     # Affichage de l'image téléchargée avec une largeur fixe
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Image téléchargée', width=400, output_format="auto")  # Ajustez la largeur ici

#     # Classification
#     with st.spinner("Classification en cours..."):
#         output = query(filename)

#     # Affichage des résultats
#     st.subheader("Résultats de la classification :")
#     results = []

#     # Vérifiez si output est une liste et contient des résultats
#     if isinstance(output, list) and len(output) > 0:
#         results.clear()  # Clear the results list
        
#         # Vider l'historique pour actualiser les résultats
#         st.session_state.history.clear()  

#         for result in output:
#             label = result.get('label')
#             score = result.get('score')
#             results.append(f"- **{label}** : {score:.2%}")

#             # Ajouter chaque résultat à l'historique
#             st.session_state.history.append({
#                 "label": label,
#                 "score": score
#             })

#             # Afficher le résultat avec une couleur de fond appropriée
#             label_class = label.replace(" ", "_").lower() + "_result"  # Créer une classe CSS
#             st.markdown(f"<div class='classification-entry {label_class}'>- **{label}** : {score:.2%}</div>", unsafe_allow_html=True)

#     else:
#         # Afficher un message d'erreur si output n'est pas une liste ou vide
#         st.write("Erreur dans la réponse de l'API : ", output.get('error', 'Aucun détail d\'erreur disponible.'))

# # Ajout d'un bouton pour supprimer l'historique
# if st.sidebar.button("Supprimer l'historique"):
#     st.session_state.history = []  # Vider l'historique
#     st.sidebar.write("Historique supprimé.")

# # Affichage de l'historique des recherches avec des couleurs
# if st.session_state.history:
#     st.sidebar.subheader("Historique des résultats")
#     for item in st.session_state.history:
#         label_class = item['label'].replace(" ", "_").lower()  # Convertir le label en une classe CSS
#         st.sidebar.markdown(f"<div class='history-entry {label_class}'>- **Label** : {item['label']} - **Score** : {item['score']:.2%}</div>", unsafe_allow_html=True)



import streamlit as st
from PIL import Image
from transformers import pipeline

# -------------------------------
# Configuration de la page
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>

.main {
    background-color:#f8f9fa;
}

.result{
    padding:15px;
    border-radius:10px;
    margin-bottom:10px;
    color:white;
    font-weight:bold;
    font-size:18px;
}

.glioma{
    background:#007bff;
}

.meningioma{
    background:#ffc107;
    color:black;
}

.pituitary{
    background:#dc3545;
}

.notumor{
    background:#28a745;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Charger le modèle UNE seule fois
# -------------------------------
@st.cache_resource
def load_model():

    classifier = pipeline(
        task="image-classification",
        model="Devarshi/Brain_Tumor_Classification"
    )

    return classifier


classifier = load_model()

# -------------------------------
# Historique
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Titre
# -------------------------------
st.title("🧠 Brain Tumor Classification")

st.write(
    "Téléchargez une image IRM du cerveau afin de classifier le type de tumeur."
)

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg","jpeg","png"]
)

# -------------------------------
# Classification
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,width=400)

    with st.spinner("Classification en cours..."):

        results = classifier(image)

    st.subheader("Résultats")

    st.session_state.history.clear()

    for result in results:

        label = result["label"]
        score = result["score"]

        css = label.lower().replace(" ","")

        st.markdown(
            f"""
            <div class="result {css}">
            {label} : {score:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state.history.append(result)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Historique")

if st.sidebar.button("Supprimer"):

    st.session_state.history=[]

if len(st.session_state.history)>0:

    for item in st.session_state.history:

        st.sidebar.write(
            f"**{item['label']}** : {item['score']:.2%}"
        )
