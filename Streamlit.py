import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from sklearn.ensemble import VotingClassifier
from PIL import Image, ImageOps

st.title("Sifferigenkänning med maskininlärningsmodeller")
st.write("Skriv en siffra nedan:                                                                                Din siffra översatt till en plot:")

# Laddar in modellerna
xgb_model = joblib.load("ml_mnist_xgb.pkl")
xgb_model_10 = joblib.load("ml_mnist_10_xgb.pkl")
# rf_model = joblib.load("ml_mnist_rf.pkl") # Slöar ner appen fruktansvärt så avaktiverad
# vote_model = joblib.load("ml_mnist_vote.pk1") # Slöar ner appen fruktansvärt så avaktiverad

# Skapar två kolumner för att kunna ha dem sida vid sida
col1, col2 = st.columns(2)

# Kolumn 1
with col1:
# Skapar en rityta / canvas
    canvas_result = st_canvas(
        fill_color="white",  
        stroke_width=30, 
        stroke_color="black",   
        background_color="white",
        height=280, width=280,  
        drawing_mode="freedraw", 
        key="canvas",
    )

    # Börjar bearbetningen av siffran/bilden
if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype("uint8"))
    image = ImageOps.grayscale(image)  # Gör bilden svartvit
    image = image.resize((28, 28))  # Ändra storlek till 28x28
    inverted_image = ImageOps.invert(image) # Inverterar färgerna 

    # Konvertera bilden till en 1D-array (flat)
    img_array = np.array(inverted_image).astype("uint8") # Gör om till en array
    img_array_10 = np.where(img_array > 10, 255, 0) # Ritar om punkterna för XGB_model_10
    # img_array = 255 - img_array # Inverterar färgerna på ett annat sätt
    img_flat = img_array.flatten().reshape(1, -1) # Plattar till data
    img_flat_10 = img_array_10.flatten().reshape(1, -1) # Plattar till omritad data
    
    # Kolumn 2
    with col2:
        # Ritar upp en plot med värden översatt från arrayen, en översättning från bild till data
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_array, cmap=plt.get_cmap("binary"))  # Använd "binary" för att efterlikna MNIST
        st.pyplot(fig)
    
    if xgb_model:
        
        # Gör en prediktion med XGBoost
        prediction = xgb_model.predict(img_flat)[0]
        probabilities = xgb_model.predict_proba(img_flat)[0]
        st.subheader(f"XGBoost tolkar ditt kladd som en: {prediction}")
        
        # Gör en prediktion med XGBoost där alla pixlar med värde över 10 ändras till 255, som jämförelse
        prediction_10 = xgb_model_10.predict(img_flat_10)[0]
        # probabilities_10 = xgb_model_10.predict_proba(img_flat_10)[0]
        st.subheader(f"XGBoost med omritning tolkar ditt kladd som en: {prediction_10}")
        
        # Gör en prediktion med Random forest, som jämförelse. #### Slöar ner appen fruktansvärt så avaktiverad
        # prediction_rf = rf_model.predict(img_flat)[0]
        # probabilities_rf = rf_model.predict_proba(img_flat)[0] 
        # st.subheader(f"Random forest tolkar ditt kladd som en: {prediction_rf}") 
        
        # Gör en prediktion med vote, som jämförelse. #### Slöar ner appen fruktansvärt så avaktiverad
        # prediction_vote = vote_model.predict(img_flat)[0]
        # probabilities_vote = vote_model.predict_proba(img_flat)[0]
        # st.subheader(f"Vote tolkar ditt kladd som en: {prediction_vote}")
        
        # Plottar sannolikhetsfördelningsgraf
        fig, ax = plt.subplots()
        ax.bar(range(10), probabilities, color="blue")
        ax.set_xticks(range(10))
        ax.set_xlabel("Siffra")
        ax.set_ylabel("Sannolikhet")
        ax.set_title("Sannolikhetsfördelning för XGBoost")
        st.pyplot(fig)
        
    else:
        st.error("Rita för att kunna göra en prediktion.")


# Kör i Terminalen: streamlit run streamlit.py