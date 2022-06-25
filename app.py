import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization, Input
from PIL import Image


def predict(img):
    img_size = (64, 64)
    img = img.resize(img_size)
    img = Image.fromarray(np.uint8(img))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis = 0)[:, :, :, :3]
    pred = model.predict(img)
    return pred

def app():
    st.title('Malaria Detection (Project by Toufiq Rahatwilkar)')
    st.markdown('This app detects if the patient has malaria or not')
    st.markdown('The app is based on Resnet50 model pre-trained on ImageNet dataset.')
    st.markdown("#")


    uploaded_image = st.file_uploader('Upload an image to predict')
    

    if uploaded_image:
        st.image(uploaded_image)
        img = Image.open(uploaded_image)
        pred_button = st.button("Predict")
        if pred_button:
            prediction = predict(img)

            if prediction > 0.5:
                st.subheader("The patient is not suffering from malaria")
                st.balloons()
            else:
                st.subheader("The patient is suffering from malaria")

def build_model():
    base_model = tf.keras.applications.resnet50.ResNet50(weights=None,include_top=False)
    model = Sequential()
    model.add(Input(shape=(64,64,3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_model()
    model.load_weights("resnet_malaria.h5")
    app()