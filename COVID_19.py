import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model("D:\Streamlit\Covid19_Prediction_Chest_XRay\Covid_19_Model.h5")  # Replace with your model path

# Define the labels for the prediction classes
labels = ['Non-COVID','COVID-19']

# Define the function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Create the Streamlit web app
def main():
    st.title("COVID-19 Prediction")
    st.text("Upload an X-ray image of Chest and predict the chance of COVID-19")

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make predictions
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Display the prediction result
        st.write("Prediction:", labels[predicted_class])
        st.write("Confidence:", round(confidence * 100, 2), "%")

# Run the app
if __name__ == "__main__":
    main()
