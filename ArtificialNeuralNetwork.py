import cv2
import numpy as np
from keras.models import load_model
from tkinter import *

# Load the trained model (make sure the model is saved as 'digit_recognition_model.h5')
model = load_model('digit.h5')

# Initialize a blank image (black canvas)
canvas_width, canvas_height = 280, 280  # Larger canvas for drawing (10x MNIST size)
image = np.zeros((canvas_height, canvas_width), dtype=np.uint8)


# Function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="white", width=10)
    cv2.circle(image, (event.x, event.y), 10, (255), thickness=-1)


# Function to clear the canvas
def clear_canvas():
    global image
    canvas.delete("all")
    image = np.zeros((canvas_height, canvas_width), dtype=np.uint8)


# Function to make predictions based on the drawn image
def predict_digit():
    # Resize the image to 28x28 pixels (same size as the model input)
    resized_image = cv2.resize(image, (28, 28))

    # Normalize the pixel values
    normalized_image = resized_image / 255.0

    # Reshape the image to fit the model input: (1, 28, 28, 1)
    input_image = normalized_image.reshape(1, 28, 28, 1)

    # Make predictions on the input image
    prediction = model.predict(input_image)
    predicted_digit = np.argmax(prediction)

    # Display the prediction in the GUI
    label_prediction.config(text=f'Prediction: {predicted_digit}')


# Setting up the GUI using tkinter
root = Tk()
root.title("Draw a Digit")

# Create a canvas for drawing
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='black')
canvas.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Bind the mouse events to the canvas for drawing
canvas.bind("<B1-Motion>", paint)

# Button to clear the canvas
clear_button = Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.grid(row=1, column=0, pady=2)

# Button to predict the drawn digit
predict_button = Button(root, text="Predict Digit", command=predict_digit)
predict_button.grid(row=1, column=1, pady=2)

# Label to display the prediction
label_prediction = Label(root, text="Prediction: None", font=("Helvetica", 16))
label_prediction.grid(row=2, column=0, columnspan=2)

# Start the tkinter loop
root.mainloop()

#97.31%