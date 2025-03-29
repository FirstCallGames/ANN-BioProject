import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load and prepare the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values / 255.0  # Normalize pixel values
    Y = data.iloc[:, 0].values
    X = X.reshape(X.shape[0], 28, 28, 1)  # Reshape for CNN
    Y = to_categorical(Y, num_classes=10)  # One-hot encode labels
    return X, Y


# Build the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Output layer for 10 classes
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to train the model
def train_and_save_model(X_train, Y_train, X_test, Y_test):
    model = create_model()

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    # Train the model
    model.fit(datagen.flow(X_train, Y_train, batch_size=32), epochs=30, validation_data=(X_test, Y_test))

    # Save the model
    model.save("digit.h5")
    print("Model trained and saved as digit.h5")

    return model


# Main function
def main():
    # Load the dataset
    X, Y = load_data('train.csv')

    # Split the dataset into training and testing (80% train, 20% test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train and save the model
    train_and_save_model(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()