import tkinter as tk
from tkinter import messagebox, Toplevel
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# CIFAR-10 class names
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Build and train the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


def generate_saliency_map(model, image, class_index):
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = predictions[:, class_index]
    gradient = tape.gradient(loss, image)
    saliency_map = tf.reduce_max(tf.abs(gradient), axis=-1)[0]
    return saliency_map


def grad_cam(model, image, class_index, layer_name='conv2d'):
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    plt.show()


def show_results(image_index):
    image = test_images[image_index]
    predictions = model.predict(np.expand_dims(image, axis=0))
    class_index = np.argmax(predictions)
    predicted_class = class_names[class_index]

    saliency_map = generate_saliency_map(model, image, class_index)
    heatmap = grad_cam(model, image, class_index)

    heatmap = cv2.resize(heatmap, (32, 32))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + image * 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(saliency_map, cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')

    axes[2].imshow(superimposed_img / 255.0)
    axes[2].set_title("Grad-CAM")
    axes[2].axis('off')

    plt.show()

    # Show description in a new window
    description_window = Toplevel(root)
    description_window.title("Περιγραφή Εικόνας")
    description_label = tk.Label(
        description_window,
        text=f"Προβλεπόμενη Κατηγορία: {predicted_class}\n"
             f"Εμπιστοσύνη: {predictions[0][class_index]:.2f}\n"
             f"Περιγραφή: Αυτή η εικόνα πιθανότατα ανήκει στην κατηγορία '{predicted_class}'.",
        justify='left',
        wraplength=400,
        fg='blue'
    )
    description_label.pack(pady=10)

def show_initial_popup():
    popup = Toplevel(root)
    popup.title("Καλώς ήρθατε!")

    # Explanation text
    explanation_label = tk.Label(
        popup,
        text=(
            "Καλώς ήρθατε στην εφαρμογή οπτικοποίησης CNN μοντέλου!\n\n"
            "Με αυτήν την εφαρμογή μπορείτε να:\n"
            "1. Εξετάσετε πώς ένα βαθύ νευρωνικό δίκτυο (CNN) αναλύει εικόνες από το σύνολο δεδομένων CIFAR-10.\n"
            "2. Επιλέξετε μια εικόνα για ανάλυση από ένα ευρύ φάσμα κατηγοριών, όπως:\n"
            "   - Αεροπλάνα\n"
            "   - Αυτοκίνητα\n"
            "   - Πουλιά\n"
            "   - Γάτες\n"
            "   - Ελάφια και άλλα.\n"
            "3. Δείτε πώς το μοντέλο προβλέπει την κατηγορία της εικόνας χρησιμοποιώντας:\n"
            "   - Saliency Maps (αναδεικνύουν τις περιοχές που επηρεάζουν την πρόβλεψη).\n"
            "   - Grad-CAM (οπτικοποίηση χαρακτηριστικών που ανιχνεύει το CNN).\n\n"
            "Επιπλέον, μπορείτε να:\n"
            "- Παρακολουθήσετε την ακρίβεια και την απώλεια εκπαίδευσης του μοντέλου.\n"
            "- Αναλύσετε ποιοτικά την απόδοση του μοντέλου για συγκεκριμένες εικόνες.\n\n"
            "Παρακαλώ, επιλέξτε αν θέλετε να συνεχίσετε.\n"
            "Επιλέγοντας 'Ναι', θα μεταφερθείτε στην κύρια εφαρμογή.\n"
            "Επιλέγοντας 'Όχι', η εφαρμογή θα κλείσει.\n\n"
            "Καλή εξερεύνηση!"
        ),
        justify="left",
        wraplength=400,
        font = ("Arial", 12)
    )
    explanation_label.pack(pady=20, padx=20)

    # Yes button to continue
    def proceed():
        popup.destroy()

    yes_button = tk.Button(popup, text="Ναι", command=proceed, bg="green", fg="white")
    yes_button.pack(side=tk.LEFT, padx=20, pady=20)

    # No button to exit the application
    def exit_app():
        root.destroy()

    no_button = tk.Button(popup, text="Όχι", command=exit_app, bg="red", fg="white")
    no_button.pack(side=tk.RIGHT, padx=20, pady=20)

    # Prevent interaction with the main window until a choice is made
    popup.transient(root)
    popup.grab_set()
    root.wait_window(popup)

# Create the tkinter UI
root = tk.Tk()
root.title("Saliency Map & Grad-CAM Visualizer")

# Call the initial pop-up window
show_initial_popup()

# Add information label
info_label = tk.Label(
    root,
    text=("Αυτό το εργαλείο οπτικοποιεί πώς ένα CNN ερμηνεύει εικόνες:\n"
          "- 'Αρχική Εικόνα': Η εικόνα που εισάγεται στο μοντέλο.\n"
          "- 'Saliency Map': Αναδεικνύει τις περιοχές της εικόνας που επηρεάζουν την πρόβλεψη.\n"
          "- 'Grad-CAM': Δείχνει τα χαρακτηριστικά που ανιχνεύει το CNN για την προβλεπόμενη κατηγορία.\n"
          "Training accuracy και loss διαγράμματα δείχνουν την απόδοση του μοντέλου."),
    justify='left',
    wraplength=400
)
info_label.pack(pady=10)

# Add instruction label
instruction_label = tk.Label(root, text="Παρακαλώ επιλέξτε μια εικόνα για ανάλυση:", font=("Arial", 12))
instruction_label.pack(pady=10)

# Create a canvas for image selection
canvas = tk.Canvas(root, width=800, height=400)
canvas.pack(pady=10)

# Variables for paging
current_page = 0
images_per_page = 40  # Number of images to display per page
image_size = 80  # Size of each thumbnail
grid_width = 10  # Number of images per row

def display_images():
    canvas.delete("all")  # Clear the canvas
    start_index = current_page * images_per_page
    end_index = min(start_index + images_per_page, len(test_images))
    thumbnails.clear()

    for i in range(start_index, end_index):
        row, col = divmod(i - start_index, grid_width)
        x, y = col * image_size, row * image_size
        image = (test_images[i] * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = tk.PhotoImage(data=cv2.imencode('.png', cv2.resize(image, (image_size, image_size)))[1].tobytes())
        canvas.create_image(x, y, anchor=tk.NW, image=img)
        thumbnails.append((i, img))  # Store image index and PhotoImage

# Click event to select an image
def on_canvas_click(event):
    col, row = event.x // image_size, event.y // image_size
    image_index = row * grid_width + col + current_page * images_per_page
    if 0 <= image_index < len(test_images):
        show_results(image_index)

canvas.bind("<Button-1>", on_canvas_click)

# Navigation buttons
def previous_page():
    global current_page
    if current_page > 0:
        current_page -= 1
        display_images()

def next_page():
    global current_page
    if (current_page + 1) * images_per_page < len(test_images):
        current_page += 1
        display_images()

prev_button = tk.Button(root, text="⟵ Προηγούμενη", command=previous_page)
prev_button.pack(side=tk.LEFT, padx=20, pady=10)

next_button = tk.Button(root, text="Επόμενη ⟶", command=next_page)
next_button.pack(side=tk.RIGHT, padx=20, pady=10)

# Add a button to display training history
training_button = tk.Button(root, text="Show Training History", command=lambda: plot_training_history(history))
training_button.pack(pady=10)

# Display the first set of images
thumbnails = []
display_images()

root.mainloop()