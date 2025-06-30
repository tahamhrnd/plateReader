import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label
from PIL import Image, ImageTk
import torch
import cv2
from torchvision import transforms
from hezar.models import Model

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
])

# Load the model
model = Model.load("hezarai/crnn-fa-license-plate-recognition")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CTCLoss(blank=0)  # مقدار blank تنظیم شده است.

def predict_and_correct_ui(image_path, correct_label):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 64))
        image = Image.fromarray(image)
        input_tensor = transform(image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            prediction = model.predict(image_path)

        if correct_label:
            update_model(model, input_tensor, correct_label)
        else:
            messagebox.showinfo("Info", "Prediction was correct!")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_model(model, input_tensor, correct_label):
    optimizer.zero_grad()

    # Convert correct_label to tensor
    target = torch.tensor([ord(c) - ord('0') for c in correct_label], dtype=torch.long).unsqueeze(0)
    target_lengths = torch.tensor([len(correct_label)], dtype=torch.long)

    # Model prediction
    output = model(input_tensor)

    # Extract log probabilities
    if isinstance(output, dict) and 'logits' in output:
        log_probs = output['logits']
    else:
        raise ValueError("Output of model does not contain 'logits' key.")

    # Calculate Loss
    T = log_probs.size(0)  # Time steps
    input_lengths = torch.tensor([T], dtype=torch.long)
    loss = criterion(log_probs, target, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()

    messagebox.showinfo("Info", "Model updated with new data.")

def check_prediction():
    image_path = file_path_var.get()

    if not image_path:
        messagebox.showwarning("Warning", "Please upload an image.")
        return

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 64))
        input_tensor = transform(Image.fromarray(image)).unsqueeze(0)

        with torch.no_grad():
            prediction = model.predict(image_path)
        predicted_plate.set(str(prediction))

    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {str(e)}")

# GUI Setup
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        file_path_var.set(file_path)
        display_image(file_path)

def display_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((256, 128))  # Resize for display
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display image: {str(e)}")

def submit():
    image_path = file_path_var.get()
    correct_label = label_var.get()

    if not image_path:
        messagebox.showwarning("Warning", "Please upload an image.")
        return

    predict_and_correct_ui(image_path, correct_label)

# Main UI
root = tk.Tk()
root.title("License Plate Recognition")

file_path_var = tk.StringVar()
label_var = tk.StringVar()
predicted_plate = tk.StringVar()

# File upload
tk.Label(root, text="Upload Image:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
file_entry = tk.Entry(root, textvariable=file_path_var, width=40)
file_entry.grid(row=0, column=1, padx=10, pady=5)

browse_btn = tk.Button(root, text="Browse", command=open_file)
browse_btn.grid(row=0, column=2, padx=10, pady=5)

# Display uploaded image
image_label = Label(root)
image_label.grid(row=1, column=1, padx=10, pady=5)

# Display prediction
tk.Label(root, textvariable=predicted_plate, fg="blue").grid(row=2, column=1, padx=10, pady=5)

# Correct label input
tk.Label(root, text="Correct Plate (if incorrect):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
label_entry = tk.Entry(root, textvariable=label_var, width=40)
label_entry.grid(row=3, column=1, padx=10, pady=5)

check_btn = tk.Button(root, text="Check", command=check_prediction)
check_btn.grid(row=4, column=0, pady=10)

submit_btn = tk.Button(root, text="Submit", command=submit)
submit_btn.grid(row=4, column=1, pady=10)

root.mainloop()