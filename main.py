import tkinter as tk
from tkinter import filedialog, messagebox, Label
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from hezar.models import Model
import os




# 1. تنظیمات و پیش‌پردازش تصویر
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
])

model_path = "fine_tuned_model.pth"

# اول مدل پایه رو بارگذاری کن (بدون وزن‌های ذخیره شده)
model = Model.load("hezarai/crnn-fa-license-plate-recognition")

if os.path.exists(model_path):
    print("Loading saved model weights...")
    try:
        state_dict = torch.load(model_path)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Warning: The saved file does not contain state_dict. Skipping loading weights.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
else:
    print("No saved model weights found. Saving initial weights...")
    torch.save(model.state_dict(), model_path)

# تنظیمات بهینه‌ساز و معیار
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CTCLoss(blank=0)

# 3. پیش‌بینی پلاک
def predict_plate(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 64))
        input_tensor = transform(Image.fromarray(image)).unsqueeze(0)

        with torch.no_grad():
            prediction = model.predict(image_path)
        return prediction
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

# 4. به‌روزرسانی مدل با اصلاحات کاربر
def update_model(input_tensor, correct_label):
    try:
        optimizer.zero_grad()

        # فرض بر این است که correct_label حاوی کاراکترهای عددی است و به شکل درست تبدیل می‌شود
        target = torch.tensor([ord(c) - ord('0') for c in correct_label], dtype=torch.long).unsqueeze(0)
        target_lengths = torch.tensor([len(correct_label)], dtype=torch.long)

        output = model(input_tensor)
        if isinstance(output, dict) and 'logits' in output:
            log_probs = output['logits']
        else:
            raise ValueError("Model output does not contain 'logits' key.")

        T = log_probs.size(0)
        input_lengths = torch.tensor([T], dtype=torch.long)
        loss = criterion(log_probs, target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        # ذخیره مدل
        torch.save(model.state_dict(), model_path)
        messagebox.showinfo("Info", "Model updated and saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Model update failed: {str(e)}")

def reset_model():
    try:
        if os.path.exists(model_path):
            os.remove(model_path)  # حذف فایل وزن ذخیره شده
        # مدل رو مجدد بارگذاری می‌کنیم (وزن‌های اولیه)
        global model
        model = Model.load("hezarai/crnn-fa-license-plate-recognition")
        # وزن‌های اولیه رو ذخیره می‌کنیم تا دفعه بعدی هم از اول شروع بشه
        torch.save(model.state_dict(), model_path)
        messagebox.showinfo("Info", "Model has been reset to initial state.")
        predicted_plate.set("")  # پاک کردن پیش‌بینی قبلی از UI
    except Exception as e:
        messagebox.showerror("Error", f"Failed to reset model: {str(e)}")

# 5. بررسی پیش‌بینی
def check_prediction():
    image_path = file_path_var.get()
    if not image_path:
        messagebox.showwarning("Warning", "Please upload an image.")
        return

    try:
        prediction = predict_plate(image_path)
        predicted_plate.set(f"Predicted Plate: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {str(e)}")

# 6. ارسال و اصلاح مدل
def submit():
    image_path = file_path_var.get()
    correct_label = label_var.get()

    if not image_path:
        messagebox.showwarning("Warning", "Please upload an image.")
        return

    if not correct_label:
        messagebox.showwarning("Warning", "Please enter a correct plate.")
        return

    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 64))
        input_tensor = transform(Image.fromarray(image)).unsqueeze(0)

        update_model(input_tensor, correct_label)

        # حذف load_state_dict بعد از آپدیت چون مدل در حافظه است
        messagebox.showinfo("Info", "Model updated and saved!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# 7. باز کردن فایل و نمایش تصویر
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        file_path_var.set(file_path)
        display_image(file_path)

def display_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((256, 128))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display image: {str(e)}")

# 8. رابط کاربری گرافیکی (GUI)
root = tk.Tk()
root.title("License Plate Recognition")

file_path_var = tk.StringVar()
label_var = tk.StringVar()
predicted_plate = tk.StringVar()

tk.Label(root, text="Upload Image:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
file_entry = tk.Entry(root, textvariable=file_path_var, width=40)
file_entry.grid(row=0, column=1, padx=10, pady=5)
browse_btn = tk.Button(root, text="Browse", command=open_file)
browse_btn.grid(row=0, column=2, padx=10, pady=5)

image_label = Label(root)
image_label.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, textvariable=predicted_plate, fg="blue").grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Correct Plate (if incorrect):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
label_entry = tk.Entry(root, textvariable=label_var, width=40)
label_entry.grid(row=3, column=1, padx=10, pady=5)

check_btn = tk.Button(root, text="Check", command=check_prediction)
check_btn.grid(row=4, column=0, pady=10)
submit_btn = tk.Button(root, text="Submit", command=submit)
submit_btn.grid(row=4, column=1, pady=10)
reset_btn = tk.Button(root, text="Reset Model", command=reset_model)
reset_btn.grid(row=4, column=2, pady=10, padx=10)

root.mainloop()
