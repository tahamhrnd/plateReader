import torch
import cv2
from torchvision import transforms
from hezar.models import Model
from PIL import Image

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
])

model = Model.load("hezarai/crnn-fa-license-plate-recognition")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CTCLoss(blank=0)  # مقدار blank تنظیم شده است.

def predict_and_correct(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 64))
    image = Image.fromarray(image)
    input_tensor = transform(image).unsqueeze(0)

    # پیش‌بینی
    with torch.no_grad():
        prediction = model.predict(image_path)
    print("Predicted Plate:", prediction)

    # اصلاح در صورت اشتباه
    correct_label = input("If incorrect, enter the correct plate (or press Enter if correct): ")
    if correct_label:
        update_model(model, input_tensor, correct_label)
    else:
        print("Prediction was correct!")


def update_model(model, input_tensor, correct_label):
    optimizer.zero_grad()

    # تبدیل correct_label به قالب موردنیاز
    target = torch.tensor([ord(c) - ord('0') for c in correct_label], dtype=torch.long).unsqueeze(
        0)  # تبدیل رشته به Tensor
    target_lengths = torch.tensor([len(correct_label)], dtype=torch.long)  # طول توالی هدف

    # پیش‌بینی مدل
    output = model(input_tensor)

    # استخراج احتمالات لاگاریتمی از logits
    if isinstance(output, dict) and 'logits' in output:
        log_probs = output['logits']
    else:
        raise ValueError("Output of model does not contain 'logits' key.")

    # استخراج طول ورودی از log_probs
    T = log_probs.size(0)  # تعداد گام‌های زمانی
    input_lengths = torch.tensor([T], dtype=torch.long)  # طول ورودی

    # محاسبه Loss
    loss = criterion(log_probs, target, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()

    print("Model updated with new data.")


# اجرا
predict_and_correct("assets/image.jpg")
