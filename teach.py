import os
from hezar.models import Model

os.environ["http_proxy"] = "http://127.0.0.1:2081"
os.environ["https_proxy"] = "http://127.0.0.1:2081"


model = Model.load("hezarai/crnn-fa-license-plate-recognition")
plate_text = model.predict("assets/Personal-license-plate.jpg")
print(plate_text)