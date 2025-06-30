# import os
from hezar.models import Model

# os.environ["http_proxy"] = "http://127.0.0.1:2081"
# os.environ["https_proxy"] = "http://127.0.0.1:2081"
# requests.packages.urllib3.disable_warnings()
# model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition", trust_remote_code=True)

# model_path = "./models/model.pt"
#
# model = Model.load(model_path)

model = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")
plate_text = model.predict("assets/Personal-license-plate.jpg")
print(plate_text)