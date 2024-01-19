from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import os
import torch 
from torchvision import transforms as tt
from torchvision import models
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 5)
resnet = model
map_location=torch.device('cpu')
resnet.load_state_dict(torch.load('model.pth', map_location=map_location))
class_dic = json.load(open('cat.json'))

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = tt.Compose([tt.ToTensor(), tt.Resize((224,224), antialias = True), tt.Lambda(lambda x: x[:3]), tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(image).unsqueeze(0) # unsqueeze to add artificial first dimension of the batch size

def make_prediction(img):

    resnet.eval()
    with torch.no_grad():
        preds = resnet(img)
        _, predicted = torch.max(preds, dim=1)
    cat = class_dic[str(predicted.item())]
    split_cat = cat.split('-')[1]
    split_cat = split_cat.replace('_', ' ')
    return split_cat




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('home.html', prediction='No file selected pz', image = None)
        file = request.files['image']
        if file.filename == '':
            return render_template('home.html', prediction='No file selected', image = None)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            prediction = make_prediction(img)
            return render_template('home.html', prediction = prediction, image = file_path)
        else:
            return render_template('home.html', prediction = 'Invalid file type', image = None)



if __name__ == '__main__':
    app.run()