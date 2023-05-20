from __future__ import division, print_function
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
from flask import Flask,request, render_template
from werkzeug.utils import secure_filename
from keras.utils import load_img

app = Flask(__name__,static_folder='uploads')
MODEL_PATH = 'Fruits/fruits_model.h5'

model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        labels=['rottenbanana', 'freshoranges', 'rottenoranges', 'freshbanana', 'rottenapples', 'freshapples']
        preds = model_predict(file_path, model)
        pred_class = preds.argmax(axis=-1)
        result = labels[pred_class[0]].title()
        f_name="uploads/"+f.filename
        return render_template("detail.html", file_loc=file_path.replace('\\','/'), predition=result,file_name=f.filename,image=f_name)
    return None
if __name__ == '__main__':
    app.run(debug=True)