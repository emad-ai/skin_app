import glob
import os
import imghdr
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
from gradcam import visualize_gradcam

import cv2
import numpy as np
from PIL import Image, ImageEnhance


app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class_names = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'

def focal_loss(gamma=2., alpha=0.25, num_classes=6):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        y_true = tf.cast(y_true, tf.float32)

        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return focal_loss_fixed

try:
    model = tf.keras.models.load_model('./model/skin_disease_detector.keras' , 
                                       custom_objects={"focal_loss_fixed": focal_loss(gamma=2., 
                                                                                      alpha=0.25,
                                                                                      num_classes=len(class_names))})
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

MEDICAL_ADVICE = {
    'Chickenpox': {
        'message': 'Potential Diagnosis: Chickenpox',
        'advice': [
            'Isolate patient for 5 days after rash appears',
            'Use calamine lotion for itch relief',
            'Avoid aspirin in children'
        ],
        'treatment': [
            'Oral acyclovir (800mg 4x/day for adults)',
            'Varicella vaccine for prevention'
        ],
        'sources': ['CDC 2023', 'WHO Varicella Guidelines']
    },
    'Cowpox': {
        'message': 'Potential Diagnosis: Cowpox',
        'advice': [
            'Avoid direct contact with infected animals',
            'Disinfect wounds with povidone-iodine',
            'Monitor for secondary infections'
        ],
        'treatment': [
            'Supportive care with wound management',
            'Topical cidofovir in severe cases'
        ],
        'sources': ['ECDC 2023', 'BMJ Case Reports']
    },
    'HFMD': {
        'message': 'Potential Diagnosis: Hand, Foot and Mouth Disease',
        'advice': [
            'Keep children home for 1 week',
            'Use aspirin-free pain relievers',
            'Avoid acidic foods'
        ],
        'treatment': [
            'Paracetamol (10-15mg/kg every 4-6hrs)',
            'Topical lidocaine for mouth ulcers'
        ],
        'sources': ['WHO HFMD Guidelines 2023']
    },
    'Healthy': {
        'message': 'No Pathology Detected',
        'advice': [
            'Regular skin self-exams',
            'Use SPF 30+ daily',
            'Annual dermatologist checkups'
        ],
        'treatment': ['No treatment required'],
        'sources': ['AAD Recommendations 2023']
    },
    'Measles': {
        'message': 'Potential Diagnosis: Measles',
        'advice': [
            'Strict isolation for 4 days post-rash',
            'Vitamin A supplementation',
            'Monitor for complications'
        ],
        'treatment': [
            'MMR vaccine prevention',
            'Supportive care with fluids'
        ],
        'sources': ['WHO Measles Guidelines 2023']
    },
    'Monkeypox': {
        'message': 'Potential Diagnosis: Monkeypox',
        'advice': [
            '21-day isolation until scabs fall off',
            'Use PPE when caring for patient',
            'Daily surface disinfection'
        ],
        'treatment': [
            'Tecovirimat (600mg twice daily for 14 days)',
            'JYNNEOS vaccine for high-risk groups'
        ],
        'sources': ['WHO Monkeypox Response 2023']
    }
}

def enhance_image(image_path):
    img = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)  
    return img

def extract_edges(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)  
    return edges

def isolate_skin(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_image(filepath):
    file_type = imghdr.what(filepath)
    return file_type in ALLOWED_EXTENSIONS

def clear_gradcam_images():
    gradcam_dir = os.path.join('static', 'gradcam_images')
    files = glob.glob(f"{gradcam_dir}/*")
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

def get_prediction(img, model, labels=['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'], 
                   target_size=(224, 224)):
    try:
        img = tf.keras.utils.load_img(img, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        prediction = model.predict(img_array)
        score = [tf.nn.softmax(prediction)[0][i].numpy() * 100 for i in range(len(labels))]
        highest_label = labels[np.argmax(score)]   
        return highest_label
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"
    
    
def save_image_to_class_folder(file_path, label):
    try:
        class_folder = os.path.join(app.config['UPLOAD_FOLDER'], label)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        new_file_path = os.path.join(class_folder, os.path.basename(file_path))
        os.rename(file_path, new_file_path)
        return new_file_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return file_path

 
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "error")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Invalid file format. Allowed: png, jpg, jpeg", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if not is_valid_image(file_path):
            os.remove(file_path)
            flash("Uploaded file is not a valid image", "error")
            return redirect(request.url)

        enhanced_image = enhance_image(file_path)  
        enhanced_image.save(file_path) 
 

        clear_gradcam_images()
        highest_label = get_prediction(file_path, model)

        try:
            gradcam_image = visualize_gradcam(file_path, model, last_conv_layer_name='conv2d')
            if not os.path.exists(GRADCAM_FOLDER):
                os.makedirs(GRADCAM_FOLDER)
            gradcam_image_path = os.path.join(GRADCAM_FOLDER, f'gradcam_{filename}')
            gradcam_pil_image = Image.fromarray(gradcam_image)
            gradcam_pil_image.save(gradcam_image_path)
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            gradcam_image_path = ""

        saved_file_path = save_image_to_class_folder(file_path, highest_label)

        if highest_label in MEDICAL_ADVICE:
            message = MEDICAL_ADVICE[highest_label]['message']
            advice = MEDICAL_ADVICE[highest_label]['advice']
            treatment = MEDICAL_ADVICE[highest_label]['treatment']
            sources = MEDICAL_ADVICE[highest_label]['sources']

        else:
            message = "Unknown classification"
            advice = "No advice available"
            treatment = "No treatment available"
            sources = "No treatment available"
        return render_template('diagnose.html', 
                               diagnosis=highest_label ,
                               image_url=saved_file_path, 
                               gradcam_image=gradcam_image_path, 
                               message=message, 
                               advice=advice,
                               treatment=treatment , 
                               sources=sources)

    return render_template('diagnose.html', diagnosis=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
