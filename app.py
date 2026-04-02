from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import re
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_login import current_user

app = Flask(__name__)
app.secret_key = 'secretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# File upload configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = 'static/uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)


# ---------------- USER MODEL ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    predictions = db.relationship('Prediction', backref='user', lazy=True)


# ---------- PREDICTION MODEL ----------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease_name = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    image_filename = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    crop_type = db.Column(db.String(50))


with app.app_context():
    db.create_all()


# ============= MODEL INITIALIZATION =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_names = None
model_path = 'crop_disease_model.pth'

# Class names from PlantVillage dataset
CLASS_MAPPING = {
    'Pepper__bell___Bacterial_spot': 'Pepper - Bacterial Spot',
    'Pepper__bell___healthy': 'Pepper - Healthy',
    'Potato___Early_blight': 'Potato - Early Blight',
    'Potato___healthy': 'Potato - Healthy',
    'Potato___Late_blight': 'Potato - Late Blight',
    'Tomato__Target_Spot': 'Tomato - Target Spot',
    'Tomato__Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato - Yellow Leaf Curl Virus',
    'Tomato_Bacterial_spot': 'Tomato - Bacterial Spot',
    'Tomato_Early_blight': 'Tomato - Early Blight',
    'Tomato_healthy': 'Tomato - Healthy',
    'Tomato_Late_blight': 'Tomato - Late Blight',
    'Tomato_Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato - Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato - Spider Mites'
}

def load_model():
    """Load the trained model"""
    global model, class_names
    num_classes = len(CLASS_MAPPING)
    
    # Initialize model architecture
    model = models.mobilenet_v3_large(weights='DEFAULT')
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    # Load weights if available
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Using pretrained weights only.")
    else:
        print(f"Model file {model_path} not found. Using pretrained MobileNet V3.")
    
    model.eval()
    class_names = list(CLASS_MAPPING.values())
    return class_names

# Load model on startup
with app.app_context():
    class_names = load_model()


def preprocess_image(image_file):
    """Preprocess image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    try:
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor.to(device)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def predict_disease(image_tensor):
    """Make prediction on image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    class_name = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100
    
    return class_name, confidence_score


def allowed_file(filename):
    """Check if file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------- PASSWORD VALIDATION FUNCTION --------
def valid_password(password):

    if len(password) < 8:
        return False

    if " " in password:
        return False

    if not re.search("[A-Z]", password):   # uppercase
        return False

    if not re.search("[a-z]", password):   # lowercase
        return False

    if not re.search("[0-9]", password):   # number
        return False

    if not re.search("[@#$%^&*!]", password):  # special character
        return False

    return True


# -------- REGISTER --------
@app.route('/register', methods=['GET','POST'])

def register():

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # validations 
        if not name or len(name.strip())<2:
            flash('Name must be at least 2 characters long.', 'error')
            return redirect(url_for('register'))

        if not re.match(r'^[\w\.-]+@gmail\.com$', email):
            flash('Enter valid Gmail (example@gmail.com)', 'error')
            return redirect(url_for('register'))
        
        if not valid_password(password):
            flash('Password must contain uppercase, lowercase, number and special character', 'error')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        #check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered.Please log in.', 'error')
            return redirect(url_for('register'))

        # create new user
        hashed_password = generate_password_hash(password)
        new_user = User(name=name.strip(), 
                        email=email.strip(),
                        password=hashed_password
        )
        try: 
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            # after sign-up send user to login page
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')


# -------- LOGIN --------
@app.route('/login', methods=['GET','POST'])

def login():

    if request.method == 'POST':


        email = request.form['email'].strip()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password,password):

            session['user_id'] = user.id
            session['user_name'] = user.name

            flash("Login successful","success")

            return redirect(url_for('index'))

        else:
            flash("Invalid email or password","error")

    return render_template('login.html')




# -------- LOGOUT --------
@app.route('/logout')
def logout():

    session.clear()
    flash("Logged out successfully","success")

    return redirect(url_for('login'))


# -------- ABOUT --------
@app.route('/about')
def about():
    return render_template('about.html',current_user =current_user )


# -------- INDEX --------
@app.route('/')
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/upload')
def upload():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('login'))
    return render_template('upload.html',current_user =current_user)


# -------- PREDICT --------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
        
        # Reset file pointer
        image_file.seek(0)
        
        # Preprocess and predict
        image_tensor = preprocess_image(image_file)
        disease_name, confidence = predict_disease(image_tensor)
        
        # Save image
        filename = secure_filename(f"{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.seek(0)
        image_file.save(filepath)
        
        # Extract crop type from disease name
        crop_type = disease_name.split(' - ')[0] if ' - ' in disease_name else disease_name
        
        # Store prediction in database
        prediction = Prediction(
            user_id=session['user_id'],
            disease_name=disease_name,
            confidence=confidence,
            image_filename=filename,
            crop_type=crop_type
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'image_path': filepath
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


# -------- DASHBOARD --------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in first', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.timestamp.desc()).all()
    
    # Get statistics
    total_predictions = len(predictions)
    disease_counts = {}
    crop_counts = {}
    
    for pred in predictions:
        disease_counts[pred.disease_name] = disease_counts.get(pred.disease_name, 0) + 1
        crop_counts[pred.crop_type] = crop_counts.get(pred.crop_type, 0) + 1
    
    return render_template('dashboard.html', 
                        user=user,
                        predictions=predictions,
                        total_predictions=total_predictions,
                        disease_counts=disease_counts,
                        crop_counts=crop_counts,current_user =current_user)


if __name__ == "__main__":
    app.run(debug=True)