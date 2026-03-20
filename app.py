from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import re


app = Flask(__name__)
app.secret_key = 'secretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ---------------- USER MODEL ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))


with app.app_context():
    db.create_all()


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
    return render_template('about.html')


# -------- INDEX --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)