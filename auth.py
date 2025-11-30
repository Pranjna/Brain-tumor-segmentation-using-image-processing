from flask import Flask, request, redirect, url_for, session, send_from_directory, g
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from urllib.parse import quote_plus
import subprocess
import sys
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'users.db')

# Serve repository root files as static files at the app root so paths like
# `/try_reg.html` and `/login_1.html` work directly in the browser.
app = Flask(__name__, static_folder=APP_DIR, static_url_path='')
app.config['SECRET_KEY'] = os.environ.get('AUTH_SECRET_KEY', os.urandom(24))


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            doctor_id TEXT NOT NULL UNIQUE,
            phone TEXT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        '''
    )
    db.commit()


# Streamlit process control (start/stop/status)
STREAMLIT_PROC = None


def start_streamlit():
    global STREAMLIT_PROC
    if STREAMLIT_PROC and STREAMLIT_PROC.poll() is None:
        return False, 'already running'

    # Use the same Python interpreter that runs this Flask app (venv)
    py = sys.executable
    cmd = [py, '-m', 'streamlit', 'run', 'try_btapp_1.py', '--server.port', '8501']
    # Start streamlit in repository root
    STREAMLIT_PROC = subprocess.Popen(cmd, cwd=APP_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # give it a moment to start
    time.sleep(0.5)
    if STREAMLIT_PROC.poll() is None:
        return True, 'started'
    else:
        return False, 'failed to start'


def stop_streamlit():
    global STREAMLIT_PROC
    if not STREAMLIT_PROC:
        return False, 'not running'
    if STREAMLIT_PROC.poll() is not None:
        STREAMLIT_PROC = None
        return False, 'process already exited'
    STREAMLIT_PROC.terminate()
    try:
        STREAMLIT_PROC.wait(timeout=5)
    except Exception:
        STREAMLIT_PROC.kill()
    STREAMLIT_PROC = None
    return True, 'stopped'


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return send_from_directory(APP_DIR, 'try_reg.html')

    # POST -> process registration
    name = request.form.get('name', '').strip()
    doctor_id = request.form.get('doctor_id', '').strip()
    phone = request.form.get('phone', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    confirm = request.form.get('confirm_password', '')

    errors = []
    if not name:
        errors.append('Name is required')
    if not doctor_id or len(doctor_id) < 4:
        errors.append('Doctor ID must be at least 4 characters')
    if not email:
        errors.append('Email is required')
    if not password or len(password) < 6:
        errors.append('Password must be at least 6 characters')
    if password != confirm:
        errors.append('Passwords do not match')

    if errors:
        # redirect back to the registration page with an encoded error message
        msg = quote_plus('; '.join(errors))
        return redirect(url_for('register') + f'?error={msg}')

    db = get_db()
    cur = db.cursor()
    # check uniqueness
    cur.execute('SELECT id FROM users WHERE email = ? OR doctor_id = ?', (email, doctor_id))
    if cur.fetchone():
        msg = quote_plus('User with that email or doctor id already exists')
        return redirect(url_for('register') + f'?error={msg}')

    password_hash = generate_password_hash(password)
    created_at = datetime.utcnow().isoformat()
    cur.execute(
        'INSERT INTO users (name, doctor_id, phone, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)',
        (name, doctor_id, phone, email, password_hash, created_at)
    )
    db.commit()

    return redirect(url_for('login') + '?registered=1')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return send_from_directory(APP_DIR, 'login_1.html')

    # POST -> process login
    doctor_id = request.form.get('doctor_id', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')

    if not doctor_id or not email or not password:
        return redirect(url_for('login') + '?error=' + quote_plus('Missing credentials'))

    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT * FROM users WHERE email = ? AND doctor_id = ?', (email, doctor_id))
    row = cur.fetchone()
    if not row:
        return redirect(url_for('login') + '?error=' + quote_plus('Invalid credentials'))

    if not check_password_hash(row['password_hash'], password):
        return redirect(url_for('login') + '?error=' + quote_plus('Invalid credentials'))

    # success: set session
    session['user_id'] = row['id']
    session['user_name'] = row['name']
    # Start Streamlit automatically on successful login and redirect there
    ok, msg = start_streamlit()
    if ok:
        return redirect('http://localhost:8501')
    # If Streamlit failed to start, fall back to dashboard and show message
    return redirect(url_for('dashboard') + '?streamlit_error=' + quote_plus(msg))


# Allow requests to the literal HTML filenames to work in case links use them.
@app.route('/try_reg.html')
def try_reg_html():
    return redirect(url_for('register'))


@app.route('/login_1.html')
def login_html():
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return f"<h2>Welcome, {session.get('user_name')}!</h2><p><a href=\"/logout\">Logout</a></p>\n" \
           f"<p><a href=\"/streamlit/start\">Start Streamlit</a> | <a href=\"/streamlit/stop\">Stop Streamlit</a> | <a href=\"http://localhost:8501\" target=\"_blank\">Open Streamlit</a></p>"


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/streamlit/start')
def streamlit_start():
    ok, msg = start_streamlit()
    if ok:
        return redirect('http://localhost:8501')
    return f"Could not start Streamlit: {msg}", 500


@app.route('/streamlit/stop')
def streamlit_stop():
    ok, msg = stop_streamlit()
    if ok:
        return redirect(url_for('dashboard'))
    return f"Could not stop Streamlit: {msg}", 500


@app.route('/streamlit/status')
def streamlit_status():
    if STREAMLIT_PROC and STREAMLIT_PROC.poll() is None:
        return {'status': 'running'}
    return {'status': 'stopped'}


if __name__ == '__main__':
    # ensure DB exists
    with app.app_context():
        init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
