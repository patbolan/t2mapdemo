# wsgi.py
from app import app  # or: from myprojectimport app

# Optional: nothing else needed. Do not call app.run() here.
'''
This file is for Gunicorn deployment.
Gunicorn needs a Python import path to a WSGI “callable” (usually your Flask instance named app). 
Typical layouts:
Single file: app.py with app = Flask(__name__).
Package: myproject/__init__.py defines app = Flask(__name__).
Gunicorn will load wsgi:app, meaning “import module wsgi, use object app”.​

Quick check: In your venv, run python -c "from wsgi import app; print(app)" and ensure it imports without errors.
'''