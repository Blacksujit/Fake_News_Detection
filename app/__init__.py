from flask import Flask

app = Flask(__name__)

print("__init__.py: before import routes")
from app import routes
print("__init__.py: after import routes")

