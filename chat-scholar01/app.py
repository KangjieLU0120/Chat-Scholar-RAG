from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from app.routes.main_routes import main

app = Flask(__name__)
app.secret_key = "chat-scholar-secret-key"

app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
