from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from app.routes.main_routes import main

app = Flask(__name__)
app.secret_key = "chat-scholar-secret-key"

# Set max file upload size to 100MB for handling large PDF documents
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

app.register_blueprint(main)

# Request Entity Too Large
@app.errorhandler(413)
def request_entity_too_large(error):
    return {
        "error": "File too large",
        "message": "Uploaded file exceeds the 100MB size limit. Please select a smaller PDF file."
    }, 413

if __name__ == "__main__":
    app.run(debug=True)
