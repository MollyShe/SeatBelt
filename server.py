from flask import Flask, send_from_directory, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve frontend files
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/face-swap')
def serve_face_swap():
    return send_from_directory('frontend', 'face-swap.html')

# Serve static files (CSS, JS)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

# API endpoints
@app.route('/api/face-swap', methods=['POST'])
def face_swap():
    if 'source_img' not in request.files or 'target_img' not in request.files:
        return jsonify({'error': 'Missing required files'}), 400
    
    source_file = request.files['source_img']
    target_file = request.files['target_img']
    
    if not (source_file and allowed_file(source_file.filename) and
            target_file and allowed_file(target_file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded files
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                 secure_filename(source_file.filename))
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                 secure_filename(target_file.filename))
        
        source_file.save(source_path)
        target_file.save(target_path)
        
        # TODO: Implement actual face swap logic here
        # For now, just return the target image
        with open(target_path, 'rb') as f:
            result = f.read()
            
        # Clean up uploaded files
        os.remove(source_path)
        os.remove(target_path)
        
        return send_file(
            io.BytesIO(result),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='face-swap-result.jpg'
        )
        
    except Exception as e:
        logger.error(f"Error in face swap: {str(e)}")
        return jsonify({'error': 'Face swap failed'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)