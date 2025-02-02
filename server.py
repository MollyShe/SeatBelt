from flask import Flask, send_from_directory, request, jsonify, send_file
from backend.swap import FaceSwapSystem
from backend.perturb import UnGANableDefense
import os
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
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
        # Save uploaded files with original filenames
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_file.filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_file.filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_result.jpg')

        if os.path.exists(output_path):
            os.remove(output_path)

        source_file.save(source_path)
        target_file.save(target_path)
        
        processor = FaceSwapSystem()
        success = processor.process_images(
            source_path=source_path,
            target_path=target_path,
            output_path=output_path
        )

        if not success:
            return jsonify({'error': 'Face swap failed'}), 500

        with open(output_path, 'rb') as f:
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
    
@app.route('/api/perturbate', methods=['POST'])
def perturbate_endpoint():
    # Get epsilon from query parameters, default to 0.1 if not provided
    epsilon = float(request.args.get('epsilon', 0.1)) / 100.0  # Convert from percentage to decimal
    
    if 'source_img' not in request.files:
        return jsonify({'error': 'Missing required files'}), 400
    
    source_file = request.files['source_img']
    
    if not (source_file and allowed_file(source_file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded files with original filenames
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_file.filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'perturbed_result.jpg')

        source_file.save(source_path)
        
        defense = UnGANableDefense(
            epsilon=epsilon,  # Now using the query parameter
            num_iterations=1,
            reg_lambda=0.01
        )
        
        if os.path.exists(source_path):
            defense.protect_image(source_path, output_path)
        else:
            print(f"Input image not found: {output_path}")
            return jsonify({'error': 'Input file not found'}), 404

        with open(output_path, 'rb') as f:
            result = f.read()
            
        # Clean up uploaded files
        os.remove(source_path)
        os.remove(output_path)
        
        return send_file(
            io.BytesIO(result),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='perturbed-result.jpg'
        )
        
    except Exception as e:
        logger.error(f"Error in perturbation: {str(e)}")
        return jsonify({'error': 'Perturbation failed'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)