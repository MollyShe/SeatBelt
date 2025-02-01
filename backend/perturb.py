import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

class FaceProtector:
    def __init__(self):
        # Initialize face analysis components
        self.face_analyser = FaceAnalysis(name='buffalo_l', root='./checkpoints')
        self.face_analyser.prepare(ctx_id=0, det_size=(320, 320))
        
        # Load face swap model
        self.swap_model = onnxruntime.InferenceSession(
            "checkpoints/inswapper_128.onnx",
            providers=['CPUExecutionProvider']
        )
        
        # Model configuration
        self.input_size = 128
        self.epsilon = 0.08  # Maximum perturbation strength
        self.steps = 40      # Number of attack iterations

    def _get_face(self, img_path):
        """Detect and extract face information"""
        img = cv2.imread(img_path)
        if img is None:
            return None
        faces = self.face_analyser.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return faces[0] if faces else None

    def _preprocess(self, img):
        """Prepare image for model input"""
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.transpose(2, 0, 1).astype(np.float32)  # HWC to CHW
        img = (img - 127.5) / 127.5  # Model-specific normalization
        return np.expand_dims(img, axis=0)  # Add batch dimension

    def _postprocess(self, img):
        """Convert model output back to image format"""
        img = (img * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        return img.transpose(1, 2, 0)  # CHW to HWC

    def protect_image(self, source_path, target_path, output_path):
        """Create protected version of target image"""
        # Get face embeddings
        source_face = self._get_face(source_path)
        target_face = self._get_face(target_path)
        if not source_face or not target_face:
            return False, "Face detection failed"

        # Original image processing
        original_img = cv2.imread(target_path)
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        processed_img = self._preprocess(img_rgb)
        adversarial_img = processed_img.copy()

        # Adversarial attack loop
        for _ in range(self.steps):
            # Prepare model inputs
            inputs = {
                'source': source_face.embedding.reshape(1, -1).astype(np.float32),
                'target': adversarial_img.astype(np.float32)
            }

            # Get model output
            output = self.swap_model.run(None, inputs)[0]

            # Calculate perturbation (simulated gradient)
            perturbation = np.sign(output - processed_img)
            adversarial_img += self.epsilon * perturbation / self.steps

            # Project back to valid range
            adversarial_img = np.clip(adversarial_img, -1, 1)

        # Convert and save protected image
        protected_img = self._postprocess(adversarial_img[0])
        protected_bgr = cv2.cvtColor(protected_img, cv2.COLOR_RGB2BGR)
        
        # Resize back to original dimensions
        final_result = cv2.resize(protected_bgr, (original_img.shape[1], original_img.shape[0]))
        cv2.imwrite(output_path, final_result)
        
        return True, "Image protected successfully"

if __name__ == "__main__":
    protector = FaceProtector()
    
    success, message = protector.protect_image(
        source_path='backend/image2.jpg',
        target_path='backend/image.jpg',
        output_path='backend/perturbed_result.jpg'
    )
    
    if success:
        print("Successfully created protected image")
        print("Face swaps using this image should now fail")
    else:
        print(f"Protection failed: {message}")