let sourceImage = null;
let targetImage = null;

function initFaceSwap() {
    const sourceImageInput = document.getElementById('sourceImageInput');
    const targetImageInput = document.getElementById('targetImageInput');
    const swapFacesBtn = document.getElementById('swapFacesBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const faceSwapCanvas = document.getElementById('faceSwapCanvas');
    const uploadSection = document.getElementById('uploadSection');
    const faceSwapSection = document.getElementById('faceSwapSection');

    // Initialize loading indicator
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.style.display = 'none';
    loadingIndicator.innerHTML = 'Processing Face Swap...';
    document.querySelector('.main-content').appendChild(loadingIndicator);

    // Handle source image upload
    sourceImageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            sourceImage = file;
            updateDropZonePreview('sourceImageZone', file);
            updateSwapButton();
        }
    });

    // Handle target image upload
    targetImageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            targetImage = file;
            updateDropZonePreview('targetImageZone', file);
            updateSwapButton();
        }
    });

    // Handle face swap button click
    swapFacesBtn.addEventListener('click', async () => {
        if (!sourceImage || !targetImage) {
            alert('Please upload both source and target images');
            return;
        }
    
        try {
            loadingIndicator.style.display = 'flex';
            swapFacesBtn.disabled = true;
    
            const result = await performFaceSwap(sourceImage, targetImage);
            
            // Hide upload section and show result
            uploadSection.style.display = 'none';
            faceSwapSection.hidden = false;
    
            // Display result in canvas
            const resultImage = new Image();
            resultImage.onload = () => {
                // Get container dimensions
                const containerWidth = faceSwapSection.clientWidth;
                const containerHeight = faceSwapSection.clientHeight;
                
                // Calculate the aspect ratio
                const imageAspectRatio = resultImage.width / resultImage.height;
                const containerAspectRatio = containerWidth / containerHeight;
                
                let finalWidth, finalHeight;
                
                // Determine dimensions while maintaining aspect ratio
                if (imageAspectRatio > containerAspectRatio) {
                    // Image is wider relative to container
                    finalWidth = containerWidth;
                    finalHeight = containerWidth / imageAspectRatio;
                } else {
                    // Image is taller relative to container
                    finalHeight = containerHeight;
                    finalWidth = containerHeight * imageAspectRatio;
                }
                
                // Set canvas dimensions
                faceSwapCanvas.width = finalWidth;
                faceSwapCanvas.height = finalHeight;
                
                // Center the canvas in its container using CSS
                faceSwapCanvas.style.position = 'absolute';
                faceSwapCanvas.style.left = '50%';
                faceSwapCanvas.style.top = '50%';
                faceSwapCanvas.style.transform = 'translate(-50%, -50%)';
                
                // Draw the image
                const ctx = faceSwapCanvas.getContext('2d');
                ctx.drawImage(resultImage, 0, 0, finalWidth, finalHeight);
                
                // Enable download button
                downloadBtn.disabled = false;
            };
            resultImage.src = URL.createObjectURL(result);
    
        } catch (error) {
            console.error('Face swap failed:', error);
            alert('Face swap failed: ' + (error.message || 'Please try again.'));
        } finally {
            loadingIndicator.style.display = 'none';
            swapFacesBtn.disabled = false;
        }
    });
}

async function performFaceSwap(sourceImage, targetImage) {
    const formData = new FormData();
    formData.append('source_img', sourceImage);
    formData.append('target_img', targetImage);

    const response = await fetch('/api/face-swap', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    return await response.blob();
}

function updateDropZonePreview(zoneId, file) {
    const dropZone = document.getElementById(zoneId);
    const previewContainer = dropZone.querySelector('.preview-container');
    
    // Clear existing preview
    previewContainer.innerHTML = '';
    
    // Create and add new preview image
    const preview = new Image();
    preview.className = 'preview-image';
    previewContainer.appendChild(preview);

    // Read and display the file
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateSwapButton() {
    const swapFacesBtn = document.getElementById('swapFacesBtn');
    swapFacesBtn.disabled = !(sourceImage && targetImage);
}

// Add required styles
const style = document.createElement('style');
style.textContent = `
    .loading-indicator {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.2em;
        color: #2196F3;
        z-index: 1000;
        font-family: "Nunito", serif;
        font-optical-sizing: auto;
        font-weight: 800;
        font-style: normal;
    }

    .preview-image {
        max-width: 90%;
        max-height: 200px;
        margin-top: 10px;
        border-radius: 4px;
        object-fit: contain;
    }

    .preview-container {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: 10px;
    }

    .drop-zone {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        text-align: center;
        margin: 10px;
        font-family: "Nunito", serif;
        font-optical-sizing: auto;
        font-weight: 800;
        font-style: normal;
        font-size: 1.5rem;
    }

    #faceSwapCanvas {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }

    .upload-section {
        display: flex;
        width: 100%;
        height: 100%;
        gap: 20px;
        padding: 20px;
    }
`;

document.head.appendChild(style);