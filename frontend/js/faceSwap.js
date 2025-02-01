let sourceImage = null;
let targetImage = null;

function initFaceSwap() {
    const sourceImageInput = document.getElementById('sourceImageInput');
    const targetImageInput = document.getElementById('targetImageInput');
    const swapFacesBtn = document.getElementById('swapFacesBtn');
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
        }
    });

    // Handle target image upload
    targetImageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            targetImage = file;
            updateDropZonePreview('targetImageZone', file);
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
                faceSwapCanvas.width = resultImage.width;
                faceSwapCanvas.height = resultImage.height;
                const ctx = faceSwapCanvas.getContext('2d');
                ctx.drawImage(resultImage, 0, 0);
            };
            resultImage.src = URL.createObjectURL(result);

        } catch (error) {
            console.error('Face swap failed:', error);
            alert('Face swap failed. Please try again.');
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
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    // Get the response as a blob
    return await response.blob();
}

function updateDropZonePreview(zoneId, file) {
    const dropZone = document.getElementById(zoneId);
    const preview = dropZone.querySelector('img') || new Image();
    preview.className = 'preview-image';
    
    if (!dropZone.querySelector('img')) {
        dropZone.appendChild(preview);
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Add this CSS to your styles.css file
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
    }

    .preview-image {
        max-width: 90%;
        max-height: 200px;
        margin-top: 10px;
        border-radius: 4px;
        object-fit: contain;
    }

    .drop-zone {
        flex-direction: column;
        align-items: center;
        padding: 20px;
        text-align: center;
    }

    #faceSwapCanvas {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
    }
`;

document.head.appendChild(style);