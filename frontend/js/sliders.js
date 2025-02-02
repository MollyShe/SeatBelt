function initSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;

        updateSliderValue(slider, valueDisplay);
        
        // Handle continuous updates for visual feedback
        slider.addEventListener('input', () => {
            updateSliderValue(slider, valueDisplay);
            if (slider.id !== 'seatbelt') {
                applyFilters();
            }
        });

        // Handle slider release for seatbelt/perturbate
        slider.addEventListener('change', async () => {
            if (slider.id === 'seatbelt') {
                await handlePerturbate(slider.value);
            }
        });
    });

    // Reset button functionality
    document.getElementById('resetBtn').addEventListener('click', () => {
        sliders.forEach(slider => {
            const valueDisplay = slider.nextElementSibling;
            const defaultValue = slider.id == 'seatbelt' ? 0 : 100;
            slider.value = defaultValue;
            updateSliderValue(slider, valueDisplay);
        });
        applyFilters();
    });

    // Download button functionality
    document.getElementById('downloadBtn').addEventListener('click', () => {
        const link = document.createElement('a');
        link.download = 'Perturbed_Image.png';
        link.href = canvas.toDataURL();
        link.click();
    });
}

async function handlePerturbate(epsilon) {
    if (!canvas || !ctx || !originalImage) return;

    try {
        // Create a loading indicator
        const loadingIndicator = document.createElement('div');
        loadingIndicator.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 8px;
            z-index: 1000;
            font-family: "Nunito", serif;
            font-weight: 800;
        `;
        loadingIndicator.textContent = 'Processing...';
        canvas.parentElement.appendChild(loadingIndicator);

        // Get current canvas state as a blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.95);
        });

        // Create FormData with current canvas state
        const formData = new FormData();
        formData.append('source_img', blob, 'current_state.jpg');

        // Call the perturbate endpoint
        const response = await fetch(`/api/perturbate?epsilon=${epsilon}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Get the result as a blob
        const resultBlob = await response.blob();

        // Create an image from the result
        const img = new Image();
        img.onload = () => {
            // Store as new original image for further editing
            originalImage = img;
            // Apply current filters to the new image
            applyFilters();
            // Remove loading indicator
            loadingIndicator.remove();
        };
        img.src = URL.createObjectURL(resultBlob);

    } catch (error) {
        console.error('Perturbation failed:', error);
        alert('Failed to process image. Please try again.');
        // Remove loading indicator on error
        document.querySelector('.loading-indicator')?.remove();
    }
}

function applyFilters() {
    if(!canvas || !ctx || !originalImage) return;

    const brightness = document.getElementById('brightness').value;
    const saturation = document.getElementById('saturation').value;
    const contrast = document.getElementById('contrast').value;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const scale = Math.min(canvas.width / originalImage.width, canvas.height / originalImage.height);
    const scaledWidth = originalImage.width * scale;
    const scaledHeight = originalImage.height * scale;
    const x = (canvas.width - scaledWidth) / 2;
    const y = (canvas.height - scaledHeight) / 2;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = scaledWidth;
    tempCanvas.height = scaledHeight;
    const tempCtx = tempCanvas.getContext('2d');

    // Reset filters
    ctx.filter = 'none';
    tempCtx.filter = 'none';
    // Apply filters
    tempCtx.filter = `brightness(${brightness}%) saturate(${saturation}%) contrast(${contrast}%)`;
    tempCtx.drawImage(originalImage, 0, 0, scaledWidth, scaledHeight);

    ctx.drawImage(tempCanvas, x, y);
}

function updateSliderValue(slider, valueDisplay) {
    const value = slider.value;
    valueDisplay.textContent = value + '%';

    const percent = (value - slider.min) / (slider.max - slider.min);
    const sliderRect = slider.getBoundingClientRect();
    const thumbWidth = 20; 

    const leftPosition = percent * (sliderRect.width - thumbWidth);
    valueDisplay.style.position = 'absolute';
    valueDisplay.style.left = `${leftPosition}px`;
}