function initSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;

        updateSliderValue(slider, valueDisplay);
        
        slider.addEventListener('input', () => {
            updateSliderValue(slider, valueDisplay);
            applyFilters();
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
        link.download = 'edited-image.png';
        link.href = canvas.toDataURL();
        link.click();
    });
}

function applyFilters() {
    if(!canvas || !ctx || !originalImage) return;

    const brightness = document.getElementById('brightness').value;
    const saturation = document.getElementById('saturation').value;
    const contrast = document.getElementById('contrast').value;
    const seatbelt = document.getElementById('seatbelt').value;

    // debug print filter values
    console.log('Filters:', { brightness, saturation, contrast, seatbelt });

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

    // reset filters
    ctx.filter = 'none';
    tempCtx.filter = 'none';
    // apply filters
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