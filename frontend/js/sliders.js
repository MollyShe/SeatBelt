function initSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;

        valueDisplay.textContent = slider.value + '%';
        
        slider.addEventListener('input', () => {
            valueDisplay.textContent = slider.value + '%';
            applyFilters();
        });
    });

    // Reset button functionality
    document.getElementById('resetBtn').addEventListener('click', () => {
        sliders.forEach(slider => {
            const defaultValue = slider.id == 'seatbelt' ? 0 : 100;
            slider.value = defaultValue;
            slider.nextElementSibling.textContent = slider.value + '%';
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

    // reset filters
    ctx.filter = 'none';
    // apply filters
    ctx.filter = `brightness(${brightness}%) saturate(${saturation}%) contrast(${contrast}%)`;

    ctx.drawImage(originalImage, x, y, scaledWidth, scaledHeight);
}