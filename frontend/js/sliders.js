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