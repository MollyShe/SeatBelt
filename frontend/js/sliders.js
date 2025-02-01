function initSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;
        
        slider.addEventListener('input', () => {
            let unit = '';
            switch(slider.id) {
                case 'blur':
                    unit = 'px';
                    break;
                default:
                    unit = '%';
            }
            valueDisplay.textContent = slider.value + unit;
            applyFilters();
        });
    });

    // Reset button functionality
    document.getElementById('resetBtn').addEventListener('click', () => {
        sliders.forEach(slider => {
            slider.value = slider.id === 'blur' ? 0 : 100;
            slider.nextElementSibling.textContent = slider.value + 
                (slider.id === 'blur' ? 'px' : '%');
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