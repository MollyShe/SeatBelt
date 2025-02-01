let canvas, ctx, originalImage;

function initImageProcessor() {
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');
}

function initCanvas(img) {
    if (!canvas || !ctx) {
        initImageProcessor();
    }
    
    originalImage = img;
    
    // Set canvas size while maintaining aspect ratio
    const maxWidth = canvas.parentElement.clientWidth;
    const maxHeight = canvas.parentElement.clientHeight || 500;
    const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
    
    canvas.width = img.width * ratio;
    canvas.height = img.height * ratio;
    
    // Clear any existing filters
    ctx.filter = 'none';

    ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
    
    // Apply any existing filter values
    applyFilters();
}
