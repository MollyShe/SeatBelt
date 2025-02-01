let canvas, ctx, originalImage;

function initImageProcessor() {
    canvas = document.getElementById('imageCanvas');
    ctx = canvas.getContext('2d');
}

function initCanvas(img) {
    originalImage = img;
    
    // Set canvas size while maintaining aspect ratio
    const maxWidth = canvas.parentElement.clientWidth;
    const maxHeight = canvas.parentElement.clientHeight;
    const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
    
    canvas.width = img.width * ratio;
    canvas.height = img.height * ratio;
    
    applyFilters();
}

function applyFilters() {
    const brightness = document.getElementById('brightness').value;
    const contrast = document.getElementById('contrast').value;
    const saturation = document.getElementById('saturation').value;
    const blur = document.getElementById('blur').value;

    ctx.filter = `
        brightness(${brightness}%) 
        contrast(${contrast}%) 
        saturate(${saturation}%)
        blur(${blur}px)
    `;
    
    ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
}