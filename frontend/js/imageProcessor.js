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
    
    canvas.width = 1024;
    canvas.height = 1024;
    
    // Clear any existing filters
    ctx.filter = 'none';
    
    // Calculate scaling and positioning for center-fit
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    
    const x = (canvas.width - scaledWidth) / 2;
    const y = (canvas.height - scaledHeight) / 2;

    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.drawImage(originalImage, x, y, scaledWidth, scaledHeight);
    applyFilters();
}