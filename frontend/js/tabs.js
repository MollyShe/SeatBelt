document.addEventListener('DOMContentLoaded', () => {
    initTabs();
});

function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs
            tabs.forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Handle navigation
            const targetTab = tab.dataset.tab;
            if (targetTab === 'photo-editor') {
                window.location.href = 'index.html';
            } else if (targetTab === 'face-swap') {
                window.location.href = 'face-swap.html';
            }
        });
    });
}