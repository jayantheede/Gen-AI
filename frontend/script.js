document.getElementById('discoverBtn').addEventListener('click', performSearch);
document.getElementById('queryInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = document.getElementById('queryInput').value.trim();
    const ragMode = document.getElementById('ragMode').value;

    if (!query) return;

    // Show Loader
    const loader = document.getElementById('loader');
    loader.classList.remove('hidden');

    // Clear previous
    const resultsArea = document.getElementById('resultsArea');
    resultsArea.classList.add('hidden');
    const imageGallery = document.getElementById('imageGallery');
    imageGallery.innerHTML = '';
    const badgesContainer = document.getElementById('badgesContainer');
    // Keep base badges, clear others
    badgesContainer.innerHTML = '<span class="badge">CLIP-MATCHED</span><span class="badge">VECTOR-VERIFIED</span>';

    try {
        const startTime = Date.now();
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: query, rag_mode: ragMode })
        });

        if (!response.ok) throw new Error('Network error');

        const data = await response.json();
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);

        renderResults(data, duration);
    } catch (err) {
        alert('An error occurred while consulting the architectural records: ' + err.message);
    } finally {
        loader.classList.add('hidden');
    }
}

function renderResults(data, duration) {
    const resultsArea = document.getElementById('resultsArea');
    const aiResponse = document.getElementById('aiResponse');
    const imageGallery = document.getElementById('imageGallery');
    const timingInfo = document.getElementById('timingInfo');
    const badgesContainer = document.getElementById('badgesContainer');

    // Text Answer
    aiResponse.innerHTML = marked.parse(data.answer);
    timingInfo.textContent = `Architectural analysis complete in ${data.generation_time || duration + 's'}`;

    // Mode specific badges
    if (data.mode === 'corrective' && data.relevance_score) {
        addBadge(`RELEVANCE: ${data.relevance_score.toFixed(2)}`, '#f59e0b');
    } else if (data.mode === 'speculative' && data.entities) {
        data.entities.slice(0, 2).forEach(ent => addBadge(`ENTITY: ${ent}`, '#8b5cf6'));
    }
    addBadge(`MODE: ${data.mode.toUpperCase()}`, '#6366f1');

    // Images
    if (data.images && data.images.length > 0) {
        data.images.forEach(img => {
            const card = document.createElement('div');
            card.className = 'image-card';

            // Map static paths
            let imgSrc = img.image_path;
            if (imgSrc.includes('Data/')) {
                // Ensure forward slashes and grab only filename
                const parts = imgSrc.split(/[\\/]/);
                const filename = parts.pop();
                imgSrc = `/images/${filename}`;
            }

            const caption = img.ocr_text ? `"${img.ocr_text.substring(0, 50)}..."` : 'Visual Reference';

            card.innerHTML = `
                <img src="${imgSrc}" onerror="this.src='https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?auto=format&fit=crop&q=80&w=400'" alt="Design Reference">
                <div class="image-info">
                    <span class="image-caption">${caption}</span>
                    <div class="image-meta">
                        <span>${img.pdf} • Pg ${img.page}</span>
                        <span style="color: #10b981;">${img.score.toFixed(2)}</span>
                    </div>
                    ${img.pdf_url ? `<a href="${img.pdf_url}" target="_blank" class="cta-btn">OPEN CATALOG</a>` : ''}
                </div>
            `;
            imageGallery.appendChild(card);
        });
    } else {
        imageGallery.innerHTML = '<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted);">No exact visual matches found in current catalog version.</p>';
    }

    resultsArea.classList.remove('hidden');
    resultsArea.scrollIntoView({ behavior: 'smooth' });
}

function addBadge(text, color) {
    const badgesContainer = document.getElementById('badgesContainer');
    const b = document.createElement('span');
    b.className = 'badge';
    b.textContent = text;
    b.style.backgroundColor = color + '22';
    b.style.color = color;
    b.style.borderColor = color + '44';
    badgesContainer.appendChild(b);
}
