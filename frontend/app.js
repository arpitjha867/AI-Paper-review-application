document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('pdfFile');
    const useMock = document.getElementById('useMock').checked;
    const useCloud = document.getElementById('useCloud').checked;
    const llmBackend = useMock ? 'mock' : (useCloud ? 'claude' : 'local');
    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const reviewContent = document.getElementById('reviewContent');

    if (!fileInput.files[0]) {
        alert('Please select a PDF file.');
        return;
    }

    submitBtn.disabled = true;
    loading.classList.remove('hidden');
    results.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('llm_backend', llmBackend);

    try {
        const response = await fetch('/api/review', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Review failed');
        }

        const review = await response.json();
        if (review.error) {
            alert('Server error: ' + review.error);
        } else {
            displayReview(review);
            results.classList.remove('hidden');
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        loading.classList.add('hidden');
    }
});

function displayReview(review) {
    const content = `
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h3 class="text-xl font-semibold mb-4">Strengths</h3>
            <p class="mb-4">${review.strengths || 'None specified'}</p>

            <h3 class="text-xl font-semibold mb-4">Weaknesses</h3>
            <p class="mb-4">${review.weaknesses || 'None specified'}</p>

            <h3 class="text-xl font-semibold mb-4">Missing Related Work</h3>
            <p class="mb-4">${review.missing_related_work || 'None specified'}</p>

            <h3 class="text-xl font-semibold mb-4">Questions for Authors</h3>
            <p class="mb-4">${review.questions_for_authors || 'None specified'}</p>

            <h3 class="text-xl font-semibold mb-4">Suggested Improvements</h3>
            <p class="mb-4">${review.suggested_improvements || 'None specified'}</p>

            <h3 class="text-xl font-semibold mb-4">Scores</h3>
            <ul class="list-disc list-inside">
                <li>Originality: ${review.scores.originality}/5</li>
                <li>Experimental Soundness: ${review.scores.experimental_soundness}/5</li>
                <li>Clarity: ${review.scores.clarity}/5</li>
                <li>Prior Work Coverage: ${review.scores.prior_work_coverage}/5</li>
                <li>Research Value: ${review.scores.research_value}/5</li>
                <li><strong>Final Score: ${review.scores.final_score.toFixed(1)}/5</strong></li>
            </ul>
        </div>
    `;
    document.getElementById('reviewContent').innerHTML = content;
}