document.addEventListener('DOMContentLoaded', () => {
    const btnNext = document.getElementById('btn-next');
    const btnReset = document.getElementById('btn-reset');
    const stepName = document.getElementById('step-name');
    
    // We have 5 steps total (indices 0 to 4)
    const totalSteps = 5;
    let currentStep = 0;

    const stepTitles = [
        "Step 1: Input Sequence",
        "Step 2: Tokenization",
        "Step 3: QKV Matrices",
        "Step 4: Attention Heatmap",
        "Step 5: Output Prediction"
    ];

    function updateView() {
        // Hide all steps
        for (let i = 0; i < totalSteps; i++) {
            const stepEl = document.getElementById(`step-${i}`);
            if (stepEl) {
                stepEl.style.display = 'none';
            }
        }

        // Show current step
        const currentEl = document.getElementById(`step-${currentStep}`);
        if (currentEl) {
            currentEl.style.display = 'block';
        }

        // Update indicator
        stepName.textContent = stepTitles[currentStep];

        // Update button states
        if (currentStep === totalSteps - 1) {
            btnNext.textContent = "Finish";
            btnNext.disabled = true;
            btnNext.style.opacity = '0.5';
        } else {
            btnNext.textContent = "Next Step";
            btnNext.disabled = false;
            btnNext.style.opacity = '1';
        }
    }

    btnNext.addEventListener('click', () => {
        if (currentStep < totalSteps - 1) {
            currentStep++;
            updateView();
        }
    });

    btnReset.addEventListener('click', () => {
        currentStep = 0;
        updateView();
    });

    // Initialize
    updateView();
});
