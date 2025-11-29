const form = document.getElementById('tumor-form');
const reportContainer = document.getElementById('report-container');

form.addEventListener('submit', (e) => {
    e.preventDefault();
    const tumorType = document.getElementById('tumor-type').value;
    const tumorGrade = document.getElementById('tumor-grade').value;
    const treatmentOptions = getTreatmentOptions(tumorGrade);
    const prognosis = getPrognosis(tumorGrade);
    const report = generateReport(tumorType, tumorGrade, treatmentOptions, prognosis);
    reportContainer.innerHTML = report;
});

function getTreatmentOptions(tumorGrade) {
    // Return treatment options based on tumor grade
    switch (tumorGrade) {
        case '1':
            return 'Surgery';
        case '2':
            return 'Surgery, Radiation Therapy';
        case '3':
            return 'Surgery, Radiation Therapy, Chemotherapy';
        case '4':
            return 'Aggressive treatment, including surgery, radiation therapy, and chemotherapy';
        default:
            return '';
    }
}

function getPrognosis(tumorGrade) {
    // Return prognosis based on tumor grade
    switch (tumorGrade) {
        case '1':
            return 'Good';
        case '2':
            return 'Fair';
        case '3':
            return 'Poor';
        case '4':
            return 'Very Poor';
        default:
            return '';
    }
}


function generateReport(tumorType, tumorGrade, treatmentOptions, prognosis) {
    return ` 
        <h2>Tumor Grade Report</h2>
        <p><strong>Tumor Type:</strong> ${tumorType}</p>
        <p><strong>Tumor Grade:</strong> Grade ${tumorGrade}</p>
        <p><strong>What it means:</strong> A Grade ${tumorGrade} tumor is considered ${getTumorGradeDescription(tumorGrade)}.</p>
        <p><strong>Treatment Options:</strong> ${treatmentOptions}</p>
        <p><strong>Prognosis:</strong> ${prognosis}</p> 
        `;
}