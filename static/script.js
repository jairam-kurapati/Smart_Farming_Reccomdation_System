/*
 * SCRIPT.JS - Smart Farming Recommendation System
 */


// ===============================
// CACHE & UI COMPONENTS
// ===============================
const weatherCache = {};

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = document.createElement('i');
    icon.className = `fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'} toast-icon`;
    
    const text = document.createElement('span');
    text.textContent = message;
    
    toast.appendChild(icon);
    toast.appendChild(text);
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        toast.addEventListener('animationend', () => toast.remove());
    }, 3000);
}

// Crop to FontAwesome Icon mapping
const cropIcons = {
    'rice': 'seedling', 'maize': 'corn', 'chickpea': 'leaf', 'kidneybeans': 'seedling',
    'pigeonpeas': 'leaf', 'mothbeans': 'leaf', 'mungbean': 'leaf', 'blackgram': 'leaf',
    'lentil': 'seedling', 'pomegranate': 'apple-alt', 'banana': 'leaf', 'mango': 'tree',
    'grapes': 'leaf', 'watermelon': 'lemon', 'muskmelon': 'lemon', 'apple': 'apple-alt',
    'orange': 'lemon', 'papaya': 'tree', 'coconut': 'tree', 'cotton': 'cloud',
    'jute': 'leaf', 'coffee': 'coffee'
};

function getCropIcon(cropName) {
    const name = cropName.toLowerCase().replace(/\s+/g, '');
    return cropIcons[name] || 'leaf';
}

function downloadPDF() {
    const element = document.getElementById('resultsContent');
    const opt = {
      margin:       0.5,
      filename:     'Crop-Recommendation-Report.pdf',
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2, scrollY: 0 },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    
    const originalScroll = window.scrollY;
    window.scrollTo(0, 0);
    
    html2pdf().set(opt).from(element).save().then(() => {
        window.scrollTo(0, originalScroll);
    });
}

// ===============================
// FORM SUBMISSION
// ===============================

const HISTORY_KEY = 'cropPredictionHistory';
let lastBatchResults = null;

document.addEventListener('DOMContentLoaded', function () {

    const form = document.getElementById('recommendationForm');
    const batchForm = document.getElementById('batchForm');
    const presetSelect = document.getElementById('preset-select');

    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    if (batchForm) {
        batchForm.addEventListener('submit', handleBatchUpload);
    }
    if (presetSelect) {
        presetSelect.addEventListener('change', handlePresetChange);
    }
    const btnFetchWeather = document.getElementById('btn-fetch-weather');
    if (btnFetchWeather) {
        btnFetchWeather.addEventListener('click', fetchLiveWeather);
    }

    renderPredictionHistory();
    
    const catalogueGrid = document.getElementById('crops-catalogue-grid');
    if (catalogueGrid) {
        loadCropsCatalogue();
    }
    
    console.log("Smart Farming Frontend Loaded");
});


async function handleFormSubmit(event) {

    event.preventDefault();

    const spinner = document.getElementById('loadingSpinner');
    const resultsContent = document.getElementById('resultsContent');
    const resultsSection = document.getElementById('results');
    const errorMessage = document.getElementById('errorMessage');

    spinner.style.display = "flex";
    resultsContent.style.display = "none";
    resultsSection.style.display = "block";
    errorMessage.style.display = "none";

    resultsSection.scrollIntoView({ behavior: "smooth" });

    try {

        const formData = new FormData(event.target);
        const data = Object.fromEntries(formData);

        Object.keys(data).forEach(key => {
            if (key !== 'model') {
                data[key] = parseFloat(data[key]);
            }
        });

        const response = await fetch("/api/predict", {

            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {

            const err = await response.json();
            throw new Error(err.error || "Prediction failed");
        }

        const result = await response.json();

        displayResults(result);

    }
    catch (error) {

        console.error(error);
        showError(error.message);
    }
    finally {
        spinner.style.display = "none";
        
        // Ensure PDF button is there
        if (!document.getElementById('btn-download-pdf')) {
            const btn = document.createElement('button');
            btn.id = 'btn-download-pdf';
            btn.className = 'btn btn-secondary';
            btn.style.marginTop = '20px';
            btn.innerHTML = '<i class="fas fa-file-pdf"></i> Download PDF Report';
            btn.onclick = downloadPDF;
            document.getElementById('resultsContent').appendChild(btn);
        }
    }
}

async function handleBatchUpload(event) {
    event.preventDefault();

    const batchForm = document.getElementById('batchForm');
    const batchFile = document.getElementById('batch-file');
    const batchSpinner = document.getElementById('batchLoadingSpinner');
    const batchResultsSection = document.getElementById('batchResultsSection');
    const batchErrorMessage = document.getElementById('batchErrorMessage');
    
    batchSpinner.style.display = 'flex';
    batchResultsSection.style.display = 'none';
    batchErrorMessage.style.display = 'none';

    try {
        const formData = new FormData(batchForm);
        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            formData.set('model', modelSelect.value);
        }

        const response = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Batch prediction failed');
        }

        displayBatchResults(result);

    } catch (error) {
        console.error(error);
        if (batchErrorMessage) {
            batchErrorMessage.textContent = error.message;
            batchErrorMessage.style.display = 'block';
        }
    } finally {
        batchSpinner.style.display = 'none';
    }
}

function displayBatchResults(result) {
    lastBatchResults = result.results || [];
    const tbody = document.getElementById('batch-results-body');
    const batchResultsSection = document.getElementById('batchResultsSection');

    if (!tbody || !batchResultsSection) return;

    tbody.innerHTML = '';

    lastBatchResults.forEach(item => {
        const row = document.createElement('tr');
        const inputs = item.input_values || {};
        const inputSummary = `N:${inputs.N}, P:${inputs.P}, K:${inputs.K}, T:${inputs.temperature}, H:${inputs.humidity}, pH:${inputs.ph}, R:${inputs.rainfall}`;
        row.innerHTML = `
            <td>${item.row}</td>
            <td>${item.recommended_crop}</td>
            <td>${item.crop_details.best_season || 'N/A'}</td>
            <td>${item.crop_details.soil_type || 'N/A'}</td>
            <td>${item.crop_details.water_requirement || 'N/A'}</td>
            <td>${item.crop_details.spacing || 'N/A'}</td>
            <td>${inputSummary}</td>
        `;
        tbody.appendChild(row);
    });

    batchResultsSection.style.display = 'block';
}

function downloadBatchCSV() {
    if (!lastBatchResults || lastBatchResults.length === 0) {
        showToast('No batch results available to download.', 'error');
        return;
    }

    const headers = ['Row', 'Crop', 'Best Season', 'Soil Type', 'Water Requirement', 'Spacing', 'Notes', 'N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'];
    const rows = lastBatchResults.map(item => {
        const inputs = item.input_values || {};
        return [
            item.row,
            item.recommended_crop,
            item.crop_details.best_season || '',
            item.crop_details.soil_type || '',
            item.crop_details.water_requirement || '',
            item.crop_details.spacing || '',
            item.crop_details.notes || '',
            inputs.N,
            inputs.P,
            inputs.K,
            inputs.temperature,
            inputs.humidity,
            inputs.ph,
            inputs.rainfall
        ];
    });

    const csvContent = [headers.join(','), ...rows.map(row => row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'batch_prediction_results.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

const quickPresets = {
    highRain: {
        N: 90,
        P: 60,
        K: 60,
        ph: 6.5,
        temperature: 28,
        humidity: 85,
        rainfall: 220
    },
    drySeason: {
        N: 70,
        P: 40,
        K: 50,
        ph: 6.8,
        temperature: 32,
        humidity: 40,
        rainfall: 20
    },
    balancedSoil: {
        N: 60,
        P: 55,
        K: 60,
        ph: 6.8,
        temperature: 25,
        humidity: 60,
        rainfall: 100
    }
};

function handlePresetChange(event) {
    const presetKey = event.target.value;
    if (!presetKey) {
        return;
    }

    const values = quickPresets[presetKey];
    if (!values) {
        return;
    }

    Object.entries(values).forEach(([key, value]) => {
        let inputId = key.toLowerCase();
        if (inputId === 'n') inputId = 'nitrogen';
        else if (inputId === 'p') inputId = 'phosphorus';
        else if (inputId === 'k') inputId = 'potassium';
        
        const input = document.getElementById(inputId);
        if (input) {
            input.value = value;
        }
    });
}

function downloadHistoryCSV() {
    const history = loadPredictionHistory();
    if (!history || history.length === 0) {
        showToast('No history available to export.', 'error');
        return;
    }

    const headers = ['Date', 'Model', 'Crop', 'Season', 'Soil', 'Water', 'Spacing', 'N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall'];
    const rows = history.map(entry => {
        const inputs = entry.inputs || {};
        return [
            entry.date,
            entry.model,
            entry.crop,
            entry.season,
            entry.soil,
            entry.water,
            entry.spacing,
            inputs.N || '',
            inputs.P || '',
            inputs.K || '',
            inputs.temperature || '',
            inputs.humidity || '',
            inputs.ph || '',
            inputs.rainfall || ''
        ];
    });

    const csvContent = [headers.join(','), ...rows.map(row => row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(',')).join('\n')].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'prediction_history.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}


// ===============================
// DISPLAY RESULTS
// ===============================

function displayResults(result) {

    const resultsContent = document.getElementById('resultsContent');
    const errorMessage = document.getElementById('errorMessage');

    try {

        const dt = result.all_models.dt;
        const rf = result.all_models.rf;
        const knn = result.all_models.knn;
        const svm = result.all_models.svm;

        const input = result.input_summary;

        // ===============================
        // MODEL PREDICTIONS
        // ===============================

        document.getElementById("dt-crop").textContent = dt.predicted_crop;
        displayAlternatives("dt-alternatives", dt.alternatives);

        document.getElementById("rf-crop").textContent = rf.predicted_crop;
        displayAlternatives("rf-alternatives", rf.alternatives);

        document.getElementById("knn-crop").textContent = knn.predicted_crop;
        displayAlternatives("knn-alternatives", knn.alternatives);

        document.getElementById("svm-crop").textContent = svm.predicted_crop;
        displayAlternatives("svm-alternatives", svm.alternatives);

        const selectedModelNameEl = document.getElementById('selected-model-name');
        const selectedModelDescriptionEl = document.getElementById('selected-model-description');
        const selectedModelBestForEl = document.getElementById('selected-model-bestfor');
        const detailCropNameEl = document.getElementById('detail-crop-name');
        const detailBestSeasonEl = document.getElementById('detail-best-season');
        const detailSoilTypeEl = document.getElementById('detail-soil-type');
        const detailWaterRequirementEl = document.getElementById('detail-water-requirement');
        const detailSpacingEl = document.getElementById('detail-spacing');
        const detailNotesEl = document.getElementById('detail-notes');

        const cropDetails = result.prediction.crop_details || {};
        
        const finalCrop = cropDetails.crop || result.prediction.recommended_crop;
        detailCropNameEl.innerHTML = `<i class="fas fa-${getCropIcon(finalCrop)}" style="margin-right: 8px; color: #4caf50;"></i> ${finalCrop}`;

        selectedModelNameEl.textContent = result.selected_model_name;
        selectedModelDescriptionEl.textContent = result.all_models[result.selected_model].model_info.description;
        selectedModelBestForEl.textContent = result.all_models[result.selected_model].model_info.best_for;
        detailCropNameEl.textContent = cropDetails.crop || result.prediction.recommended_crop;
        detailBestSeasonEl.textContent = cropDetails.best_season || 'N/A';
        detailSoilTypeEl.textContent = cropDetails.soil_type || 'N/A';
        detailWaterRequirementEl.textContent = cropDetails.water_requirement || 'N/A';
        detailSpacingEl.textContent = cropDetails.spacing || 'N/A';
        detailNotesEl.textContent = cropDetails.notes || 'No additional crop details available.';

        document.getElementById('selected-model-confidence').textContent = result.prediction.confidence.toFixed(1) + '%';
        const consensus = getConsensusSummary(result.all_models);
        document.getElementById('consensus-result').textContent = consensus.message;
        document.getElementById('consensus-count').textContent = consensus.agreement;

        savePredictionHistory({
            date: new Date().toISOString(),
            model: result.selected_model_name,
            crop: detailCropNameEl.textContent,
            season: detailBestSeasonEl.textContent,
            soil: detailSoilTypeEl.textContent,
            water: detailWaterRequirementEl.textContent,
            spacing: detailSpacingEl.textContent,
            inputs: input
        });

        // ===============================
        // FEATURE IMPORTANCE
        // ===============================

        const importance = result.prediction.feature_importance;

        if (importance) {

            displayImportance("dt-importance", importance);
            displayImportance("rf-importance", importance);
        }

        // KNN and SVM message
        document.getElementById("knn-importance").innerHTML =
            "<p>KNN is distance-based and does not calculate feature importance.</p>";

        document.getElementById("svm-importance").innerHTML =
            "<p>SVM does not provide direct feature importance.</p>";


        // ===============================
        // INPUT SUMMARY
        // ===============================

        document.getElementById('sum-N').textContent = input.N + " kg/ha";
        document.getElementById('sum-P').textContent = input.P + " kg/ha";
        document.getElementById('sum-K').textContent = input.K + " kg/ha";
        document.getElementById('sum-temp').textContent = input.temperature + "°C";
        document.getElementById('sum-hum').textContent = input.humidity + "%";
        document.getElementById('sum-ph').textContent = input.ph;
        document.getElementById('sum-rain').textContent = input.rainfall + " mm";

        // ===============================
        // CROP RANGE FIT (XAI)
        // ===============================
        const cropRangesCard = document.getElementById('crop-ranges-card');
        const cropRangesContainer = document.getElementById('crop-ranges-container');
        if (cropRangesCard && cropRangesContainer && result.prediction.crop_ranges) {
            displayCropRanges(cropRangesContainer, result.prediction.crop_ranges, input);
            cropRangesCard.style.display = "block";
        } else if (cropRangesCard) {
            cropRangesCard.style.display = "none";
        }

        resultsContent.style.display = "block";
        errorMessage.style.display = "none";

    }
    catch (error) {

        console.error("Display error:", error);
        showError("Error displaying results");
    }
}


// ===============================
// FEATURE IMPORTANCE DISPLAY
// ===============================

function displayImportance(containerId, importance) {

    const container = document.getElementById(containerId);

    if (!container) return;

    container.innerHTML = "";

    const maxScore = Math.max(...importance.scores);

    importance.features.forEach((feature, index) => {

        const score = importance.scores[index];
        const percentage = (score / maxScore) * 100;

        const bar = document.createElement("div");
        bar.className = "importance-bar";

        const label = document.createElement("div");
        label.className = "feature-name";
        label.textContent = feature;

        const barContainer = document.createElement("div");
        barContainer.className = "bar-container";

        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = percentage + "%";
        fill.textContent = score.toFixed(3);

        barContainer.appendChild(fill);

        bar.appendChild(label);
        bar.appendChild(barContainer);

        container.appendChild(bar);
    });
}


// ===============================
// ALTERNATIVE CROPS
// ===============================

function getConsensusSummary(allModels) {
    const votes = {};
    Object.values(allModels).forEach(model => {
        const crop = model.predicted_crop;
        votes[crop] = (votes[crop] || 0) + 1;
    });

    const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
    const topCrop = sorted[0] ? sorted[0][0] : 'Unknown';
    const agreement = sorted[0] ? sorted[0][1] : 0;
    const totalModels = Object.keys(allModels).length;
    let message = '';

    if (agreement === totalModels) {
        message = `All ${totalModels} models agree on ${topCrop}.`;
    } else if (agreement >= Math.ceil(totalModels / 2)) {
        message = `${topCrop} is the consensus recommendation from ${agreement} of ${totalModels} models.`;
    } else {
        const secondCrop = sorted[1] ? sorted[1][0] : null;
        message = `No single consensus; top picks are ${topCrop}${secondCrop ? ' and ' + secondCrop : ''}.`;
    }

    return {
        message,
        agreement: `${agreement}/${totalModels}`
    };
}

function displayAlternatives(containerId, alternatives) {

    const container = document.getElementById(containerId);

    if (!container) return;

    container.innerHTML = "";

    alternatives.forEach((alt, index) => {

        const item = document.createElement("div");

        item.innerHTML =
            `<span>${index + 1}. ${alt.crop}</span>
             <span>${alt.confidence.toFixed(1)}%</span>`;

        container.appendChild(item);
    });
}

function loadPredictionHistory() {
    try {
        const raw = localStorage.getItem(HISTORY_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (error) {
        console.error('Failed to load history:', error);
        return [];
    }
}

function savePredictionHistory(entry) {
    const history = loadPredictionHistory();
    history.unshift(entry);
    const trimmed = history.slice(0, 10);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
    renderPredictionHistory();
}

function clearPredictionHistory() {
    localStorage.removeItem(HISTORY_KEY);
    renderPredictionHistory();
}

function renderPredictionHistory() {
    const history = loadPredictionHistory();
    const tbody = document.getElementById('history-body');
    const historyEmpty = document.getElementById('historyEmpty');
    const historySection = document.getElementById('history-section');

    if (!tbody || !historyEmpty || !historySection) return;

    tbody.innerHTML = '';

    if (history.length === 0) {
        historyEmpty.style.display = 'block';
        historySection.style.display = 'block';
        return;
    }

    historyEmpty.style.display = 'none';
    historySection.style.display = 'block';

    history.forEach(entry => {
        const row = document.createElement('tr');
        const date = new Date(entry.date).toLocaleString();
        const inputs = entry.inputs || {};
        const inputSummary = `N:${inputs.N}, P:${inputs.P}, K:${inputs.K}, T:${inputs.temperature}, H:${inputs.humidity}, pH:${inputs.ph}, R:${inputs.rainfall}`;

        row.innerHTML = `
            <td>${date}</td>
            <td>${entry.model}</td>
            <td>${entry.crop}</td>
            <td>${entry.season}</td>
            <td>${entry.soil}</td>
            <td>${entry.water}</td>
            <td>${entry.spacing}</td>
            <td>${inputSummary}</td>
        `;

        tbody.appendChild(row);
    });
}


// ===============================
// ERROR DISPLAY
// ===============================

function showError(message) {

    const errorMessage = document.getElementById("errorMessage");
    const resultsContent = document.getElementById("resultsContent");

    if (errorMessage) {

        errorMessage.textContent = message;
        errorMessage.style.display = "block";
        resultsContent.style.display = "none";
    }
}

function resetForm() {
    const form = document.getElementById('recommendationForm');
    const resultsSection = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');

    if (form) {
        form.reset();
    }
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
    if (resultsContent) {
        resultsContent.style.display = 'none';
    }
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }
    if (errorMessage) {
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';
    }
}

async function loadCropsCatalogue() {
    const catalogueGrid = document.getElementById('crops-catalogue-grid');
    if (!catalogueGrid) return;
    
    try {
        const response = await fetch('/api/crops');
        if (!response.ok) throw new Error('Failed to load crop catalogue');
        const data = await response.json();
        
        if (data.success) {
            renderCatalogue(catalogueGrid, data.crops);
        }
    } catch (error) {
        console.error(error);
        catalogueGrid.innerHTML = `<p class="error-msg">Error loading crops catalogue: ${error.message}</p>`;
    }
}

function renderCatalogue(grid, crops) {
    grid.innerHTML = '';
    
    crops.forEach(crop => {
        const card = document.createElement('div');
        card.className = 'crop-card';
        
        let icon = 'fa-seedling';
        const name = crop.crop.toLowerCase();
        if (['rice', 'maize', 'blackgram', 'chickpea', 'lentil', 'pigeonpeas', 'mothbeans', 'mungbean'].some(c => name.includes(c))) {
            icon = 'fa-wheat-awn';
        } else if (['apple', 'banana', 'grapes', 'mango', 'orange', 'papaya', 'pomegranate', 'watermelon', 'muskmelon'].some(c => name.includes(c))) {
            icon = 'fa-apple-whole';
        } else if (name.includes('coconut')) {
            icon = 'fa-tree';
        } else if (name.includes('cotton') || name.includes('jute')) {
            icon = 'fa-shirt';
        } else if (name.includes('coffee')) {
            icon = 'fa-mug-hot';
        }
        
        card.innerHTML = `
            <div class="crop-card-header">
                <i class="fas ${icon}"></i>
                <h3>${crop.crop}</h3>
            </div>
            <div class="crop-card-body">
                <p><strong>Season:</strong> ${crop.best_season}</p>
                <p><strong>Soil:</strong> ${crop.soil_type}</p>
                <p><strong>Water:</strong> ${crop.water_requirement}</p>
                <p><strong>Spacing:</strong> ${crop.spacing}</p>
                <p class="crop-card-notes">${crop.notes}</p>
            </div>
        `;
        grid.appendChild(card);
    });
}

async function fetchLiveWeather() {
    const cityInput = document.getElementById('city-input');
    if (!cityInput) return;
    
    const cityName = cityInput.value.trim();
    if (!cityName) {
        showToast('Please enter a city name.', 'error');
        return;
    }
    
    const btn = document.getElementById('btn-fetch-weather');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Geocoding...';
    btn.disabled = true;
    
    try {
        // Step 1: Geocode place name to get lat/lon via our backend (uses OpenStreetMap Nominatim)
        const geocodeUrl = `/api/geocode?name=${encodeURIComponent(cityName)}`;
        const geocodeResponse = await fetch(geocodeUrl);
        if (!geocodeResponse.ok) {
            const err = await geocodeResponse.json();
            throw new Error(err.error || 'Location geocoding failed');
        }
        const geocodeData = await geocodeResponse.json();
        
        const lat = geocodeData.latitude;
        const lon = geocodeData.longitude;
        const formattedName = geocodeData.name;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Fetching weather...';
        
        // Step 2: Fetch weather for coordinates
        const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,rain&timezone=auto`;
        const weatherResponse = await fetch(weatherUrl);
        if (!weatherResponse.ok) throw new Error('Weather API request failed');
        const weatherData = await weatherResponse.json();
        
        if (weatherData.current) {
            const tempInput = document.getElementById('temperature');
            const humInput = document.getElementById('humidity');
            const rainInput = document.getElementById('rainfall');
            
            if (tempInput) tempInput.value = weatherData.current.temperature_2m.toFixed(1);
            if (humInput) humInput.value = weatherData.current.relative_humidity_2m.toFixed(1);
            
            // Estimate climatic rainfall based on latitude absolute value
            const absLat = Math.abs(lat);
            let baselineRain = 80; // moderate baseline
            if (absLat < 10) {
                baselineRain = 160; // Equatorial wet zone
            } else if (absLat < 23.5) {
                baselineRain = 120; // Tropical monsoonal zone
            } else if (absLat < 35) {
                baselineRain = 70;  // Subtropical dry/moderate zone
            } else {
                baselineRain = 50;  // Temperate zone
            }
            
            if (weatherData.current.rain > 0) {
                baselineRain += weatherData.current.rain * 10;
            }
            
            const finalRain = Math.min(300, Math.max(0, baselineRain));
            if (rainInput) rainInput.value = finalRain.toFixed(1);
            
            showToast(`Live IoT sensor data loaded successfully for:\n${formattedName}\n\nTemperature: ${weatherData.current.temperature_2m}°C\nHumidity: ${weatherData.current.relative_humidity_2m}%\nRainfall (Simulated): ${finalRain.toFixed(1)} mm`, 'error');
        }
    } catch (error) {
        console.error(error);
        showToast(`Sensor retrieval failed: ${error.message}`, 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function displayCropRanges(container, ranges, input) {
    container.innerHTML = '';
    
    const features = [
        { key: 'N', name: 'Nitrogen (N)', unit: 'kg/ha' },
        { key: 'P', name: 'Phosphorus (P)', unit: 'kg/ha' },
        { key: 'K', name: 'Potassium (K)', unit: 'kg/ha' },
        { key: 'temperature', name: 'Temperature', unit: '°C' },
        { key: 'humidity', name: 'Humidity', unit: '%' },
        { key: 'ph', name: 'pH Level', unit: '' },
        { key: 'rainfall', name: 'Rainfall', unit: 'mm' }
    ];
    
    features.forEach(f => {
        const val = input[f.key];
        const range = ranges[f.key];
        if (!range) return;
        
        const isOptimal = val >= range.min && val <= range.max;
        const statusText = isOptimal ? 'Optimal Fit' : 'Suboptimal';
        const statusClass = isOptimal ? 'badge-optimal' : 'badge-suboptimal';
        
        const maxRangeVal = Math.max(range.max * 1.3, val * 1.1, 1.0);
        const minPercent = (range.min / maxRangeVal) * 100;
        const maxPercent = (range.max / maxRangeVal) * 100;
        const currentPercent = (val / maxRangeVal) * 100;
        
        const row = document.createElement('div');
        row.className = `range-gauge-row ${isOptimal ? 'optimal' : 'suboptimal'}`;
        
        row.innerHTML = `
            <div class="range-gauge-header">
                <span class="range-gauge-name"><strong>${f.name}</strong></span>
                <span class="range-gauge-status-badge ${statusClass}">${statusText}</span>
            </div>
            <div class="range-gauge-track-wrapper">
                <div class="range-gauge-track">
                    <div class="range-gauge-optimal-zone" style="left: ${minPercent}%; width: ${maxPercent - minPercent}%;"></div>
                    <div class="range-gauge-marker" style="left: ${currentPercent}%;">
                        <div class="marker-pin"></div>
                        <span class="marker-value-label">${val.toFixed(1)}${f.unit}</span>
                    </div>
                </div>
            </div>
            <div class="range-gauge-footer">
                <span>Optimal Range: <strong>${range.min.toFixed(1)} - ${range.max.toFixed(1)}</strong> ${f.unit}</span>
                <span>Dataset Mean: <strong>${range.mean.toFixed(1)}</strong> ${f.unit}</span>
            </div>
        `;
        container.appendChild(row);
    });
}
