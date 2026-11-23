/**
 * FakeLens Frontend Application Logic
 * Single Page Application router, image drag-and-drop, API communication,
 * multi-dataset domain switching (DeepFakeFace vs ArtiFact), multi-model family selection,
 * conditional two-stage inference display, and academic benchmark visualization.
 */

// Global state
let currentSelectedFile = null;
let currentDataset = 'deepfakeface'; // 'deepfakeface' or 'artifact'
let cachedRegistryData = null;

const MODEL_DESCRIPTIONS = {
    'efficientnet_b0': 'Compound scaled deep convolutional neural network pretrained on ImageNet and fine-tuned for forensic artifact detection (Champion Model).',
    'cnn': 'End-to-end 4-stage convolutional neural network (FakeLensSimpleCNN) processing raw 224×224 RGB pixels directly on GPU.',
    'random_forest': 'Bagging ensemble classifier with balanced class weights operating on 1,798 extracted forensic features.',
    'mlp': 'Multi-Layer Perceptron (3-layer feedforward network: 512-256-128) operating on 1,798 scaled forensic features.',
    'knn': 'K-Nearest Neighbors (K=15) distance-based classifier operating on 1,798 forensic features.'
};

const DATASET_INFO = {
    'deepfakeface': {
        'title': 'DeepFakeFace',
        'badge': 'DeepFakeFace (Faces)',
        'domain': 'Human Facial Portraits',
        'desc': 'Facial Forgery Detection & Attribution • 60K V4 Benchmark • 10 Verified Models',
        'suitability': '<strong>Optimal Domain Suitability:</strong> DeepFakeFace models are specialized for human facial portraits. For general scenes, objects, or animal images, switch to <strong>ArtiFact</strong> domain.',
        'heroTitle': 'Detect Facial Forgeries.<br>Attribute Their Origins.',
        'heroSubtitle': 'Dual-domain AI authenticity analysis and forensic source attribution. Built on DeepFakeFace (facial deepfakes) and ArtiFact (general scene & object manipulations) across 20 verified models.',
        'stat1Val': '91.87%',
        'stat1Meta': 'Val Accuracy (0.9788 ROC-AUC)',
        'stat2Val': '84.73%',
        'stat2Meta': 'Val Accuracy (0.9752 ROC-AUC)',
        'binarySubheading': 'Evaluation Split: DeepFakeFace V4 6,000 Validation Images',
        'multiclassSubheading': 'Classes: wiki (REAL), text2img (Synthetic), inpainting (Edited), insight (FaceSwap)'
    },
    'artifact': {
        'title': 'ArtiFact',
        'badge': 'ArtiFact (General)',
        'domain': 'General Objects & Scenes',
        'desc': 'General Image Forgery Detection & Attribution • 55K Benchmark • 10 Verified Models',
        'suitability': '<strong>Optimal Domain Suitability:</strong> ArtiFact models are trained on general natural imagery, animals, architecture, and objects across diverse generative models (GAN, Diffusion, Transformer).',
        'heroTitle': 'Detect General Image Forgeries.<br>Attribute Generative Engines.',
        'heroSubtitle': 'Broad-domain AI authenticity analysis and forensic source attribution across natural objects, scenes, and complex generative artifacts (StyleGAN, Diffusion, Inpainting, Transformers).',
        'stat1Val': '86.60%',
        'stat1Meta': 'Val Accuracy (0.9416 ROC-AUC)',
        'stat2Val': '82.55%',
        'stat2Meta': 'Val Accuracy (0.9634 ROC-AUC)',
        'binarySubheading': 'Evaluation Split: ArtiFact 55K Benchmark Validation Images',
        'multiclassSubheading': 'Classes: REAL, GAN, DIFFUSION, INPAINTING, TRANSFORMER'
    },
    'cifake': {
        'title': 'CIFAKE',
        'badge': 'CIFAKE (CIFAR-10 AI)',
        'domain': '32×32 RGB Natural vs Diffusion',
        'desc': 'CIFAR-10 Real vs AI Generated • 119K Decontaminated Benchmark • 4 Verified Models',
        'suitability': '<strong>Optimal Domain Suitability:</strong> CIFAKE models are specialized for native 32×32 pixel real vs Stable Diffusion synthetic images (Binary classification: Custom CNN, MLP, Random Forest, KNN).',
        'heroTitle': 'Detect CIFAR-10 AI Imagery.<br>Distinguish Natural from Diffusion.',
        'heroSubtitle': 'Dual-class authenticity screening comparing authentic CIFAR-10 photographs against Stable Diffusion v1.4 synthetic imagery across deep and classical models.',
        'stat1Val': '97.18%',
        'stat1Meta': 'Val Accuracy (0.9965 ROC-AUC)',
        'stat2Val': 'N/A',
        'stat2Meta': 'Binary Only (No Multiclass)',
        'binarySubheading': 'Evaluation Split: CIFAKE 19,619 Decontaminated Validation Images',
        'multiclassSubheading': 'Multiclass attribution is not applicable for CIFAKE (Binary Only)'
    }
};

// Bi-directional dataset change from Comparison page dropdown
function onCompDatasetChange(ds) {
    if (ds && ds !== currentDataset) {
        switchGlobalDataset(ds);
    } else {
        filterBenchmarks();
        renderComparisonCharts();
    }
}
function navigateTo(viewId) {
    const views = document.querySelectorAll('.view-section');
    const navItems = document.querySelectorAll('.nav-item');

    views.forEach(section => {
        if (section.id === `view-${viewId}`) {
            section.classList.add('active');
        } else {
            section.classList.remove('active');
        }
    });

    navItems.forEach(item => {
        if (item.getAttribute('data-view') === viewId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================
// 2. Global Dataset Domain Switcher
// ==========================================
function switchGlobalDataset(ds) {
    if (ds !== 'deepfakeface' && ds !== 'artifact' && ds !== 'cifake') return;
    currentDataset = ds;

    // Update segmented buttons
    const btnDff = document.getElementById('btnDsDeepfakeface');
    const btnArt = document.getElementById('btnDsArtifact');
    const btnCif = document.getElementById('btnDsCifake');
    if (btnDff) btnDff.classList.toggle('active', ds === 'deepfakeface');
    if (btnArt) btnArt.classList.toggle('active', ds === 'artifact');
    if (btnCif) btnCif.classList.toggle('active', ds === 'cifake');

    // Update compact select if present
    const compactSelect = document.getElementById('datasetCompactSelect');
    if (compactSelect && compactSelect.value !== ds) {
        compactSelect.value = ds;
    }

    // Update Banner & UI elements
    const info = DATASET_INFO[ds];
    const domainBadgeActive = document.getElementById('domainBadgeActive');
    const domainDescription = document.getElementById('domainDescription');
    const heroTitle = document.getElementById('heroTitle');
    const heroSubtitle = document.getElementById('heroSubtitle');
    const statCard1Val = document.getElementById('statCard1Val');
    const statCard1Meta = document.getElementById('statCard1Meta');
    const statCard2Val = document.getElementById('statCard2Val');
    const statCard2Meta = document.getElementById('statCard2Meta');
    const analysisDomainPillText = document.getElementById('analysisDomainPillText');
    const domainSuitabilityText = document.getElementById('domainSuitabilityText');
    const compDatasetSelect = document.getElementById('compDatasetSelect');

    if (domainBadgeActive) domainBadgeActive.textContent = info.badge;
    if (domainDescription) domainDescription.textContent = info.desc;
    if (heroTitle) heroTitle.innerHTML = info.heroTitle;
    if (heroSubtitle) heroSubtitle.textContent = info.heroSubtitle;
    if (statCard1Val) statCard1Val.textContent = info.stat1Val;
    if (statCard1Meta) statCard1Meta.textContent = info.stat1Meta;
    if (statCard2Val) statCard2Val.textContent = info.stat2Val;
    if (statCard2Meta) statCard2Meta.textContent = info.stat2Meta;
    if (analysisDomainPillText) analysisDomainPillText.textContent = `Active Domain: ${info.badge}`;
    if (domainSuitabilityText) domainSuitabilityText.innerHTML = info.suitability;
    if (compDatasetSelect) compDatasetSelect.value = ds;

    // Update Model Family Dropdown dynamically based on selected dataset
    const modelFamilySelect = document.getElementById('modelFamilySelect');
    if (modelFamilySelect) {
        const prevFam = modelFamilySelect.value;
        if (ds === 'cifake') {
            modelFamilySelect.innerHTML = `
                <option value="cnn" selected>Custom CNN (3-Block Baseline) &bull; Champion Model (97.18%)</option>
                <option value="mlp">MLP (3072-D Deep Feedforward) &bull; Baseline Model (85.82%)</option>
                <option value="random_forest">Random Forest &bull; 300 Trees Regularized (83.71%)</option>
                <option value="knn">KNN (K=5 Cosine Distance, PCA 100-D) &bull; (76.75%)</option>
            `;
            if (['cnn', 'mlp', 'random_forest', 'knn'].includes(prevFam)) {
                modelFamilySelect.value = prevFam;
            } else {
                modelFamilySelect.value = 'cnn';
            }
        } else {
            modelFamilySelect.innerHTML = `
                <option value="efficientnet_b0" selected>EfficientNet-B0 (Transfer Learning) &bull; Champion Model</option>
                <option value="cnn">Custom CNN (FakeLensSimpleCNN) &bull; Convolutional Baseline</option>
                <option value="random_forest">Random Forest &bull; Bagging Ensemble on 1,798 Forensic Features</option>
                <option value="mlp">MLP (Multi-Layer Perceptron) &bull; Deep Feedforward Network</option>
                <option value="knn">KNN (K-Nearest Neighbors) &bull; Distance Classifier (K=15)</option>
            `;
            if (['efficientnet_b0', 'cnn', 'random_forest', 'mlp', 'knn'].includes(prevFam)) {
                modelFamilySelect.value = prevFam;
            } else {
                modelFamilySelect.value = 'efficientnet_b0';
            }
        }
        onModelFamilyChange();
    }

    // Reset previous inference view to reflect domain switch
    const resultsWrapper = document.getElementById('resultsWrapper');
    const disagreementBanner = document.getElementById('disagreementBanner');
    if (resultsWrapper) resultsWrapper.classList.remove('active');
    if (disagreementBanner) disagreementBanner.style.display = 'none';

    // Re-render comparison benchmarks for selected dataset
    filterBenchmarks();
    renderComparisonCharts();
}

// ==========================================
// 3. Health & Registry Initializer
// ==========================================
async function checkEngineHealth() {
    const pill = document.getElementById('engineStatusPill');
    const statusText = document.getElementById('engineStatusText');
    const statusDot = pill ? pill.querySelector('.status-dot') : null;

    try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error('API degraded');
        const data = await res.json();

        if (data.engine_ready) {
            const count = data.total_registered_models || 24;
            if (data.device && data.device.toLowerCase().includes('cuda')) {
                statusText.textContent = `${count} Models Ready (CUDA)`;
                if (statusDot) statusDot.style.background = '#10b981';
            } else {
                statusText.textContent = `${count} Models Ready (CPU)`;
                if (statusDot) statusDot.style.background = '#06b6d4';
            }
        } else {
            statusText.textContent = 'Models Unavailable';
            if (statusDot) statusDot.style.background = '#ef4444';
        }
    } catch (err) {
        statusText.textContent = 'Server Offline';
        if (statusDot) statusDot.style.background = '#ef4444';
    }
}

// ==========================================
// 4. Model Selector Handler
// ==========================================
function onModelFamilyChange() {
    const select = document.getElementById('modelFamilySelect');
    const hint = document.getElementById('modelArchHint');
    const binaryModelTag = document.getElementById('binaryModelTag');
    const multiclassModelTag = document.getElementById('multiclassModelTag');
    const resultsWrapper = document.getElementById('resultsWrapper');
    const disagreementBanner = document.getElementById('disagreementBanner');

    if (!select) return;
    const selectedFamily = select.value;

    // Update architecture hint
    if (hint && MODEL_DESCRIPTIONS[selectedFamily]) {
        hint.textContent = MODEL_DESCRIPTIONS[selectedFamily];
    }

    // Update badges
    const familyLabels = {
        'efficientnet_b0': 'EfficientNet-B0',
        'cnn': 'Custom CNN',
        'random_forest': 'Random Forest',
        'mlp': 'MLP',
        'knn': 'KNN'
    };
    const displayFam = familyLabels[selectedFamily] || selectedFamily;
    if (binaryModelTag) binaryModelTag.textContent = `Model: ${displayFam}`;
    if (multiclassModelTag) multiclassModelTag.textContent = `Model: ${displayFam}`;

    // Reset previous inference display
    if (resultsWrapper) resultsWrapper.classList.remove('active');
    if (disagreementBanner) disagreementBanner.style.display = 'none';

    // Highlight button to re-run
    const btnAnalyze = document.getElementById('btnAnalyze');
    if (btnAnalyze && currentSelectedFile) {
        btnAnalyze.disabled = false;
    }
}

// ==========================================
// 5. Image Selection & Validation
// ==========================================
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function handleImageFile(file) {
    if (!file) return;

    const validMimes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validMimes.includes(file.type.toLowerCase())) {
        alert('Invalid file format. Please upload a JPEG, PNG, or WEBP image.');
        return;
    }

    if (file.size > 25 * 1024 * 1024) {
        alert('File size exceeds the 25 MB safety limit.');
        return;
    }

    currentSelectedFile = file;

    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileMeta = document.getElementById('previewFileMeta');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const resultsWrapper = document.getElementById('resultsWrapper');
    const disagreementBanner = document.getElementById('disagreementBanner');

    // Hide old results and disagreement banners on new selection
    if (resultsWrapper) resultsWrapper.classList.remove('active');
    if (disagreementBanner) disagreementBanner.style.display = 'none';

    // Populate preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewFileName.textContent = file.name;

        // Measure natural dimensions
        const tmpImg = new Image();
        tmpImg.onload = () => {
            previewFileMeta.textContent = `Size: ${formatFileSize(file.size)} | Dimensions: ${tmpImg.naturalWidth} × ${tmpImg.naturalHeight} px`;
        };
        tmpImg.src = e.target.result;

        previewArea.style.display = 'block';
        if (btnAnalyze) btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentSelectedFile = null;
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    const resultsWrapper = document.getElementById('resultsWrapper');
    const disagreementBanner = document.getElementById('disagreementBanner');
    const multiclassCard = document.getElementById('multiclassCard');
    const multiclassList = document.getElementById('multiclassList');
    const realAttributionNote = document.getElementById('realAttributionNote');
    const resultsGrid = document.getElementById('resultsGrid');
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnAnalyzeText = document.getElementById('btnAnalyzeText');

    if (fileInput) fileInput.value = '';
    if (previewArea) previewArea.style.display = 'none';
    if (previewImage) previewImage.src = '';
    if (resultsWrapper) resultsWrapper.classList.remove('active');
    if (disagreementBanner) disagreementBanner.style.display = 'none';
    if (multiclassList) multiclassList.innerHTML = '';
    if (multiclassCard) multiclassCard.style.display = 'none';
    if (realAttributionNote) realAttributionNote.style.display = 'none';
    if (resultsGrid) resultsGrid.classList.remove('single-card');
    if (btnAnalyze) btnAnalyze.disabled = true;
    if (btnAnalyzeText) btnAnalyzeText.textContent = 'Run Forensic Analysis';
}

// ==========================================
// 6. Real Model Inference Execution
// ==========================================
async function executeAnalysis() {
    if (!currentSelectedFile) {
        alert('Please select or drop an image first.');
        return;
    }

    const modelSelect = document.getElementById('modelFamilySelect');
    const selectedFamily = modelSelect ? modelSelect.value : 'efficientnet_b0';

    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnAnalyzeText = document.getElementById('btnAnalyzeText');
    const resultsWrapper = document.getElementById('resultsWrapper');
    const disagreementBanner = document.getElementById('disagreementBanner');
    const multiclassCard = document.getElementById('multiclassCard');
    const multiclassList = document.getElementById('multiclassList');
    const realAttributionNote = document.getElementById('realAttributionNote');
    const resultsGrid = document.getElementById('resultsGrid');

    // UI Loading state
    btnAnalyze.disabled = true;
    btnAnalyzeText.textContent = `Analyzing with ${selectedFamily}...`;
    if (resultsWrapper) resultsWrapper.classList.remove('active');
    if (disagreementBanner) disagreementBanner.style.display = 'none';
    if (multiclassList) multiclassList.innerHTML = '';
    if (multiclassCard) multiclassCard.style.display = 'none';
    if (realAttributionNote) realAttributionNote.style.display = 'none';
    if (resultsGrid) resultsGrid.classList.remove('single-card');

    try {
        const formData = new FormData();
        formData.append('image', currentSelectedFile);
        formData.append('dataset', currentDataset);
        formData.append('model_family', selectedFamily);

        const resp = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson.error || `Server responded with status ${resp.status}`);
        }

        const data = await resp.json();
        renderAnalysisResults(data, selectedFamily);

    } catch (err) {
        alert(`Analysis failed: ${err.message}`);
    } finally {
        btnAnalyze.disabled = false;
        btnAnalyzeText.textContent = 'Run Forensic Analysis';
    }
}

function renderAnalysisResults(data, selectedFamily) {
    const resultsWrapper = document.getElementById('resultsWrapper');
    const resultsGrid = document.getElementById('resultsGrid');
    const binaryBadge = document.getElementById('binaryDecisionBadge');
    const binaryRealPct = document.getElementById('binaryRealPct');
    const binaryFakePct = document.getElementById('binaryFakePct');
    const binaryRealBar = document.getElementById('binaryRealBar');
    const binaryFakeBar = document.getElementById('binaryFakeBar');
    const binaryExplanation = document.getElementById('binaryExplanation');
    const realAttributionNote = document.getElementById('realAttributionNote');
    const binaryModelTag = document.getElementById('binaryModelTag');
    const multiclassModelTag = document.getElementById('multiclassModelTag');

    const multiclassCard = document.getElementById('multiclassCard');
    const multiclassList = document.getElementById('multiclassList');
    const multiclassExplanation = document.getElementById('multiclassExplanation');
    const inferenceMeta = document.getElementById('inferenceMeta');
    const disagreementBanner = document.getElementById('disagreementBanner');
    const disagreementText = document.getElementById('disagreementText');

    const bin = data.binary_result;
    const multi = data.multiclass_result;

    if (binaryModelTag) binaryModelTag.textContent = `${data.model_family} (${data.dataset.toUpperCase()})`;
    if (multiclassModelTag) multiclassModelTag.textContent = `${data.model_family} (${data.dataset.toUpperCase()})`;

    // 1. Render Dedicated Binary Result
    const isReal = bin.decision === 'REAL';
    binaryBadge.textContent = `${bin.decision} (${bin.confidence_pct}%)`;
    binaryBadge.className = `decision-badge ${isReal ? 'real' : 'fake'}`;

    const realProb = bin.probabilities.REAL ?? 0.0;
    const fakeProb = bin.probabilities.FAKE ?? 0.0;

    binaryRealPct.textContent = `${realProb.toFixed(1)}%`;
    binaryFakePct.textContent = `${fakeProb.toFixed(1)}%`;
    binaryRealBar.style.width = `${realProb}%`;
    binaryFakeBar.style.width = `${fakeProb}%`;
    binaryExplanation.textContent = bin.explanation;

    // 2. Conditional Two-Stage Display
    if (data.dataset === 'cifake') {
        // CIFAKE is strictly binary
        if (resultsGrid) resultsGrid.classList.add('single-card');
        if (multiclassCard) multiclassCard.style.display = 'none';
        if (multiclassList) multiclassList.innerHTML = '';
        if (realAttributionNote) {
            realAttributionNote.innerHTML = `
                <span class="info-icon">&#9432;</span>
                <span><strong>CIFAKE Binary Domain:</strong> This dataset is focused strictly on binary detection (CIFAR-10 real vs Stable Diffusion v1.4 synthetic). Fine-grained multi-generator attribution is supported on DeepFakeFace and ArtiFact.</span>
            `;
            realAttributionNote.style.display = 'flex';
        }
    } else if (isReal) {
        if (resultsGrid) resultsGrid.classList.add('single-card');
        if (multiclassCard) multiclassCard.style.display = 'none';
        if (multiclassList) multiclassList.innerHTML = '';
        if (realAttributionNote) {
            realAttributionNote.innerHTML = `
                <span class="info-icon">&#9432;</span>
                <span>Source attribution was bypassed because the selected <strong>${data.model_family}</strong> binary classifier determined the image is authentic <strong>REAL</strong>.</span>
            `;
            realAttributionNote.style.display = 'flex';
        }
    } else {
        if (resultsGrid) resultsGrid.classList.remove('single-card');
        if (realAttributionNote) realAttributionNote.style.display = 'none';

        if (multiclassCard) {
            multiclassCard.style.display = 'block';
            multiclassList.innerHTML = '';

            const probs = (multi && multi.probabilities) ? multi.probabilities : [];
            if (probs.length > 0) {
                // Sort probabilities descending
                const sortedProbs = [...probs].sort((a, b) => b.probability_pct - a.probability_pct);
                const topAttribution = sortedProbs[0];

                // Top Attributed Banner
                const topBanner = document.createElement('div');
                topBanner.className = 'top-attributed-banner';
                topBanner.innerHTML = `
                    <div class="top-attributed-info">
                        <span class="top-attributed-tag">Primary Attributed Source</span>
                        <span class="top-attributed-title">${topAttribution.label}</span>
                    </div>
                    <div class="top-attributed-pct">${topAttribution.probability_pct.toFixed(1)}%</div>
                `;
                multiclassList.appendChild(topBanner);

                // Render all classes
                sortedProbs.forEach((item, index) => {
                    const isWinner = index === 0;
                    let barClass = 'fake';
                    const cName = item.name.toLowerCase();
                    if (cName === 'wiki' || cName === 'real') barClass = 'real';
                    else if (cName === 'text2img' || cName === 'gan') barClass = 'synthetic';
                    else if (cName === 'inpainting') barClass = 'inpainting';
                    else if (cName === 'insight' || cName === 'transformer') barClass = 'faceswap';

                    const itemDiv = document.createElement('div');
                    itemDiv.className = `multiclass-item ${isWinner ? 'winner' : ''}`;
                    itemDiv.innerHTML = `
                        <div class="prob-meter-labels">
                            <span style="font-weight: ${isWinner ? '700' : '500'}; color: var(--text-main);">
                                ${item.label} ${isWinner ? '<span class="top-source-tag">Top Attribution</span>' : ''}
                            </span>
                            <span style="font-family: var(--font-mono); font-size: 0.85rem; font-weight: ${isWinner ? '700' : '400'};">${item.probability_pct.toFixed(1)}%</span>
                        </div>
                        <div class="prob-bar-track" style="margin-top: 0.35rem; margin-bottom: 0.4rem;">
                            <div class="prob-bar-fill ${barClass}" style="width: ${item.probability_pct}%;"></div>
                        </div>
                    `;
                    multiclassList.appendChild(itemDiv);
                });
            }

            if (multiclassExplanation) {
                multiclassExplanation.textContent = `Attribution calculated via ${data.model_family} trained on ${data.dataset.toUpperCase()} benchmark (${multi.status || 'Active'}).`;
            }
        }
    }

    // 3. Disagreement Warning
    if (data.disagreement_warning) {
        disagreementText.textContent = data.disagreement_warning;
        disagreementBanner.style.display = 'flex';
    } else {
        disagreementBanner.style.display = 'none';
    }

    // 4. Metadata
    inferenceMeta.textContent = `Domain: ${data.dataset_title} | Model: ${data.model_family} | Latency: ${data.processing_time_ms} ms | Device: ${data.device} | Resolution: ${data.image_meta.original_dimensions}`;

    // Reveal results panel
    resultsWrapper.classList.add('active');
    resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==========================================
// 7. Academic Benchmarks Dashboard
// ==========================================
async function loadBenchmarks() {
    try {
        const res = await fetch('/api/benchmarks');
        if (!res.ok) throw new Error('Failed to fetch benchmarks');
        cachedRegistryData = await res.json();

        filterBenchmarks();
        renderComparisonCharts();
    } catch (err) {
        console.warn('Could not load academic benchmarks:', err);
    }
}

function filterBenchmarks() {
    if (!cachedRegistryData) return;

    const dsSelect = document.getElementById('compDatasetSelect');
    const selectedDs = dsSelect ? dsSelect.value : currentDataset;

    const taskFilter = document.getElementById('compTaskSelect')?.value || 'all';
    const familyFilter = document.getElementById('compFamilySelect')?.value || 'all';

    const binarySection = document.getElementById('binaryTableSection');
    const multiclassSection = document.getElementById('multiclassTableSection');
    const binaryTable = document.getElementById('binaryTableBody');
    const multiclassTable = document.getElementById('multiclassTableBody');

    const binarySubhead = document.getElementById('binaryTableSubheading');
    const multiSubhead = document.getElementById('multiclassTableSubheading');

    if (binarySubhead) binarySubhead.textContent = DATASET_INFO[selectedDs].binarySubheading;
    if (multiSubhead) multiSubhead.textContent = DATASET_INFO[selectedDs].multiclassSubheading;

    // Show/hide sections
    if (binarySection) {
        binarySection.style.display = (taskFilter === 'all' || taskFilter === 'binary') ? 'block' : 'none';
    }
    if (multiclassSection) {
        multiclassSection.style.display = (selectedDs !== 'cifake' && (taskFilter === 'all' || taskFilter === 'multiclass')) ? 'block' : 'none';
    }

    const dsData = cachedRegistryData[selectedDs];
    if (!dsData) return;

    const taskSelect = document.getElementById('compTaskSelect');
    if (taskSelect) {
        if (selectedDs === 'cifake') {
            taskSelect.innerHTML = `<option value="binary" selected>Dedicated Binary (Real vs Fake)</option>`;
        } else {
            taskSelect.innerHTML = `
                <option value="all" selected>All Tasks (Binary & Multiclass)</option>
                <option value="binary">Dedicated Binary (Real vs Fake)</option>
                <option value="multiclass">Source Attribution (Multiclass)</option>
            `;
        }
    }

    // Populate Binary Table
    if (binaryTable && dsData.binary) {
        binaryTable.innerHTML = '';
        Object.keys(dsData.binary).forEach(fam => {
            if (familyFilter !== 'all' && familyFilter !== fam) return;
            const m = dsData.binary[fam];
            const tr = document.createElement('tr');
            if (m.is_champion) tr.className = 'champion-row';
            const f1Str = (m.val_macro_f1 !== null && m.val_macro_f1 !== undefined) ? Number(m.val_macro_f1).toFixed(4) : 'N/A';
            const aucStr = (m.val_roc_auc !== null && m.val_roc_auc !== undefined) ? Number(m.val_roc_auc).toFixed(4) : 'N/A';
            const trSampStr = m.training_samples ? Number(m.training_samples).toLocaleString() : 'N/A';
            tr.innerHTML = `
                <td><strong>${m.family}</strong></td>
                <td>${m.architecture} ${m.is_champion ? '<span class="champ-pill">Champion</span>' : ''}</td>
                <td><strong>${(m.val_accuracy * 100).toFixed(2)}%</strong></td>
                <td>${f1Str}</td>
                <td>${aucStr}</td>
                <td>${trSampStr}</td>
                <td><span class="status-badge-inline">${m.status}</span></td>
            `;
            binaryTable.appendChild(tr);
        });
    }

    // Populate Multiclass Table
    if (multiclassTable && dsData.multiclass) {
        multiclassTable.innerHTML = '';
        Object.keys(dsData.multiclass).forEach(fam => {
            if (familyFilter !== 'all' && familyFilter !== fam) return;
            const m = dsData.multiclass[fam];
            const tr = document.createElement('tr');
            if (m.is_champion) tr.className = 'champion-row';
            tr.innerHTML = `
                <td><strong>${m.family}</strong></td>
                <td>${m.architecture} ${m.is_champion ? '<span class="champ-pill">Champion</span>' : ''}</td>
                <td><strong>${(m.val_accuracy * 100).toFixed(2)}%</strong></td>
                <td>${m.val_macro_f1.toFixed(4)}</td>
                <td>${m.val_roc_auc.toFixed(4)}</td>
                <td>${m.num_classes} Classes (${m.class_order.join(', ')})</td>
                <td><span class="status-badge-inline">${m.status}</span></td>
            `;
            multiclassTable.appendChild(tr);
        });
    }
}

// ==========================================
// 8. Benchmark Charts
// ==========================================
function renderComparisonCharts() {
    if (!cachedRegistryData) return;

    const dsSelect = document.getElementById('compDatasetSelect');
    const selectedDs = dsSelect ? dsSelect.value : currentDataset;
    const dsData = cachedRegistryData[selectedDs];
    if (!dsData) return;

    const binChart = document.getElementById('binaryAccuracyChart');
    const multiChart = document.getElementById('multiclassMacroF1Chart');

    const binChartTitle = document.getElementById('binaryChartTitle');
    const binChartSub = document.getElementById('binaryChartSubtitle');
    const multiChartTitle = document.getElementById('multiclassChartTitle');
    const multiChartSub = document.getElementById('multiclassChartSubtitle');

    if (binChartTitle) binChartTitle.textContent = `${DATASET_INFO[selectedDs].title} Binary Validation Accuracy`;
    if (binChartSub) binChartSub.textContent = DATASET_INFO[selectedDs].binarySubheading;
    if (multiChartTitle) multiChartTitle.textContent = `${DATASET_INFO[selectedDs].title} Multiclass Validation Accuracy`;
    const multiCard = document.getElementById('multiclassChartTitle')?.closest('.chart-card');
    if (multiCard) {
        multiCard.style.display = (selectedDs === 'cifake') ? 'none' : 'block';
    }

    // Gather and sort binary models by accuracy
    const binList = Object.values(dsData.binary || {}).map(m => ({
        name: m.family + (m.is_champion ? ' (Champ)' : ''),
        val: m.val_accuracy * 100,
        isChamp: m.is_champion,
        color: m.is_champion ? '#10b981' : '#38bdf8'
    })).sort((a, b) => b.val - a.val);

    if (binChart) {
        binChart.innerHTML = '';
        binList.forEach(item => {
            const row = document.createElement('div');
            row.className = 'chart-bar-row';
            row.innerHTML = `
                <div class="chart-bar-label">${item.name}</div>
                <div class="chart-bar-track">
                    <div class="chart-bar-fill" style="width: ${item.val}%; background: ${item.color};"></div>
                </div>
                <div class="chart-bar-value">${item.val.toFixed(2)}%</div>
            `;
            binChart.appendChild(row);
        });
    }

    // Gather and sort multiclass models by accuracy
    const multiList = Object.values(dsData.multiclass || {}).map(m => ({
        name: m.family + (m.is_champion ? ' (Champ)' : ''),
        val: m.val_accuracy * 100,
        isChamp: m.is_champion,
        color: m.is_champion ? '#8b5cf6' : '#818cf8'
    })).sort((a, b) => b.val - a.val);

    if (multiChart) {
        multiChart.innerHTML = '';
        multiList.forEach(item => {
            const row = document.createElement('div');
            row.className = 'chart-bar-row';
            row.innerHTML = `
                <div class="chart-bar-label">${item.name}</div>
                <div class="chart-bar-track">
                    <div class="chart-bar-fill" style="width: ${item.val}%; background: ${item.color};"></div>
                </div>
                <div class="chart-bar-value">${item.val.toFixed(2)}%</div>
            `;
            multiChart.appendChild(row);
        });
    }
}

// ==========================================
// 9. DOM Initialization & Event Listeners
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    // 1. Dropzone & File Input
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleImageFile(e.target.files[0]);
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('drag-active');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('drag-active');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleImageFile(e.dataTransfer.files[0]);
            }
        });
    }

    // 2. Initial health check & benchmarks load
    checkEngineHealth();
    loadBenchmarks();
});
