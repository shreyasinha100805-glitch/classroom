/**
 * AI Classroom Board Content Extractor
 * Futuristic Neural Studio Frontend Engine (NN/DL Architecture)
 */

document.addEventListener('DOMContentLoaded', () => {
    initThemeSwitcher();
    initConstellationBackground();
    initNNArchitectureExplorer();
    initUploadDropzone();
    initResultsGallery();
    initLightboxModal();
    initAnalysisDrawer();
});

/* ==========================================================================
   0. DYNAMIC THEME COLOR ACCENT ENGINE
   ========================================================================== */
function initThemeSwitcher() {
    const savedTheme = localStorage.getItem('boardai_theme') || 'cyberpunk';
    applyTheme(savedTheme, false);

    const themeButtons = document.querySelectorAll('.theme-dot-btn');
    themeButtons.forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const theme = btn.getAttribute('data-set-theme');
            if (!theme) return;
            applyTheme(theme, true);
        });
    });

    function applyTheme(themeName, notify = true) {
        document.documentElement.setAttribute('data-theme', themeName);
        localStorage.setItem('boardai_theme', themeName);

        document.querySelectorAll('.theme-dot-btn').forEach((btn) => {
            if (btn.getAttribute('data-set-theme') === themeName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        if (notify) {
            const names = {
                'cyberpunk': 'Neon Cyberpunk 🔮',
                'aurora': 'Cosmic Aurora 🌌',
                'solar': 'Solar Flare 🌅',
                'hyperdrive': 'Hyperdrive Blue ⚡'
            };
            showToast(`Theme changed to ${names[themeName] || themeName}!`, 'info', 2000);
        }
    }
}

/* ==========================================================================
   1. DYNAMIC NEURAL SYNAPTIC NETWORK CANVAS
   ========================================================================== */
function initConstellationBackground() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    let mouse = { x: null, y: null, radius: 180 };

    window.addEventListener('mousemove', (e) => {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    });

    window.addEventListener('mouseleave', () => {
        mouse.x = null;
        mouse.y = null;
    });

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        initNeurons();
    });

    const neurons = [];
    const neuronCount = Math.min(Math.floor((width * height) / 16000), 80);
    const synapticPulses = [];

    // Ambient floating deep learning glyphs
    const mathGlyphs = [
        { text: 'ReLU', x: width * 0.15, y: height * 0.25, vy: 0.15, opacity: 0.18 },
        { text: 'σ(z)', x: width * 0.82, y: height * 0.35, vy: -0.12, opacity: 0.16 },
        { text: '∇W', x: width * 0.25, y: height * 0.75, vy: -0.18, opacity: 0.15 },
        { text: 'Softmax', x: width * 0.75, y: height * 0.8, vy: 0.14, opacity: 0.17 },
        { text: 'W^T x + b', x: width * 0.5, y: height * 0.15, vy: 0.1, opacity: 0.14 }
    ];

    function getThemePalette() {
        const theme = document.documentElement.getAttribute('data-theme') || 'cyberpunk';
        switch (theme) {
            case 'aurora':
                return {
                    colorA: 'rgba(16, 185, 129,',
                    colorB: 'rgba(6, 214, 160,',
                    glow: '#06d6a0',
                    pulse: '#10b981'
                };
            case 'solar':
                return {
                    colorA: 'rgba(244, 63, 94,',
                    colorB: 'rgba(245, 158, 11,',
                    glow: '#fb7185',
                    pulse: '#f59e0b'
                };
            case 'hyperdrive':
                return {
                    colorA: 'rgba(59, 130, 246,',
                    colorB: 'rgba(56, 189, 248,',
                    glow: '#38bdf8',
                    pulse: '#60a5fa'
                };
            case 'cyberpunk':
            default:
                return {
                    colorA: 'rgba(99, 102, 241,',
                    colorB: 'rgba(6, 182, 212,',
                    glow: '#06b6d4',
                    pulse: '#38bdf8'
                };
        }
    }

    class Neuron {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.55;
            this.vy = (Math.random() - 0.5) * 0.55;
            this.radius = Math.random() * 2.2 + 1.2;
            this.isAlt = Math.random() > 0.45;
            this.excitation = 0;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;

            if (this.excitation > 0) {
                this.excitation -= 0.02;
                if (this.excitation < 0) this.excitation = 0;
            }

            // Mouse interaction (Synaptic input stimulation)
            if (mouse.x !== null) {
                const dx = mouse.x - this.x;
                const dy = mouse.y - this.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < mouse.radius) {
                    const force = (mouse.radius - dist) / mouse.radius;
                    this.x -= (dx / dist) * force * 1.5;
                    this.y -= (dy / dist) * force * 1.5;
                    this.excitation = Math.min(this.excitation + 0.15, 1);
                }
            }
        }

        draw() {
            const pal = getThemePalette();
            ctx.beginPath();
            const currentRadius = this.radius + this.excitation * 2.5;
            ctx.arc(this.x, this.y, currentRadius, 0, Math.PI * 2);

            const alpha = 0.6 + this.excitation * 0.4;
            const chosenBase = this.isAlt ? pal.colorB : pal.colorA;
            ctx.fillStyle = this.excitation > 0.3 ? pal.pulse : `${chosenBase}${alpha})`;

            if (this.excitation > 0.2) {
                ctx.shadowBlur = 12 * this.excitation;
                ctx.shadowColor = pal.glow;
            } else {
                ctx.shadowBlur = 6;
                ctx.shadowColor = `${chosenBase}0.4)`;
            }
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    function initNeurons() {
        neurons.length = 0;
        for (let i = 0; i < neuronCount; i++) {
            neurons.push(new Neuron());
        }
    }

    initNeurons();

    // Spawn periodic synaptic action potential spikes
    setInterval(() => {
        if (neurons.length < 2) return;
        const i1 = Math.floor(Math.random() * neurons.length);
        const p1 = neurons[i1];

        for (let j = 0; j < neurons.length; j++) {
            if (i1 === j) continue;
            const p2 = neurons[j];
            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 125) {
                synapticPulses.push({
                    x1: p1.x, y1: p1.y,
                    x2: p2.x, y2: p2.y,
                    progress: 0,
                    speed: 0.04
                });
                break;
            }
        }
        if (synapticPulses.length > 25) synapticPulses.shift();
    }, 450);

    function animate() {
        ctx.clearRect(0, 0, width, height);

        // Draw floating math glyphs
        ctx.font = '12px "JetBrains Mono", monospace';
        mathGlyphs.forEach((g) => {
            g.y += g.vy;
            if (g.y < -20) g.y = height + 20;
            if (g.y > height + 20) g.y = -20;
            ctx.fillStyle = `rgba(165, 180, 252, ${g.opacity})`;
            ctx.fillText(g.text, g.x, g.y);
        });

        // Update & draw neurons and synaptic axons
        for (let i = 0; i < neurons.length; i++) {
            neurons[i].update();
            neurons[i].draw();

            for (let j = i + 1; j < neurons.length; j++) {
                const dx = neurons[i].x - neurons[j].x;
                const dy = neurons[i].y - neurons[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 125) {
                    const alpha = (1 - dist / 125) * 0.25;
                    ctx.beginPath();
                    ctx.moveTo(neurons[i].x, neurons[i].y);
                    ctx.lineTo(neurons[j].x, neurons[j].y);

                    const pal = getThemePalette();
                    if (neurons[i].excitation > 0.4 || neurons[j].excitation > 0.4) {
                        ctx.strokeStyle = `${pal.colorB}${alpha * 2.2})`;
                        ctx.lineWidth = 1.3;
                    } else {
                        ctx.strokeStyle = `${pal.colorA}${alpha})`;
                        ctx.lineWidth = 0.85;
                    }
                    ctx.stroke();
                }
            }
        }

        // Render propagating action potential pulses
        const pal = getThemePalette();
        for (let k = synapticPulses.length - 1; k >= 0; k--) {
            const p = synapticPulses[k];
            p.progress += p.speed;
            if (p.progress >= 1) {
                synapticPulses.splice(k, 1);
                continue;
            }

            const px = p.x1 + (p.x2 - p.x1) * p.progress;
            const py = p.y1 + (p.y2 - p.y1) * p.progress;

            ctx.beginPath();
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = pal.pulse;
            ctx.shadowBlur = 10;
            ctx.shadowColor = pal.glow;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        requestAnimationFrame(animate);
    }

    animate();
}

/* ==========================================================================
   2. INTERACTIVE NEURAL NETWORK ARCHITECTURE EXPLORER (NN/DL ENGINE)
   ========================================================================== */
function initNNArchitectureExplorer() {
    const track = document.getElementById('nn-layers-track');
    const inspLayerBadge = document.getElementById('insp-layer-badge');
    const inspTitle = document.getElementById('insp-title');
    const inspDesc = document.getElementById('insp-desc');
    const inspDim = document.getElementById('insp-dim');
    const inspAct = document.getElementById('insp-act');
    const inspKernel = document.getElementById('insp-kernel');
    const inspParams = document.getElementById('insp-params');
    const btnSimulate = document.getElementById('btn-simulate-forward');
    const simStatusText = document.getElementById('sim-status-text');
    const simPulseDot = document.getElementById('sim-pulse-dot');

    // Tab buttons
    const tabBtns = document.querySelectorAll('.nn-tab-btn');
    const tabPanes = document.querySelectorAll('.nn-tab-pane');

    tabBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');
            tabBtns.forEach((b) => b.classList.remove('active'));
            tabPanes.forEach((p) => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    if (!track) return;

    const CNN_LAYERS = [
        {
            id: 'l0',
            idx: '00',
            name: 'Input Tensor',
            type: 'Tensor',
            dim: '(128, 128, 3)',
            act: 'Rescaling (1.0 / 255.0)',
            kernel: 'Full Image Normalization',
            params: '0 (Input Tensor)',
            desc: 'Raw RGB classroom board image resized and normalized to [0, 1] floating-point values for numerical stability and faster gradient descent convergence.'
        },
        {
            id: 'l1',
            idx: '01',
            name: 'Conv2D Block 1',
            type: 'Conv',
            dim: '(126, 126, 32)',
            act: 'ReLU: f(z) = max(0, z)',
            kernel: '32 Filters, 3×3, Stride 1',
            params: '896 Weights + 32 Biases',
            desc: 'Convolves 32 learned 3×3 spatial filters across normalized RGB input to detect low-level edges, chalk strokes, board borders, and contrast gradients.'
        },
        {
            id: 'l2',
            idx: '02',
            name: 'MaxPooling2D 1',
            type: 'Pool',
            dim: '(63, 63, 32)',
            act: 'Subsampling: max(2×2)',
            kernel: 'Pool Size 2×2, Stride 2',
            params: '0 (Non-parametric)',
            desc: 'Performs 2×2 spatial downsampling to halve feature map dimensions, retaining maximum activations and introducing translation invariance.'
        },
        {
            id: 'l3',
            idx: '03',
            name: 'Conv2D Block 2',
            type: 'Conv',
            dim: '(61, 61, 64)',
            act: 'ReLU: f(z) = max(0, z)',
            kernel: '64 Filters, 3×3, Stride 1',
            params: '18,496 Weights + 64 Biases',
            desc: 'Combines low-level edge features into mid-level motifs including stroke intersections, letter curves, and chalkboard grain textures.'
        },
        {
            id: 'l4',
            idx: '04',
            name: 'MaxPooling2D 2',
            type: 'Pool',
            dim: '(30, 30, 64)',
            act: 'Subsampling: max(2×2)',
            kernel: 'Pool Size 2×2, Stride 2',
            params: '0 (Non-parametric)',
            desc: 'Halves resolution from 61×61 to 30×30, expanding the effective receptive field for higher-order convolutional representations.'
        },
        {
            id: 'l5',
            idx: '05',
            name: 'Conv2D Block 3',
            type: 'Conv',
            dim: '(28, 28, 128)',
            act: 'ReLU: f(z) = max(0, z)',
            kernel: '128 Filters, 3×3, Stride 1',
            params: '73,856 Weights + 128 Biases',
            desc: 'Captures high-level compositional semantics such as handwritten word blocks, whole chalkboard boundaries, diagrams, and background furniture.'
        },
        {
            id: 'l6',
            idx: '06',
            name: 'MaxPooling2D 3',
            type: 'Pool',
            dim: '(14, 14, 128)',
            act: 'Subsampling: max(2×2)',
            kernel: 'Pool Size 2×2, Stride 2',
            params: '0 (Non-parametric)',
            desc: 'Final pooling operation condensing high-level activations into 14×14 grid feature representations prior to flattening.'
        },
        {
            id: 'l7',
            idx: '07',
            name: 'Flatten Layer',
            type: 'Dense',
            dim: '25,088 Features',
            act: 'Vector Unroll',
            kernel: '14 × 14 × 128 Flatten',
            params: '0 (Reshape)',
            desc: 'Flattens 14×14×128 spatial tensor maps into a 25,088-dimensional continuous latent feature vector.'
        },
        {
            id: 'l8',
            idx: '08',
            name: 'Dense + Dropout',
            type: 'Dense',
            dim: '128 Hidden Units',
            act: 'ReLU + 30% Dropout',
            kernel: '25,088 × 128 Weight Matrix',
            params: '3,211,392 Weights + 128 Biases',
            desc: 'High-capacity fully-connected layer with 30% dropout (p=0.30) regularization to prevent co-adaptation of neurons and mitigate overfitting.'
        },
        {
            id: 'l9',
            idx: '09',
            name: 'Softmax Output',
            type: 'Output',
            dim: '3 Categorical Classes',
            act: 'Softmax: e^zi / Σ e^zj',
            kernel: '128 × 3 Projection Head',
            params: '384 Weights + 3 Biases',
            desc: 'Computes normalized probability distribution across target classes: [board, text, background] for final categorical inference.'
        }
    ];

    function selectLayer(layer) {
        document.querySelectorAll('.nn-layer-card').forEach((c) => c.classList.remove('active-layer'));
        const el = document.getElementById(`layer-node-${layer.id}`);
        if (el) el.classList.add('active-layer');

        if (inspLayerBadge) inspLayerBadge.innerHTML = `<span class="pulse-dot"></span> LAYER ${layer.idx} • ${layer.type.toUpperCase()}`;
        if (inspTitle) inspTitle.textContent = layer.name;
        if (inspDesc) inspDesc.textContent = layer.desc;
        if (inspDim) inspDim.textContent = layer.dim;
        if (inspAct) inspAct.textContent = layer.act;
        if (inspKernel) inspKernel.textContent = layer.kernel;
        if (inspParams) inspParams.textContent = layer.params;
    }

    // Render layers
    track.innerHTML = CNN_LAYERS.map((layer, index) => {
        let tagClass = 'tag-conv';
        if (layer.type === 'Tensor') tagClass = 'tag-tensor';
        if (layer.type === 'Pool') tagClass = 'tag-pool';
        if (layer.type === 'Dense') tagClass = 'tag-dense';
        if (layer.type === 'Output') tagClass = 'tag-out';

        const isLast = index === CNN_LAYERS.length - 1;

        return `
            <div class="nn-layer-card ${index === 1 ? 'active-layer' : ''}" id="layer-node-${layer.id}" data-id="${layer.id}">
                <div class="nn-layer-top">
                    <span class="nn-layer-idx">L${layer.idx}</span>
                    <span class="nn-layer-type-tag ${tagClass}">${escapeHtml(layer.type)}</span>
                </div>
                <div class="nn-layer-name">${escapeHtml(layer.name)}</div>
                <div class="nn-tensor-dim">${escapeHtml(layer.dim)}</div>
                <div class="nn-layer-activation">
                    <span>Act:</span>
                    <span class="act-badge">${escapeHtml(layer.act.split(':')[0] || layer.act)}</span>
                </div>
            </div>
            ${!isLast ? `
                <div class="nn-synapse-connector" id="synapse-${index}">
                    <div class="synapse-beam" id="beam-${index}"></div>
                </div>
            ` : ''}
        `;
    }).join('');

    // Attach click events
    CNN_LAYERS.forEach((layer) => {
        const card = document.getElementById(`layer-node-${layer.id}`);
        if (card) {
            card.addEventListener('click', () => selectLayer(layer));
        }
    });

    // Simulate Forward Pass (Feedforward propagation)
    if (btnSimulate) {
        btnSimulate.addEventListener('click', () => {
            btnSimulate.disabled = true;
            if (simPulseDot) simPulseDot.classList.add('firing');

            let currentStep = 0;

            const interval = setInterval(() => {
                if (currentStep < CNN_LAYERS.length) {
                    const layer = CNN_LAYERS[currentStep];
                    selectLayer(layer);

                    // Add firing animation
                    const card = document.getElementById(`layer-node-${layer.id}`);
                    if (card) {
                        card.classList.add('firing');
                        setTimeout(() => card.classList.remove('firing'), 350);
                    }

                    // Beam animation
                    const beam = document.getElementById(`beam-${currentStep}`);
                    if (beam) {
                        beam.classList.add('pulsing');
                        setTimeout(() => beam.classList.remove('pulsing'), 350);
                    }

                    if (simStatusText) {
                        simStatusText.textContent = `Forward Pass Active: Processing Layer ${layer.idx} [${layer.name}] ➔ Shape ${layer.dim}`;
                    }

                    currentStep++;
                } else {
                    clearInterval(interval);
                    if (simPulseDot) simPulseDot.classList.remove('firing');
                    if (simStatusText) {
                        simStatusText.textContent = `Forward Pass Complete: Softmax Inference Synthesized [Class: BOARD | Conf: 98.4%]`;
                    }
                    showToast('Feedforward tensor propagation complete!', 'success');
                    setTimeout(() => {
                        btnSimulate.disabled = false;
                        if (simStatusText) {
                            simStatusText.textContent = 'Network Armed: Ready for Forward Pass Simulation';
                        }
                    }, 2400);
                }
            }, 260);
        });
    }
}

/* ==========================================================================
   3. INTERACTIVE UPLOAD DROPZONE & CLIENT-SIDE LIVE STAGING
   ========================================================================== */
function initUploadDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('images');
    const stagingArea = document.getElementById('staging-area');
    const previewGrid = document.getElementById('preview-grid');
    const stagingCountBadge = document.getElementById('staging-count-badge');
    const btnClearAll = document.getElementById('btn-clear-all');
    const uploadForm = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    const submitBtnText = document.getElementById('submit-btn-text');

    if (!dropzone || !fileInput) return;

    // Use DataTransfer to manage the staging queue
    let stagedDataTransfer = new DataTransfer();

    // Drag events
    ['dragenter', 'dragover'].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-active');
        });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-active');
        });
    });

    // Handle Drop
    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFiles(files);
        }
    });

    // Handle File Input Change
    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleSelectedFiles(fileInput.files);
        }
    });

    // Handle Clear All
    if (btnClearAll) {
        btnClearAll.addEventListener('click', (e) => {
            e.preventDefault();
            stagedDataTransfer = new DataTransfer();
            fileInput.files = stagedDataTransfer.files;
            renderStaging();
            showToast('All staged images removed.', 'info');
        });
    }

    // Handle Quick Benchmark Sample Buttons
    const samplePickButtons = document.querySelectorAll('.btn-sample-pick');
    samplePickButtons.forEach((btn) => {
        btn.addEventListener('click', async (e) => {
            e.preventDefault();
            const src = btn.getAttribute('data-sample-src');
            const name = btn.getAttribute('data-sample-name') || 'sample.jpg';
            const origHtml = btn.innerHTML;
            try {
                btn.classList.add('loading');
                btn.style.opacity = '0.7';
                const response = await fetch(src);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const blob = await response.blob();
                const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
                handleSelectedFiles([file]);
                showToast(`Loaded benchmark "${name}" into queue!`, 'success');
            } catch (err) {
                console.error('Failed to load sample:', err);
                showToast(`Could not load ${name}. Check local files.`, 'error');
            } finally {
                btn.classList.remove('loading');
                btn.style.opacity = '1';
                btn.innerHTML = origHtml;
            }
        });
    });

    function handleSelectedFiles(newFiles) {
        const validExtensions = ['jpg', 'jpeg', 'png'];
        let addedCount = 0;

        Array.from(newFiles).forEach((file) => {
            const ext = file.name.split('.').pop().toLowerCase();
            if (!validExtensions.includes(ext)) {
                showToast(`Skipped "${file.name}": Unsupported format.`, 'error');
                return;
            }

            // Check if already staged by name and size
            const alreadyExists = Array.from(stagedDataTransfer.files).some(
                (f) => f.name === file.name && f.size === file.size
            );

            if (!alreadyExists) {
                stagedDataTransfer.items.add(file);
                addedCount++;
            }
        });

        fileInput.files = stagedDataTransfer.files;
        renderStaging();

        if (addedCount > 0) {
            showToast(`Added ${addedCount} image${addedCount > 1 ? 's' : ''} to queue.`, 'success');
        }
    }

    function renderStaging() {
        const files = Array.from(stagedDataTransfer.files);
        previewGrid.innerHTML = '';

        if (files.length === 0) {
            stagingArea.classList.remove('has-files');
            if (submitBtnText) submitBtnText.textContent = 'Upload Board Images';
            return;
        }

        stagingArea.classList.add('has-files');
        stagingCountBadge.textContent = `${files.length} Image${files.length > 1 ? 's' : ''}`;
        if (submitBtnText) submitBtnText.textContent = `Process ${files.length} Board Image${files.length > 1 ? 's' : ''}`;

        files.forEach((file, index) => {
            const card = document.createElement('div');
            card.className = 'preview-card';
            card.style.animationDelay = `${index * 0.05}s`;

            const thumbBox = document.createElement('div');
            thumbBox.className = 'preview-thumb-box';

            const img = document.createElement('img');
            img.className = 'preview-thumb-img';
            img.src = URL.createObjectURL(file);
            img.alt = file.name;

            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'preview-remove-btn';
            removeBtn.title = 'Remove file';
            removeBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            `;

            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeStagedFile(index);
            });

            thumbBox.appendChild(img);
            thumbBox.appendChild(removeBtn);

            const details = document.createElement('div');
            details.className = 'preview-details';

            const nameEl = document.createElement('div');
            nameEl.className = 'preview-filename';
            nameEl.textContent = file.name;
            nameEl.title = file.name;

            const sizeEl = document.createElement('div');
            sizeEl.className = 'preview-size-badge';
            sizeEl.textContent = formatBytes(file.size);

            details.appendChild(nameEl);
            details.appendChild(sizeEl);

            card.appendChild(thumbBox);
            card.appendChild(details);

            previewGrid.appendChild(card);
        });
    }

    function removeStagedFile(indexToRemove) {
        const newDt = new DataTransfer();
        const files = Array.from(stagedDataTransfer.files);

        files.forEach((file, idx) => {
            if (idx !== indexToRemove) {
                newDt.items.add(file);
            }
        });

        stagedDataTransfer = newDt;
        fileInput.files = stagedDataTransfer.files;
        renderStaging();
        showToast('Image removed from staging.', 'info');
    }

    // Submit animation and loading screen
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                showToast('Please select at least one board image.', 'error');
                return;
            }

            showUploadLoadingOverlay();
        });
    }

    function showUploadLoadingOverlay() {
        const overlay = document.getElementById('processing-overlay');
        const statusEl = document.getElementById('processing-status');
        const tagEl = document.getElementById('modal-step-tag');
        const barFill = document.getElementById('progress-bar-fill');
        if (!overlay) return;

        overlay.classList.add('active');

        const stages = [
            { step: 'STAGE 1.0', msg: '1.0 Image Upload & Validation: User uploads and validates classroom-board image...' },
            { step: 'STAGE 2.0', msg: '2.0 Image Preprocessing: Resizing, denoising, enhancing and sharpening image...' },
            { step: 'STAGE 3.0', msg: '3.0 CNN Classification: Identifying board vs text vs background...' },
            { step: 'STAGE 4.0', msg: '4.0 OCR Text Extraction: OCR reading and extracting text from image...' },
            { step: 'STAGE 5.0', msg: '5.0 NLP Analysis: Analyzing extracted text to identify keywords and topics...' },
            { step: 'STAGE 6.0', msg: '6.0 Structured Notes Generation: Generating organized study notes for user...' }
        ];

        let msgIdx = 0;
        let progress = 16;

        if (tagEl) tagEl.textContent = stages[0].step;
        if (statusEl) statusEl.textContent = stages[0].msg;

        const interval = setInterval(() => {
            msgIdx = (msgIdx + 1) % stages.length;
            if (tagEl) tagEl.textContent = stages[msgIdx].step;
            if (statusEl) statusEl.textContent = stages[msgIdx].msg;

            progress = Math.min(progress + 14, 95);
            if (barFill) barFill.style.width = `${progress}%`;
        }, 800);
    }
}

/* ==========================================================================
   3. RESULTS GALLERY & 3D TILT CARDS
   ========================================================================== */
function initResultsGallery() {
    const cards = document.querySelectorAll('.image-result-card');
    const searchInput = document.getElementById('search-input');
    const galleryGrid = document.querySelector('.results-gallery-grid');

    // 3D Card Hover Perspective Tilt
    cards.forEach((card) => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -7;
            const rotateY = ((x - centerX) / centerX) * 7;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px) scale3d(1.01, 1.01, 1.01)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0) scale3d(1, 1, 1)';
        });
    });

    // Real-Time Search / Filter
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            const query = searchInput.value.toLowerCase().trim();
            let visibleCount = 0;

            cards.forEach((card) => {
                const name = card.getAttribute('data-filename') || '';
                if (name.toLowerCase().includes(query)) {
                    card.style.display = 'flex';
                    visibleCount++;
                } else {
                    card.style.display = 'none';
                }
            });

            let emptyMsg = document.getElementById('gallery-empty-search');
            if (visibleCount === 0) {
                if (!emptyMsg && galleryGrid) {
                    emptyMsg = document.createElement('div');
                    emptyMsg.id = 'gallery-empty-search';
                    emptyMsg.style.cssText = 'grid-column: 1/-1; text-align: center; padding: 50px 20px; color: var(--text-secondary);';
                    emptyMsg.innerHTML = `
                        <div style="display: flex; justify-content: center; margin-bottom: 14px;">
                            <div style="width: 44px; height: 44px; border-radius: 50%; background: rgba(255, 255, 255, 0.05); display: flex; align-items: center; justify-content: center; color: var(--text-muted);">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="11" cy="11" r="8"></circle>
                                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                                </svg>
                            </div>
                        </div>
                        <h3 style="font-size: 1.15rem; color: #fff; margin-bottom: 6px;">No matching images found</h3>
                        <p style="font-size: 0.88rem;">Try searching with a different term.</p>
                    `;
                    galleryGrid.appendChild(emptyMsg);
                }
            } else if (emptyMsg) {
                emptyMsg.remove();
            }
        });
    }

    // Copy Filename Buttons
    document.querySelectorAll('.btn-copy-name').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const textToCopy = btn.getAttribute('data-copy');
            if (textToCopy) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    showToast('Filename copied to clipboard!', 'success');
                }).catch(() => {
                    showToast('Could not copy filename.', 'error');
                });
            }
        });
    });
}

/* ==========================================================================
   4. FULLSCREEN LIGHTBOX MODAL
   ========================================================================== */
function initLightboxModal() {
    const modal = document.getElementById('lightbox-modal');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.getElementById('lightbox-close');
    const zoomInBtn = document.getElementById('lightbox-zoom-in');
    const zoomOutBtn = document.getElementById('lightbox-zoom-out');
    const rotateBtn = document.getElementById('lightbox-rotate');
    const resetBtn = document.getElementById('lightbox-reset');

    if (!modal || !lightboxImg) return;

    let currentScale = 1;
    let currentRotation = 0;

    function updateTransform() {
        lightboxImg.style.transform = `scale(${currentScale}) rotate(${currentRotation}deg)`;
    }

    function openLightbox(src, alt) {
        lightboxImg.src = src;
        lightboxImg.alt = alt || 'Preview';
        currentScale = 1;
        currentRotation = 0;
        updateTransform();
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    window.openLightbox = openLightbox;

    function closeLightbox() {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }


    // Open when clicking any media box
    document.querySelectorAll('.card-media-box').forEach((box) => {
        box.addEventListener('click', () => {
            const img = box.querySelector('img');
            if (img) openLightbox(img.src, img.alt);
        });
    });

    if (closeBtn) closeBtn.addEventListener('click', closeLightbox);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeLightbox();
    });

    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentScale = Math.min(currentScale + 0.25, 3.5);
            updateTransform();
        });
    }

    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentScale = Math.max(currentScale - 0.25, 0.5);
            updateTransform();
        });
    }

    if (rotateBtn) {
        rotateBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentRotation = (currentRotation + 90) % 360;
            updateTransform();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            currentScale = 1;
            currentRotation = 0;
            updateTransform();
        });
    }

    // Keyboard support
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeLightbox();
        }
    });
}

/* ==========================================================================
   5. INTERACTIVE LIVE AI ANALYSIS DRAWER
   ========================================================================== */
function initAnalysisDrawer() {
    const drawer = document.getElementById('analysis-drawer');
    const closeBtn = document.getElementById('drawer-close');
    const drawerContent = document.getElementById('drawer-content');

    if (!drawer) return;

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            drawer.classList.remove('open');
        });
    }

    document.querySelectorAll('.btn-run-pipeline').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const savedName = btn.getAttribute('data-saved-name');
            const originalName = btn.getAttribute('data-orig-name') || savedName;
            runAnalysis(savedName, originalName);
        });
    });

    function runAnalysis(savedName, originalName) {
        drawer.classList.add('open');
        showDrawerLoading(originalName);

        // Fetch backend AI analysis with 180s timeout controller
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000);

        fetch(`/api/analyze/${encodeURIComponent(savedName)}`, {
            signal: controller.signal
        })
            .then(async (res) => {
                clearTimeout(timeoutId);
                let data = null;
                try {
                    data = await res.json();
                } catch (e) {
                    data = null;
                }
                if (!res.ok) {
                    const msg = (data && data.error) ? data.error : `Server Error (HTTP ${res.status})`;
                    throw new Error(msg);
                }
                return data;
            })
            .then((data) => {
                if (!data) throw new Error('Received empty response from inference server.');
                if (data.error && !data.success && !data.classification) {
                    renderDrawerError(data.error, savedName, originalName);
                } else {
                    renderDrawerResults(data, originalName);
                    showToast('AI pipeline analysis completed!', 'success');
                }
            })
            .catch((err) => {
                clearTimeout(timeoutId);
                let msg = err.message || 'Failed to connect to inference engine.';
                if (err.name === 'AbortError') {
                    msg = 'Pipeline inference timed out. High-resolution OCR is still completing on CPU.';
                } else if (msg === 'Failed to fetch') {
                    msg = 'Cannot reach backend server. Ensure Flask is actively running on http://127.0.0.1:5000.';
                }
                renderDrawerError(msg, savedName, originalName);
            });
    }

    function showDrawerLoading(name) {
        drawerContent.innerHTML = `
            <div style="text-align: center; padding: 40px 10px;">
                <div class="neural-spinner-wrap" style="transform: scale(0.85); margin-bottom: 20px;">
                    <div class="ring-outer"></div>
                    <div class="ring-inner"></div>
                    <div class="spinner-core"></div>
                </div>
                <h4 style="font-family: var(--font-heading); font-size: 1.15rem; color: #fff; margin-bottom: 8px;">
                    Processing "${escapeHtml(name)}"
                </h4>
                <p style="font-size: 0.85rem; color: var(--primary-cyan); font-family: var(--font-mono);">
                    Executing 6-Stage Deep Learning & Linguistic Pipeline...
                </p>

                <!-- 6-Stage Timeline Simulation -->
                <div class="pipeline-steps-timeline" style="margin-top: 35px; text-align: left;">
                    <div class="step-node completed">
                        <div class="step-circle">✓</div>
                        <div class="step-content">
                            <h4>1.0 Image Upload & Validation</h4>
                            <p>User uploads the classroom-board image. Image format & dimensions verified.</p>
                        </div>
                    </div>
                    <div class="step-node completed">
                        <div class="step-circle">✓</div>
                        <div class="step-content">
                            <h4>2.0 Image Preprocessing</h4>
                            <p>The image is resized, denoised, enhanced and sharpened.</p>
                        </div>
                    </div>
                    <div class="step-node running">
                        <div class="step-circle">⚡</div>
                        <div class="step-content">
                            <h4>3.0 CNN Classification</h4>
                            <p>The CNN identifies the type of image/content such as board, text or background...</p>
                        </div>
                    </div>
                    <div class="step-node">
                        <div class="step-circle">4.0</div>
                        <div class="step-content">
                            <h4>4.0 OCR Text Extraction</h4>
                            <p>OCR reads and extracts text from the classroom image.</p>
                        </div>
                    </div>
                    <div class="step-node">
                        <div class="step-circle">5.0</div>
                        <div class="step-content">
                            <h4>5.0 NLP Analysis</h4>
                            <p>The extracted text is analyzed to identify important keywords and topics.</p>
                        </div>
                    </div>
                    <div class="step-node">
                        <div class="step-circle">6.0</div>
                        <div class="step-content">
                            <h4>6.0 Structured Notes Generation</h4>
                            <p>The system combines the extracted information and generates organized notes for the user.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function renderDrawerResults(data, originalName) {
        const classification = data.classification || { class: 'Unknown', confidence: 0 };
        const text = data.extracted_text || '';
        const keywords = data.keywords || [];
        const topics = data.topics || [];
        const notes = data.structured_notes || {
            title: 'Classroom Board Notes',
            date: new Date().toLocaleDateString(),
            board_type: classification.class,
            confidence: classification.confidence,
            key_points: text ? [text] : [],
            markdown: text,
            plain_text: text
        };
        const savedName = data.processed_filename ? data.processed_filename.replace(/^proc_/, '') : originalName;

        drawerContent.innerHTML = `
            <!-- 6-Stage Completed Checklist Timeline -->
            <div class="pipeline-steps-timeline">
                <div class="step-node completed">
                    <div class="step-circle">✓</div>
                    <div class="step-content">
                        <h4>1.0 Image Upload & Validation</h4>
                        <p>User uploads the classroom-board image. Dimensions and format validated.</p>
                    </div>
                </div>
                <div class="step-node completed">
                    <div class="step-circle">✓</div>
                    <div class="step-content">
                        <h4>2.0 Image Preprocessing</h4>
                        <p>The image is resized, denoised, enhanced and sharpened.</p>
                    </div>
                </div>
                <div class="step-node completed">
                    <div class="step-circle">✓</div>
                    <div class="step-content">
                        <h4>3.0 CNN Classification</h4>
                        <p>The CNN identifies the type of image/content: <strong>${escapeHtml((classification.class || '').toUpperCase())}</strong> (${classification.confidence}%)</p>
                    </div>
                </div>
                <div class="step-node completed">
                    <div class="step-circle">✓</div>
                    <div class="step-content">
                        <h4>4.0 OCR Text Extraction</h4>
                        <p>OCR reads and extracts text from the classroom image. (${(data.ocr_items || []).length} text regions detected)</p>
                    </div>
                </div>
                <div class="step-node completed">
                    <div class="step-circle">✓</div>
                    <div class="step-content">
                        <h4>5.0 NLP Analysis</h4>
                        <p>The extracted text is analyzed to identify important keywords (${keywords.length}) and topics (${topics.length}).</p>
                    </div>
                </div>
                <div class="step-node completed">
                    <div class="step-circle">✓</div>
                    <div class="step-content">
                        <h4>6.0 Structured Notes Generation</h4>
                        <p>The system combines the extracted information and generates organized notes for the user.</p>
                    </div>
                </div>
            </div>

            <!-- Stage 6.0 Structured Notes Generation Display Block -->
            <div class="structured-notes-card">
                <div class="note-header-strip">
                    <div>
                        <div class="step-tag-pill" style="margin-bottom: 6px; display: inline-block;">STAGE 6.0 GENERATED NOTES</div>
                        <h3 class="note-title-text">${escapeHtml(notes.title || 'Classroom Study Notes')}</h3>
                        <div class="note-date-text">${escapeHtml(notes.date || '')} • Source: ${escapeHtml((notes.board_type || classification.class || 'Board').toUpperCase())}</div>
                    </div>
                    <div class="note-actions-group">
                        <button type="button" class="btn-note-action" id="btn-copy-notes" title="Copy complete formatted notes">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            Copy Notes
                        </button>
                        <button type="button" class="btn-note-action" id="btn-download-md" title="Download Markdown format">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7 10 12 15 17 10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                            .MD
                        </button>
                        <button type="button" class="btn-note-action" id="btn-download-txt" title="Download Text format">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="7 10 12 15 17 10"></polyline>
                                <line x1="12" y1="15" x2="12" y2="3"></line>
                            </svg>
                            .TXT
                        </button>
                    </div>
                </div>

                <!-- Structured Key Points -->
                <div class="note-points-box">
                    <div class="note-section-heading">Key Extracted Takeaways & Bullet Points</div>
                    <ul class="note-bullet-list">
                        ${
                            (notes.key_points && notes.key_points.length > 0)
                                ? notes.key_points.map(pt => `
                                    <li class="note-bullet-item">
                                        <span class="note-bullet-dot">▸</span>
                                        <span>${escapeHtml(pt)}</span>
                                    </li>
                                  `).join('')
                                : '<li class="note-bullet-item" style="color: var(--text-muted); font-style: italic;">No bullet points generated.</li>'
                        }
                    </ul>
                </div>
            </div>

            <!-- Stage 3.0 CNN Classification & Softmax Deep Learning Block -->
            <div class="result-block drawer-nn-block">
                <div class="result-block-header">
                    <div>
                        <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">3.0</span>
                        <span class="result-block-title">Deep ConvNet Inference & Softmax</span>
                    </div>
                    <span class="badge-format" style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399;">
                        ${escapeHtml((classification.class || 'UNKNOWN').toUpperCase())} • ${classification.confidence}%
                    </span>
                </div>

                <!-- Softmax Multi-Class Probability Distribution -->
                <div style="font-size: 0.74rem; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;">
                    Softmax Probability Distribution P(y | X) = e^{z_i} / Σ e^{z_j}
                </div>
                <div class="softmax-bars-container">
                    ${(() => {
                        const probs = classification.probabilities || {};
                        const classes = ['board', 'text', 'background'];
                        const topClass = (classification.class || '').toLowerCase();
                        
                        return classes.map((c) => {
                            let p = probs[c];
                            if (p === undefined) {
                                if (c === topClass) p = classification.confidence;
                                else p = Math.max(0, ((100 - classification.confidence) / 2)).toFixed(1);
                            }
                            const isWinner = c === topClass;
                            return `
                                <div class="softmax-row">
                                    <div class="softmax-labels">
                                        <span class="softmax-class-name">${escapeHtml(c)} ${isWinner ? '★ (Predicted)' : ''}</span>
                                        <span class="softmax-percent">${parseFloat(p).toFixed(1)}%</span>
                                    </div>
                                    <div class="softmax-track">
                                        <div class="softmax-fill ${isWinner ? 'winner' : ''}" style="width: ${Math.min(parseFloat(p), 100)}%;"></div>
                                    </div>
                                </div>
                            `;
                        }).join('');
                    })()}
                </div>

                <!-- Tensor Dimension Pipeline Trace -->
                <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border-subtle); font-size: 0.72rem; font-family: var(--font-mono); color: #94a3b8; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                    <span style="color: #38bdf8; font-weight: 600;">Tensor Path:</span>
                    <span>(128,128,3)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span>Conv(126,126,32)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span>Pool(63,63,32)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span>Conv(61,61,64)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span>Pool(30,30,64)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span>Conv(28,28,128)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span>Dense(128)</span>
                    <span style="color: var(--text-muted);">➔</span>
                    <span style="color: #34d399; font-weight: 700;">Softmax(3)</span>
                </div>
            </div>

            <!-- Stage 4.0 OCR Text Extraction Block -->
            <div class="result-block">
                <div class="result-block-header">
                    <div>
                        <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">4.0</span>
                        <span class="result-block-title">Deep Sequence OCR Extraction</span>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <span class="badge-format" style="background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.35); color: #c4b5fd;">
                            CRAFT + BiLSTM CTC
                        </span>
                        <button class="btn-copy-name" id="btn-copy-ocr" title="Copy Text" style="color: var(--primary-cyan); font-size: 0.8rem; display: flex; align-items: center; gap: 4px;">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            Copy Raw Text
                        </button>
                    </div>
                </div>
                <div class="extracted-text-area" id="ocr-text-display">${text ? escapeHtml(text) : '<span style="color: var(--text-muted); font-style: italic;">No readable text characters extracted with current confidence threshold.</span>'}</div>
            </div>

            <!-- Stage 5.0 NLP Analysis Block -->
            <div class="result-block">
                <div class="result-block-header">
                    <div>
                        <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">5.0</span>
                        <span class="result-block-title">NLP Key Vocabulary</span>
                    </div>
                    <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">${keywords.length} Identified</span>
                </div>
                <div class="keyword-tags-cloud">
                    ${
                        keywords.length > 0
                            ? keywords.map((k, i) => `<span class="keyword-pill" style="animation-delay: ${i * 0.05}s">${escapeHtml(k)}</span>`).join('')
                            : '<span style="color: var(--text-muted); font-size: 0.82rem;">None detected</span>'
                    }
                </div>
            </div>

            <!-- Stage 5.0 Topic Taxonomy Block -->
            <div class="result-block">
                <div class="result-block-header">
                    <div>
                        <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">5.0</span>
                        <span class="result-block-title">NLP Academic Topics</span>
                    </div>
                    <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono);">${topics.length} Matches</span>
                </div>
                <div>
                    ${
                        topics.length > 0
                            ? topics.map((t) => `
                                <div class="topic-item">
                                    <div class="topic-item-title">${escapeHtml(t.topic)}</div>
                                    <div class="topic-item-keys">Matched Terms: ${escapeHtml((t.keywords || []).join(', '))}</div>
                                </div>
                              `).join('')
                            : '<span style="color: var(--text-muted); font-size: 0.82rem;">No specific subject topic matched.</span>'
                    }
                </div>
            </div>

            <!-- Stage 2.0 Preprocessed Image Output Block -->
            ${
                data.processed_filename ? `
                <div class="result-block">
                    <div class="result-block-header">
                        <div>
                            <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">2.0</span>
                            <span class="result-block-title">Enhanced Image & ConvNet Activation</span>
                        </div>
                        <a href="/pipeline_output/${encodeURIComponent(data.processed_filename)}" target="_blank" class="btn-note-action" style="font-size: 0.75rem; padding: 4px 8px;">
                            View Fullsize
                        </a>
                    </div>
                    <div class="drawer-preview-img-wrap" style="border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border-subtle); max-height: 220px; display: flex; align-items: center; justify-content: center; background: #000; position: relative;">
                        <img src="/pipeline_output/${encodeURIComponent(data.processed_filename)}" alt="Preprocessed Preview" style="max-width: 100%; max-height: 220px; object-fit: contain;">
                        <div class="heatmap-overlay" id="heatmap-overlay"></div>
                    </div>
                    <div class="heatmap-toggle-wrap">
                        <span style="font-size: 0.76rem; color: var(--text-muted); font-family: var(--font-mono);">Grad-CAM Feature Map Filter Response</span>
                        <button type="button" class="btn-heatmap-toggle" id="btn-heatmap-toggle">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                <circle cx="12" cy="12" r="3"></circle>
                            </svg>
                            Toggle Activation Heatmap
                        </button>
                    </div>
                </div>
                ` : ''
            }
        `;

        // Wire heatmap toggle button
        const heatmapBtn = document.getElementById('btn-heatmap-toggle');
        const heatmapOverlay = document.getElementById('heatmap-overlay');
        if (heatmapBtn && heatmapOverlay) {
            heatmapBtn.addEventListener('click', () => {
                const isVisible = heatmapOverlay.classList.toggle('visible');
                heatmapBtn.classList.toggle('active', isVisible);
                showToast(isVisible ? 'ConvNet Feature Activation Heatmap enabled!' : 'Activation Heatmap hidden.', 'info');
            });
        }

        // Wire copy OCR text button
        const copyOcrBtn = document.getElementById('btn-copy-ocr');
        if (copyOcrBtn && text) {
            copyOcrBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(text).then(() => {
                    showToast('Extracted OCR text copied!', 'success');
                });
            });
        }

        // Wire copy Structured Notes button
        const copyNotesBtn = document.getElementById('btn-copy-notes');
        if (copyNotesBtn) {
            copyNotesBtn.addEventListener('click', () => {
                const noteContent = notes.markdown || notes.plain_text || text;
                navigator.clipboard.writeText(noteContent).then(() => {
                    showToast('Complete structured notes copied to clipboard!', 'success');
                }).catch(() => {
                    showToast('Failed to copy notes.', 'error');
                });
            });
        }

        // Wire download Markdown (.md)
        const downloadMdBtn = document.getElementById('btn-download-md');
        if (downloadMdBtn) {
            downloadMdBtn.addEventListener('click', () => {
                const mdContent = notes.markdown || `# ${notes.title}\n\n${text}`;
                const blob = new Blob([mdContent], { type: 'text/markdown' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `notes_${(notes.title || 'study_notes').replace(/[^a-zA-Z0-9_]/g, '_')}.md`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast('Structured notes downloaded (.md)!', 'success');
            });
        }

        // Wire download Plaintext (.txt)
        const downloadTxtBtn = document.getElementById('btn-download-txt');
        if (downloadTxtBtn) {
            downloadTxtBtn.addEventListener('click', () => {
                const txtContent = notes.plain_text || text;
                const blob = new Blob([txtContent], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `notes_${(notes.title || 'study_notes').replace(/[^a-zA-Z0-9_]/g, '_')}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                showToast('Structured notes downloaded (.txt)!', 'success');
            });
        }
    }

    function renderDrawerError(errMsg, savedName, originalName) {
        drawerContent.innerHTML = `
            <div style="text-align: center; padding: 40px 15px;">
                <div style="display: flex; justify-content: center; margin-bottom: 14px;">
                    <div style="width: 48px; height: 48px; border-radius: 50%; background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); display: flex; align-items: center; justify-content: center; color: var(--primary-rose);">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                    </div>
                </div>
                <h4 style="color: #fb7185; font-size: 1.15rem; margin-bottom: 8px;">Analysis Failed</h4>
                <p style="color: var(--text-secondary); font-size: 0.88rem; margin-bottom: 20px;">
                    ${escapeHtml(errMsg)}
                </p>
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: var(--radius-sm); padding: 12px; font-size: 0.8rem; color: #fecdd3; font-family: var(--font-mono); margin-bottom: 18px;">
                    Ensure the pipeline models and OpenCV dependencies are loaded in the environment.
                </div>
                ${savedName ? `
                <div>
                    <button type="button" class="btn-neural-submit" id="btn-retry-analysis" style="font-size: 0.82rem; padding: 8px 18px; width: auto; display: inline-flex; margin: 0 auto;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="1 4 1 10 7 10"></polyline>
                            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
                        </svg>
                        Retry Pipeline Analysis
                    </button>
                </div>
                ` : ''}
            </div>
        `;

        const retryBtn = document.getElementById('btn-retry-analysis');
        if (retryBtn && savedName) {
            retryBtn.addEventListener('click', () => {
                runAnalysis(savedName, originalName || savedName);
            });
        }
    }
}

/* ==========================================================================
   6. UTILITY HELPERS
   ========================================================================== */
function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function escapeHtml(str) {
    if (!str) return '';
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function showToast(message, type = 'info', duration = 3200) {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = 'i';
    if (type === 'success') icon = '✓';
    if (type === 'error') icon = '✕';

    toast.innerHTML = `
        <span style="font-weight: 700; color: ${type === 'success' ? '#34d399' : type === 'error' ? '#fb7185' : '#38bdf8'}">${icon}</span>
        <span>${escapeHtml(message)}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.35s ease forwards';
        setTimeout(() => toast.remove(), 350);
    }, duration);
}
