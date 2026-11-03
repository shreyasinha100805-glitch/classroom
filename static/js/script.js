/**
 * AI Classroom Board Content Extractor
 * Futuristic Neural Studio Frontend Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    initConstellationBackground();
    initUploadDropzone();
    initResultsGallery();
    initLightboxModal();
    initAnalysisDrawer();
});

/* ==========================================================================
   1. DYNAMIC CONSTELLATION CANVAS
   ========================================================================== */
function initConstellationBackground() {
    const canvas = document.getElementById('particle-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    let mouse = { x: null, y: null, radius: 150 };

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
        initParticles();
    });

    const particles = [];
    const particleCount = Math.min(Math.floor((width * height) / 18000), 75);

    class Particle {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.7;
            this.vy = (Math.random() - 0.5) * 0.7;
            this.radius = Math.random() * 1.8 + 0.8;
            this.baseColor = Math.random() > 0.5 ? 'rgba(99, 102, 241,' : 'rgba(6, 182, 212,';
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;

            // Mouse interaction
            if (mouse.x !== null) {
                const dx = mouse.x - this.x;
                const dy = mouse.y - this.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < mouse.radius) {
                    const force = (mouse.radius - dist) / mouse.radius;
                    this.x -= (dx / dist) * force * 1.2;
                    this.y -= (dy / dist) * force * 1.2;
                }
            }
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.baseColor + '0.7)';
            ctx.shadowBlur = 8;
            ctx.shadowColor = this.baseColor + '0.5)';
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }

    function initParticles() {
        particles.length = 0;
        for (let i = 0; i < particleCount; i++) {
            particles.push(new Particle());
        }
    }

    initParticles();

    function animate() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            particles[i].update();
            particles[i].draw();

            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 115) {
                    const alpha = (1 - dist / 115) * 0.22;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }

    animate();
}

/* ==========================================================================
   2. INTERACTIVE UPLOAD DROPZONE & CLIENT-SIDE LIVE STAGING
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

        // Fetch backend AI analysis
        fetch(`/api/analyze/${encodeURIComponent(savedName)}`)
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.json();
            })
            .then((data) => {
                if (data.error) {
                    renderDrawerError(data.error);
                } else {
                    renderDrawerResults(data, originalName);
                    showToast('AI pipeline analysis completed!', 'success');
                }
            })
            .catch((err) => {
                renderDrawerError(err.message || 'Failed to connect to inference engine.');
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

            <!-- Stage 3.0 CNN Classification Block -->
            <div class="result-block">
                <div class="result-block-header">
                    <div>
                        <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">3.0</span>
                        <span class="result-block-title">CNN Classification</span>
                    </div>
                    <span class="badge-format" style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399;">
                        ${escapeHtml(classification.class.toUpperCase())}
                    </span>
                </div>
                <div class="confidence-gauge-box">
                    <div class="confidence-labels">
                        <span>Model Prediction Confidence</span>
                        <strong>${classification.confidence}%</strong>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill" style="width: ${Math.min(classification.confidence, 100)}%;"></div>
                    </div>
                </div>
            </div>

            <!-- Stage 4.0 OCR Text Extraction Block -->
            <div class="result-block">
                <div class="result-block-header">
                    <div>
                        <span class="step-tag-pill" style="font-size: 0.65rem; margin-right: 6px;">4.0</span>
                        <span class="result-block-title">OCR Text Extraction</span>
                    </div>
                    <button class="btn-copy-name" id="btn-copy-ocr" title="Copy Text" style="color: var(--primary-cyan); font-size: 0.8rem; display: flex; align-items: center; gap: 4px;">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        Copy Raw Text
                    </button>
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
                            <span class="result-block-title">Enhanced Preprocessed Image</span>
                        </div>
                        <a href="/pipeline_output/${encodeURIComponent(data.processed_filename)}" target="_blank" class="btn-note-action" style="font-size: 0.75rem; padding: 4px 8px;">
                            View Fullsize
                        </a>
                    </div>
                    <div style="border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border-subtle); max-height: 200px; display: flex; align-items: center; justify-content: center; background: #000;">
                        <img src="/pipeline_output/${encodeURIComponent(data.processed_filename)}" alt="Preprocessed Preview" style="max-width: 100%; max-height: 200px; object-fit: contain;">
                    </div>
                </div>
                ` : ''
            }
        `;

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

    function renderDrawerError(errMsg) {
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
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.25); border-radius: var(--radius-sm); padding: 12px; font-size: 0.8rem; color: #fecdd3; font-family: var(--font-mono);">
                    Ensure the pipeline models and OpenCV dependencies are loaded in the environment.
                </div>
            </div>
        `;
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
