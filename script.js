// ===== CONSTANTS =====
const FREQUENCIES = [250, 500, 1000, 2000, 3000, 4000, 8000];
const KEY_FREQUENCIES_FOR_AVG = [500, 1000, 2000, 4000];
const API_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models';

// ===== TOAST =====
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        toast.addEventListener('transitionend', () => toast.remove());
    }, duration);
}

// ===== GEMINI API CLIENT (MULTI-BACKEND) =====
class GeminiAPIClient {
    constructor() {
        this.backend = localStorage.getItem('ai_backend') || 'cloud';
        // Cloud settings
        this.apiKey = localStorage.getItem('gemma_api_key') || '';
        this.model = localStorage.getItem('gemma_model') || 'gemini-3.1-flash-lite-preview';
        // Auto-migrate from old defaults
        if (['gemma-4-27b-it', 'gemma-4-12b-it', 'gemma-4-4b-it', 'gemma-4-26b-a4b-it'].includes(this.model)) {
            this.model = 'gemini-3.1-flash-lite-preview';
            localStorage.setItem('gemma_model', this.model);
        }
        // Ollama settings
        this.ollamaUrl = localStorage.getItem('ollama_url') || 'http://localhost:11434';
        this.ollamaModel = localStorage.getItem('ollama_model') || 'gemma3:27b';
        // Custom settings
        this.customUrl = localStorage.getItem('custom_url') || '';
        this.customApiKey = localStorage.getItem('custom_api_key') || '';
        this.customModel = localStorage.getItem('custom_model_name') || '';
        // On-device settings
        this.onDeviceModelUrl = localStorage.getItem('ondevice_model_url') || '';
        this.llmInference = null; // MediaPipe LlmInference instance
        this.onDeviceReady = false;
    }

    get isConfigured() {
        switch (this.backend) {
            case 'cloud': return !!this.apiKey;
            case 'ondevice': return this.onDeviceReady;
            case 'ollama': return !!this.ollamaUrl;
            case 'custom': return !!this.customUrl;
            default: return false;
        }
    }

    get backendLabel() {
        switch (this.backend) {
            case 'cloud': return `☁️ Cloud (${this.model})`;
            case 'ondevice': return `📱 On-Device (WebGPU)`;
            case 'ollama': return `🖥️ Ollama (${this.ollamaModel})`;
            case 'custom': return `⚙️ Custom (${this.customModel})`;
            default: return 'No configurado';
        }
    }

    // --- CLOUD (Google AI Studio) ---
    async _cloudRequest(body) {
        const url = `${API_ENDPOINT}/${this.model}:generateContent?key=${this.apiKey}`;
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error?.message || `Error API Cloud: ${res.status}`); }
        const data = await res.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    }

    async _cloudAnalyzeImage(base64Data, mimeType, prompt) {
        return this._cloudRequest({ contents: [{ parts: [{ inlineData: { mimeType, data: base64Data } }, { text: prompt }] }] });
    }

    async _cloudGenerateText(prompt) {
        return this._cloudRequest({ contents: [{ parts: [{ text: prompt }] }] });
    }

    // --- OLLAMA (Local Server) ---
    async _ollamaRequest(prompt, images = null) {
        const url = `${this.ollamaUrl.replace(/\/$/, '')}/api/generate`;
        const body = { model: this.ollamaModel, prompt, stream: false };
        if (images) body.images = images;
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!res.ok) throw new Error(`Error Ollama (${res.status}): Verifica que Ollama esté corriendo y el modelo descargado.`);
        const data = await res.json();
        return data.response || '';
    }

    async _ollamaAnalyzeImage(base64Data, mimeType, prompt) {
        return this._ollamaRequest(prompt, [base64Data]);
    }

    async _ollamaGenerateText(prompt) {
        return this._ollamaRequest(prompt);
    }

    // --- CUSTOM (OpenAI-compatible) ---
    async _customRequest(messages) {
        const headers = { 'Content-Type': 'application/json' };
        if (this.customApiKey) headers['Authorization'] = `Bearer ${this.customApiKey}`;
        const body = { model: this.customModel, messages, stream: false };
        const res = await fetch(this.customUrl, { method: 'POST', headers, body: JSON.stringify(body) });
        if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error?.message || `Error Custom API: ${res.status}`); }
        const data = await res.json();
        return data.choices?.[0]?.message?.content || '';
    }

    async _customAnalyzeImage(base64Data, mimeType, prompt) {
        return this._customRequest([{ role: 'user', content: [
            { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64Data}` } },
            { type: 'text', text: prompt }
        ]}]);
    }

    async _customGenerateText(prompt) {
        return this._customRequest([{ role: 'user', content: prompt }]);
    }

    // --- ON-DEVICE (MediaPipe WebGPU) ---
    async loadModel(modelUrl, onProgress) {
        if (typeof LlmInference === 'undefined' && typeof self.LlmInference === 'undefined') {
            // Try to access from MediaPipe tasks-genai bundle
            const mp = self.FilesetResolver ? self : (self.mediapipe || {});
            if (!mp.FilesetResolver) throw new Error('MediaPipe GenAI no disponible. Recarga la página.');
            const genai = await mp.FilesetResolver.forGenAiTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm'
            );
            this.llmInference = await mp.LlmInference.createFromOptions(genai, {
                baseOptions: { modelAssetPath: modelUrl },
                maxTokens: 2048,
                maxNumImages: 1
            });
        } else {
            const LlmInferenceClass = self.LlmInference || self.mediapipe?.LlmInference;
            const FilesetResolverClass = self.FilesetResolver || self.mediapipe?.FilesetResolver;
            const genai = await FilesetResolverClass.forGenAiTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm'
            );
            this.llmInference = await LlmInferenceClass.createFromOptions(genai, {
                baseOptions: { modelAssetPath: modelUrl },
                maxTokens: 2048,
                maxNumImages: 1
            });
        }
        this.onDeviceReady = true;
    }

    async _onDeviceAnalyzeImage(base64Data, mimeType, prompt) {
        if (!this.llmInference) throw new Error('Modelo on-device no cargado. Ve a Configuración.');
        // Create an image element from base64 for MediaPipe
        const img = new Image();
        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
            img.src = `data:${mimeType};base64,${base64Data}`;
        });
        const response = await this.llmInference.generateResponse([
            { type: 'image', data: img },
            { type: 'text', text: prompt }
        ]);
        return response;
    }

    async _onDeviceGenerateText(prompt) {
        if (!this.llmInference) throw new Error('Modelo on-device no cargado.');
        return await this.llmInference.generateResponse(prompt);
    }

    // --- PUBLIC INTERFACE ---
    async analyzeImage(base64Data, mimeType, prompt) {
        if (!this.isConfigured) throw new Error('Backend no configurado. Ve a ⚙️ Configuración.');
        switch (this.backend) {
            case 'cloud': return this._cloudAnalyzeImage(base64Data, mimeType, prompt);
            case 'ondevice': return this._onDeviceAnalyzeImage(base64Data, mimeType, prompt);
            case 'ollama': return this._ollamaAnalyzeImage(base64Data, mimeType, prompt);
            case 'custom': return this._customAnalyzeImage(base64Data, mimeType, prompt);
            default: throw new Error('Backend desconocido');
        }
    }

    async generateText(prompt) {
        if (!this.isConfigured) throw new Error('Backend no configurado.');
        switch (this.backend) {
            case 'cloud': return this._cloudGenerateText(prompt);
            case 'ondevice': return this._onDeviceGenerateText(prompt);
            case 'ollama': return this._ollamaGenerateText(prompt);
            case 'custom': return this._customGenerateText(prompt);
            default: throw new Error('Backend desconocido');
        }
    }
}

// ===== IMAGE PREPROCESSOR =====
class ImagePreprocessor {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.originalImage = null;
        this.brightness = 0;
        this.contrast = 0;
        this.rotation = 0;
    }

    async loadImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => { this.originalImage = img; this.brightness = 0; this.contrast = 0; this.rotation = 0; this.render(); resolve(); };
                img.onerror = reject;
                img.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    render() {
        if (!this.originalImage) return;
        const img = this.originalImage;
        const isRotated = this.rotation % 180 !== 0;
        const w = isRotated ? img.height : img.width;
        const h = isRotated ? img.width : img.height;
        const maxW = Math.min(this.canvas.parentElement.clientWidth - 24, 700);
        const maxH = 400;
        const scale = Math.min(maxW / w, maxH / h, 1);
        this.canvas.width = w * scale;
        this.canvas.height = h * scale;
        this.ctx.save();
        this.ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
        this.ctx.rotate((this.rotation * Math.PI) / 180);
        this.ctx.filter = `brightness(${100 + this.brightness}%) contrast(${100 + this.contrast}%)`;
        this.ctx.drawImage(img, -img.width * scale / 2, -img.height * scale / 2, img.width * scale, img.height * scale);
        this.ctx.restore();
    }

    setBrightness(v) { this.brightness = parseInt(v); this.render(); }
    setContrast(v) { this.contrast = parseInt(v); this.render(); }
    rotate90() { this.rotation = (this.rotation + 90) % 360; this.render(); }
    reset() { this.brightness = 0; this.contrast = 0; this.rotation = 0; this.render(); }
    getBase64Data() { return this.canvas.toDataURL('image/jpeg', 0.88).split(',')[1]; }
}

// ===== AUDIOMETRY CALCULATOR =====
class AudiometryCalculator {
    static calculateAverage(data, freqs) {
        let sum = 0, count = 0;
        freqs.forEach(f => { const v = data[f]; if (!isNaN(v) && v !== null) { sum += v; count++; } });
        return count > 0 ? sum / count : 0;
    }

    static diagnoseHipoacusia(va, vo) {
        const avgVA = this.calculateAverage(va, KEY_FREQUENCIES_FOR_AVG);
        const avgVO = this.calculateAverage(vo, KEY_FREQUENCIES_FOR_AVG);
        const avgGAP = avgVA - avgVO;
        if (avgVA <= 20) return "Audición Normal";
        if (avgGAP > 10) return avgVO <= 20 ? "De Transmisión (Conductiva)" : "Mixta";
        return "Neurosensorial";
    }

    static calculateBinauralDeficiency(vaOD, vaOI) {
        const freqs = KEY_FREQUENCIES_FOR_AVG;
        const sumOD = freqs.reduce((a, f) => a + (vaOD[f] || 0), 0);
        const sumOI = freqs.reduce((a, f) => a + (vaOI[f] || 0), 0);
        let impOD = Math.min(100, Math.max(0, (sumOD / 4 - 25) * 1.5));
        let impOI = Math.min(100, Math.max(0, (sumOI / 4 - 25) * 1.5));
        const better = Math.min(impOD, impOI), worse = Math.max(impOD, impOI);
        return { binauralDeficiency: (better * 5 + worse) / 6, impairmentOD: impOD, impairmentOI: impOI, sumOD, sumOI };
    }

    static getDisabilityFromTableA(d) {
        const t = [[1.6,1],[3.2,2],[4.8,3],[6.4,4],[8,5],[9.6,6],[11.2,7],[12.8,8],[14.4,9],[16,10],[17.6,11],[19.2,12],[20.8,13],[22.4,14],[23.9,15],[25.4,16],[26.9,17],[28.4,18],[29.9,19],[32.5,20],[35,21],[37.5,22],[40,23],[42.5,24],[45,25],[47.5,26],[50,27],[52.5,28],[54.9,29],[59.5,30],[64,31],[68.5,32],[73,33],[77.5,34],[81.9,35],[85.6,36],[89.2,37],[92.8,38],[96.4,39],[100,40]];
        for (const [threshold, val] of t) { if (d <= threshold) return val; }
        return 40;
    }

    static getDisabilityFromTableB(d) {
        const t = [[1.4,0],[4.2,1],[7.1,2],[9.9,3],[12.8,4],[15.7,5],[18.5,6],[21.4,7],[24.2,8],[27.1,9],[29.9,10],[32.8,11],[35.7,12],[38.5,13],[41.4,14],[44.2,15],[47.1,16],[49.9,17],[52.8,18],[55.7,19],[58.5,20],[61.4,21],[64.2,22],[67.1,23],[69.9,24],[72.8,25],[75.7,26],[78.5,27],[81.4,28],[84.2,29],[87.1,30],[89.9,31],[92.8,32],[95.7,33],[98.5,34],[100,35]];
        for (const [threshold, val] of t) { if (d <= threshold) return val; }
        return 35;
    }

    static getDisabilityPercentage(bDef, diagOD, diagOI) {
        const rd = parseFloat(bDef.toFixed(1));
        const useB = diagOD === "De Transmisión (Conductiva)" && diagOI === "De Transmisión (Conductiva)";
        return useB ? this.getDisabilityFromTableB(rd) : this.getDisabilityFromTableA(rd);
    }

    static getDisabilityClass(pct) {
        if (pct > 95) return { class: "Clase 5 (Absoluta)", range: "> 95%" };
        if (pct >= 75) return { class: "Clase 4 (Total)", range: "75% - 95%" };
        if (pct >= 50) return { class: "Clase 3 (Grave)", range: "50% - 74%" };
        if (pct >= 25) return { class: "Clase 2 (Moderada)", range: "25% - 49%" };
        if (pct >= 5) return { class: "Clase 1 (Leve)", range: "5% - 24%" };
        return { class: "Clase 0 (Sin discapacidad)", range: "0% - 4%" };
    }
}

// ===== AUDIOGRAM RENDERER =====
class AudiogramRenderer {
    static draw(odVA, odVO, oiVA, oiVO, containerId = 'audiogram-container') {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';
        const width = 800, height = 500, padding = 50;
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        container.appendChild(tooltip);

        const xScale = f => padding + (FREQUENCIES.indexOf(f) / (FREQUENCIES.length - 1)) * (width - 2 * padding);
        const yScale = db => padding + ((db + 10) / 130) * (height - 2 * padding);
        const dbLevels = Array.from({ length: 14 }, (_, i) => i * 10 - 10);

        let gridHtml = '';
        dbLevels.forEach(db => {
            const y = yScale(db);
            gridHtml += `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>`;
            gridHtml += `<text x="${padding - 15}" y="${y + 4}" text-anchor="end" font-size="11" fill="rgba(255,255,255,0.5)">${db}</text>`;
        });
        FREQUENCIES.forEach(f => {
            const x = xScale(f);
            gridHtml += `<line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>`;
            gridHtml += `<text x="${x}" y="${height - padding + 20}" text-anchor="middle" font-size="11" fill="rgba(255,255,255,0.5)">${f >= 1000 ? f / 1000 + 'K' : f}</text>`;
        });
        gridHtml += `<text x="${padding / 2}" y="${height / 2}" transform="rotate(-90, ${padding / 2}, ${height / 2})" text-anchor="middle" font-weight="bold" font-size="12" fill="rgba(255,255,255,0.6)">Intensidad (dB HL)</text>`;
        gridHtml += `<text x="${width / 2}" y="${height - 8}" text-anchor="middle" font-weight="bold" font-size="12" fill="rgba(255,255,255,0.6)">Frecuencia (Hz)</text>`;
        svg.innerHTML = gridHtml;
        container.appendChild(svg);

        const drawPath = (data, symbol, color, isDashed, ear, type) => {
            let pathData = "";
            const points = FREQUENCIES.map(f => { const v = data[f]; return !isNaN(v) && v !== null ? { x: xScale(f), y: yScale(v), freq: f, value: v } : null; }).filter(Boolean);
            points.forEach((p, i) => {
                pathData += `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`;
                const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
                el.setAttribute("x", p.x); el.setAttribute("y", p.y);
                Object.assign(el.style, { textAnchor: "middle", dominantBaseline: "middle", fontSize: "20px", fill: color, cursor: "pointer" });
                el.textContent = symbol;
                el.onmouseover = () => { tooltip.style.opacity = '1'; tooltip.innerHTML = `${ear.toUpperCase()} ${type}: ${p.freq}Hz @ ${p.value}dB`; };
                el.onmousemove = (e) => { const r = container.getBoundingClientRect(); tooltip.style.left = `${e.clientX - r.left}px`; tooltip.style.top = `${e.clientY - r.top}px`; };
                el.onmouseout = () => { tooltip.style.opacity = '0'; };
                svg.appendChild(el);
            });
            if (pathData) {
                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("d", pathData);
                Object.assign(path.style, { fill: "none", stroke: color, strokeWidth: "2" });
                if (isDashed) path.style.strokeDasharray = "5,5";
                svg.appendChild(path);
            }
        };

        const legends = [{ s: 'O', t: 'OD VA', c: '#ef4444' }, { s: '[', t: 'OD VO', c: '#ef4444' }, { s: 'X', t: 'OI VA', c: '#60a5fa' }, { s: ']', t: 'OI VO', c: '#60a5fa' }];
        legends.forEach((item, i) => {
            const y = padding + 20 + i * 22;
            const sym = document.createElementNS("http://www.w3.org/2000/svg", "text");
            sym.setAttribute('x', width - padding - 80); sym.setAttribute('y', y);
            sym.style.fill = item.c; sym.style.fontSize = '18px'; sym.textContent = item.s;
            svg.appendChild(sym);
            const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
            txt.setAttribute('x', width - padding - 60); txt.setAttribute('y', y);
            txt.style.fill = 'rgba(255,255,255,0.6)'; txt.style.fontSize = '11px'; txt.style.dominantBaseline = 'middle';
            txt.textContent = item.t; svg.appendChild(txt);
        });

        drawPath(odVA, 'O', '#ef4444', false, 'od', 'VA');
        drawPath(odVO, '[', '#ef4444', true, 'od', 'VO');
        drawPath(oiVA, 'X', '#60a5fa', false, 'oi', 'VA');
        drawPath(oiVO, ']', '#60a5fa', true, 'oi', 'VO');
    }
}

// ===== CLIPBOARD HELPER =====
async function copyToClipboard(html, text) {
    try {
        await navigator.clipboard.write([new ClipboardItem({
            'text/html': new Blob([html], { type: 'text/html' }),
            'text/plain': new Blob([text], { type: 'text/plain' })
        })]);
        return true;
    } catch { try { await navigator.clipboard.writeText(text); return true; } catch { return false; } }
}

// ===== REPORT HELPERS =====
const freqRow = (label, data) => {
    let cells = `<td>${label}</td>`;
    FREQUENCIES.forEach(f => { cells += `<td>${!isNaN(data[f]) && data[f] !== null ? data[f] : '-'}</td>`; });
    return `<tr>${cells}</tr>`;
};

// ===== APP CONTROLLER =====
class AppController {
    constructor() {
        this.api = new GeminiAPIClient();
        this.preprocessor = null;
        this.currentResults = null;
        this.records = [];
        this.imageLoaded = false;
    }

    init() {
        this.preprocessor = new ImagePreprocessor(document.getElementById('image-canvas'));
        this.initDate();
        this.loadRecords();
        this.setupEventListeners();
        this.updateAPIStatus();
        if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
    }

    initDate() { document.getElementById('audiometry-date').value = new Date().toISOString().split('T')[0]; }

    // --- Navigation ---
    navigateTo(page) {
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.getElementById(`page-${page}`)?.classList.add('active');
        document.querySelector(`[data-page="${page}"]`)?.classList.add('active');
        document.getElementById('pages-container').scrollTop = 0;
    }

    // --- Event Listeners ---
    setupEventListeners() {
        document.querySelectorAll('.nav-item').forEach(btn => {
            btn.addEventListener('click', () => this.navigateTo(btn.dataset.page));
        });

        const handleFile = (e) => { if (e.target.files[0]) this.handleImageCapture(e.target.files[0]); };
        document.getElementById('camera-input')?.addEventListener('change', handleFile);
        document.getElementById('gallery-input')?.addEventListener('change', handleFile);
        document.getElementById('import-file')?.addEventListener('change', (e) => this.importData(e));

        const bs = document.getElementById('brightness-slider');
        const cs = document.getElementById('contrast-slider');
        bs?.addEventListener('input', () => { this.preprocessor.setBrightness(bs.value); document.getElementById('brightness-value').textContent = bs.value; });
        cs?.addEventListener('input', () => { this.preprocessor.setContrast(cs.value); document.getElementById('contrast-value').textContent = cs.value; });

        ['od', 'oi'].forEach(ear => {
            FREQUENCIES.forEach(freq => {
                const vaInput = document.getElementById(`va-${ear}-${freq}`);
                vaInput?.addEventListener('input', () => {
                    const voInput = document.getElementById(`vo-${ear}-${freq}`);
                    if (voInput && voInput.value === '0') voInput.value = vaInput.value;
                });
            });
        });
    }

    // --- Image Handling ---
    async handleImageCapture(file) {
        try {
            await this.preprocessor.loadImage(file);
            this.imageLoaded = true;
            document.getElementById('capture-empty').classList.add('hidden');
            document.getElementById('capture-preview').classList.remove('hidden');
            document.getElementById('brightness-slider').value = 0;
            document.getElementById('contrast-slider').value = 0;
            document.getElementById('brightness-value').textContent = '0';
            document.getElementById('contrast-value').textContent = '0';
            showToast('Imagen cargada correctamente', 'success');
        } catch { showToast('Error al cargar la imagen', 'error'); }
    }

    rotateImage() { this.preprocessor.rotate90(); }
    resetAdjustments() {
        this.preprocessor.reset();
        document.getElementById('brightness-slider').value = 0;
        document.getElementById('contrast-slider').value = 0;
        document.getElementById('brightness-value').textContent = '0';
        document.getElementById('contrast-value').textContent = '0';
    }
    changeImage() {
        document.getElementById('capture-empty').classList.remove('hidden');
        document.getElementById('capture-preview').classList.add('hidden');
        this.imageLoaded = false;
        document.getElementById('camera-input').value = '';
        document.getElementById('gallery-input').value = '';
    }

    // --- AI Analysis ---
    async analyzeWithAI() {
        if (!this.imageLoaded) { showToast('Carga una imagen primero', 'warning'); return; }
        if (!this.api.isConfigured) { showToast('Configura tu API Key en ⚙️ Configuración', 'warning'); this.openSettings(); return; }

        this.showLoading('Analizando audiograma con Gemini...');
        const prompt = `Analiza esta imagen de un audiograma clínico. Es una gráfica de audiometría tonal que muestra umbrales auditivos.

SIMBOLOGÍA ESTÁNDAR:
- Círculos rojos (O): Vía aérea, oído derecho (OD)
- Cruces azules (X): Vía aérea, oído izquierdo (OI)
- Corchete izquierdo ([) o < : Vía ósea, oído derecho (OD)
- Corchete derecho (]) o > : Vía ósea, oído izquierdo (OI)

EJES:
- Eje X (horizontal): Frecuencias en Hz (125, 250, 500, 1000, 2000, 3000, 4000, 8000)
- Eje Y (vertical): Nivel de audición en dB HL (de -10 arriba a 120 abajo, los valores AUMENTAN hacia abajo)

INSTRUCCIONES: Extrae los valores de decibelios para cada frecuencia y cada combinación oído/vía visibles en la gráfica. Responde EXCLUSIVAMENTE con un JSON válido con esta estructura exacta (usa null para frecuencias sin datos visibles, los valores son números enteros, típicamente múltiplos de 5):

{"va":{"od":{"250":null,"500":null,"1000":null,"2000":null,"3000":null,"4000":null,"8000":null},"oi":{"250":null,"500":null,"1000":null,"2000":null,"3000":null,"4000":null,"8000":null}},"vo":{"od":{"250":null,"500":null,"1000":null,"2000":null,"3000":null,"4000":null,"8000":null},"oi":{"250":null,"500":null,"1000":null,"2000":null,"3000":null,"4000":null,"8000":null}}}`;

        try {
            const base64 = this.preprocessor.getBase64Data();
            const response = await this.api.analyzeImage(base64, 'image/jpeg', prompt);
            const parsed = this.extractJSON(response);
            this.populateFromAI(parsed);
            this.hideLoading();
            document.getElementById('ai-source-badge').style.display = 'inline-flex';
            this.navigateTo('results');
            showToast('¡Audiograma digitalizado con éxito!', 'success');
        } catch (err) {
            this.hideLoading();
            showToast(`Error en análisis IA: ${err.message}`, 'error', 5000);
            console.error('AI Analysis error:', err);
        }
    }

    extractJSON(text) {
        const codeBlock = text.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (codeBlock) return JSON.parse(codeBlock[1].trim());
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) return JSON.parse(jsonMatch[0]);
        throw new Error('No se encontró JSON válido en la respuesta de la IA');
    }

    populateFromAI(data) {
        ['od', 'oi'].forEach(ear => {
            FREQUENCIES.forEach(freq => {
                const vaVal = data?.va?.[ear]?.[String(freq)];
                const voVal = data?.vo?.[ear]?.[String(freq)];
                const vaInput = document.getElementById(`va-${ear}-${freq}`);
                const voInput = document.getElementById(`vo-${ear}-${freq}`);
                if (vaInput) vaInput.value = vaVal !== null && vaVal !== undefined ? vaVal : 0;
                if (voInput) voInput.value = voVal !== null && voVal !== undefined ? voVal : 0;
            });
        });
    }

    // --- Calculations ---
    calculateResults() {
        const tableBody = document.querySelector('#audiometry-table tbody');
        if (!tableBody) return;
        tableBody.innerHTML = '';
        const results = {};

        ['od', 'oi'].forEach(ear => {
            const vaValues = {}, voValues = {};
            FREQUENCIES.forEach(f => {
                vaValues[f] = parseInt(document.getElementById(`va-${ear}-${f}`).value) || 0;
                voValues[f] = parseInt(document.getElementById(`vo-${ear}-${f}`).value) || 0;
            });
            const avgVA = AudiometryCalculator.calculateAverage(vaValues, KEY_FREQUENCIES_FOR_AVG);
            results[ear] = { va: vaValues, vo: voValues, avgVA, diagnosis: AudiometryCalculator.diagnoseHipoacusia(vaValues, voValues) };

            const rowVA = tableBody.insertRow();
            rowVA.innerHTML = `<td>VA ${ear.toUpperCase()}</td>` + FREQUENCIES.map(f => `<td>${vaValues[f] ?? '-'}</td>`).join('');
            const rowVO = tableBody.insertRow();
            rowVO.innerHTML = `<td>VO ${ear.toUpperCase()}</td>` + FREQUENCIES.map(f => `<td>${voValues[f] ?? '-'}</td>`).join('');
            const rowGAP = tableBody.insertRow();
            rowGAP.innerHTML = `<td>GAP ${ear.toUpperCase()}</td>` + FREQUENCIES.map(f => { const g = vaValues[f] - voValues[f]; return `<td>${!isNaN(g) ? g : '-'}</td>`; }).join('');
            const sumVA = KEY_FREQUENCIES_FOR_AVG.reduce((a, f) => a + (vaValues[f] || 0), 0);
            const rowSum = tableBody.insertRow();
            rowSum.innerHTML = `<td>Σ ${ear.toUpperCase()}</td><td colspan="${FREQUENCIES.length}">${sumVA} dB HL (Media VA: ${avgVA.toFixed(1)} dB)</td>`;
        });

        const binauralDef = AudiometryCalculator.calculateBinauralDeficiency(results.od.va, results.oi.va);
        this.currentResults = { ...results, binauralDeficiency: binauralDef };

        AudiogramRenderer.draw(results.od.va, results.od.vo, results.oi.va, results.oi.vo);
        this.renderDiagnosisSummary();
        this.displayReports();
        document.getElementById('calc-results').classList.remove('hidden');
        showToast('Resultados calculados', 'success');
    }

    renderDiagnosisSummary() {
        const { od, oi, binauralDeficiency: bd } = this.currentResults;
        const disabilityPct = AudiometryCalculator.getDisabilityPercentage(bd.binauralDeficiency, od.diagnosis, oi.diagnosis);
        const disabilityClass = AudiometryCalculator.getDisabilityClass(disabilityPct);
        const el = document.getElementById('diagnosis-summary');
        el.innerHTML = `
            <h3>🩺 Resumen Diagnóstico</h3>
            <div class="diagnosis-grid">
                <div class="diagnosis-card od"><h4>Oído Derecho</h4><div class="diagnosis-value">${od.avgVA.toFixed(1)} dB</div><div class="diagnosis-label">${od.diagnosis}</div></div>
                <div class="diagnosis-card oi"><h4>Oído Izquierdo</h4><div class="diagnosis-value">${oi.avgVA.toFixed(1)} dB</div><div class="diagnosis-label">${oi.diagnosis}</div></div>
                <div class="diagnosis-card binaural">
                    <h4>Deficiencia Binaural</h4>
                    <div class="diagnosis-value">${bd.binauralDeficiency.toFixed(2)}%</div>
                    <div class="diagnosis-label">Discapacidad: ${disabilityPct.toFixed(2)}% — ${disabilityClass.class}</div>
                </div>
            </div>`;
    }

    // --- Reports ---
    displayReports() {
        if (!this.currentResults) return;
        document.getElementById('basic-report-content').innerHTML = this.generateBasicReport();
        document.getElementById('detailed-report-content').innerHTML = this.generateDetailedReport();
    }

    generateBasicReport() {
        const { od, oi } = this.currentResults;
        return `<h3>Umbrales Auditivos y Diagnóstico</h3>
            <div class="report-box"><h4>Oído Derecho (OD)</h4>
            <table class="report-data-table"><thead><tr><th>Hz</th>${FREQUENCIES.map(f => `<th>${f}</th>`).join('')}</tr></thead>
            <tbody>${freqRow('VA', od.va)}${freqRow('VO', od.vo)}</tbody></table>
            <p style="margin-top:10px;"><strong>Diagnóstico:</strong> ${od.diagnosis}</p></div>
            <div class="report-box"><h4>Oído Izquierdo (OI)</h4>
            <table class="report-data-table"><thead><tr><th>Hz</th>${FREQUENCIES.map(f => `<th>${f}</th>`).join('')}</tr></thead>
            <tbody>${freqRow('VA', oi.va)}${freqRow('VO', oi.vo)}</tbody></table>
            <p style="margin-top:10px;"><strong>Diagnóstico:</strong> ${oi.diagnosis}</p></div>`;
    }

    generateDetailedReport() {
        const { od, oi, binauralDeficiency: bd } = this.currentResults;
        const disPct = AudiometryCalculator.getDisabilityPercentage(bd.binauralDeficiency, od.diagnosis, oi.diagnosis);
        const disCls = AudiometryCalculator.getDisabilityClass(disPct);
        const useTableB = od.diagnosis === "De Transmisión (Conductiva)" && oi.diagnosis === "De Transmisión (Conductiva)";
        return `
            <h3>1. Datos</h3><ul><li><strong>Nº Historia:</strong> ${document.getElementById('history-number').value || 'N/D'}</li><li><strong>Fecha:</strong> ${document.getElementById('audiometry-date').value || 'N/D'}</li></ul>
            <h3>2. Umbrales Auditivos (dB HL)</h3>
            <div class="report-box"><h4>OD</h4><table class="report-data-table"><thead><tr><th>Hz</th>${FREQUENCIES.map(f => `<th>${f}</th>`).join('')}</tr></thead><tbody>${freqRow('VA', od.va)}${freqRow('VO', od.vo)}</tbody></table><p><strong>Diagnóstico:</strong> ${od.diagnosis}</p></div>
            <div class="report-box"><h4>OI</h4><table class="report-data-table"><thead><tr><th>Hz</th>${FREQUENCIES.map(f => `<th>${f}</th>`).join('')}</tr></thead><tbody>${freqRow('VA', oi.va)}${freqRow('VO', oi.vo)}</tbody></table><p><strong>Diagnóstico:</strong> ${oi.diagnosis}</p></div>
            <h3>3. Deficiencia y Discapacidad (RD 888/2022)</h3>
            <h4>Deficiencias Monoaurales</h4><ul><li><strong>OD:</strong> Suma ${bd.sumOD} dB → ${bd.impairmentOD.toFixed(2)}%</li><li><strong>OI:</strong> Suma ${bd.sumOI} dB → ${bd.impairmentOI.toFixed(2)}%</li></ul>
            <h4>Deficiencia Binaural (ASHA)</h4><ul><li><strong>${bd.binauralDeficiency.toFixed(2)}%</strong></li></ul>
            <h4>Discapacidad</h4><ul><li><strong>${disPct.toFixed(2)}%</strong> (Tabla ${useTableB ? '5.3B' : '5.3A'})</li><li><strong>${disCls.class}</strong> (${disCls.range})</li></ul>
            <h3>4. Observaciones</h3><p>${(document.getElementById('observations').value || 'Sin observaciones.').replace(/\n/g, '<br>')}</p>`;
    }

    // --- AI Report ---
    async generateAIReport() {
        if (!this.currentResults) { showToast('Calcula los resultados primero', 'warning'); return; }
        if (!this.api.isConfigured) { showToast('Configura tu API Key', 'warning'); this.openSettings(); return; }

        this.showLoading('Generando informe clínico con Gemini...');
        const { od, oi, binauralDeficiency: bd } = this.currentResults;
        const disPct = AudiometryCalculator.getDisabilityPercentage(bd.binauralDeficiency, od.diagnosis, oi.diagnosis);
        const disCls = AudiometryCalculator.getDisabilityClass(disPct);

        const prompt = `Eres un asistente médico especialista en audiología. Genera un informe clínico narrativo profesional en español basado en estos datos audiométricos:

DATOS:
- Oído Derecho: VA=${JSON.stringify(od.va)}, VO=${JSON.stringify(od.vo)}, PTA=${od.avgVA.toFixed(1)}dB, Diagnóstico=${od.diagnosis}
- Oído Izquierdo: VA=${JSON.stringify(oi.va)}, VO=${JSON.stringify(oi.vo)}, PTA=${oi.avgVA.toFixed(1)}dB, Diagnóstico=${oi.diagnosis}
- Deficiencia Binaural: ${bd.binauralDeficiency.toFixed(2)}%
- Discapacidad: ${disPct.toFixed(2)}%, ${disCls.class}
- Observaciones: ${document.getElementById('observations').value || 'Ninguna'}

INSTRUCCIONES:
1. Escribe un informe clínico narrativo profesional
2. Incluye: resumen de hallazgos, descripción por oído, valoración binaural, y recomendaciones
3. Menciona que la valoración sigue RD 888/2022 y Orden DSA/934/2023
4. Tono profesional, claro, apto para adjuntar a un informe médico
5. NO incluyas datos del paciente (los añadirá el médico)
6. Escribe en prosa narrativa, no en formato de lista`;

        try {
            const response = await this.api.generateText(prompt);
            const formatted = response.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            document.getElementById('ai-report-content').innerHTML = `<div class="ai-report-text">${formatted}</div>`;
            this.hideLoading();
            this.navigateTo('report');
            showToast('Informe IA generado', 'success');
        } catch (err) {
            this.hideLoading();
            showToast(`Error generando informe: ${err.message}`, 'error', 5000);
        }
    }

    // --- Copy Functions ---
    copyTable() {
        const table = document.getElementById('audiometry-table');
        let text = ''; table.querySelectorAll('tr').forEach(tr => { tr.querySelectorAll('th, td').forEach(td => { text += td.textContent + '\t'; }); text = text.trim() + '\n'; });
        const clone = table.cloneNode(true); clone.style.borderCollapse = 'collapse'; clone.setAttribute('border', '1');
        clone.querySelectorAll('th,td').forEach(c => { c.style.border = '1px solid #999'; c.style.padding = '5px'; });
        copyToClipboard(clone.outerHTML, text).then(ok => showToast(ok ? 'Tabla copiada' : 'Error al copiar', ok ? 'success' : 'error'));
    }

    copyBasicReport() {
        const html = document.getElementById('basic-report-content').innerHTML;
        const text = document.getElementById('basic-report-content').innerText;
        copyToClipboard(html, text).then(ok => showToast(ok ? 'Informe copiado' : 'Error', ok ? 'success' : 'error'));
    }

    copyDetailedReport() {
        const html = document.getElementById('detailed-report-content').innerHTML;
        const text = document.getElementById('detailed-report-content').innerText;
        copyToClipboard(html, text).then(ok => showToast(ok ? 'Informe copiado' : 'Error', ok ? 'success' : 'error'));
    }

    copyAIReport() {
        const html = document.getElementById('ai-report-content').innerHTML;
        const text = document.getElementById('ai-report-content').innerText;
        copyToClipboard(html, text).then(ok => showToast(ok ? 'Informe IA copiado' : 'Error', ok ? 'success' : 'error'));
    }

    copyAudiogram() {
        const container = document.getElementById('audiogram-container');
        const svgEl = container.querySelector('svg');
        if (!svgEl) { showToast('No hay audiograma', 'warning'); return; }
        const svgData = new XMLSerializer().serializeToString(svgEl);
        const canvas = document.createElement('canvas');
        canvas.width = 800; canvas.height = 500;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1a2e'; ctx.fillRect(0, 0, 800, 500);
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            canvas.toBlob(blob => {
                navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
                    .then(() => showToast('Gráfica copiada', 'success'))
                    .catch(() => showToast('Error al copiar', 'error'));
            });
        };
        img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)));
    }

    copyComparisonReport() {
        const el = document.getElementById('comparison-report');
        copyToClipboard(el.innerHTML, el.innerText).then(ok => showToast(ok ? 'Copiado' : 'Error', ok ? 'success' : 'error'));
    }

    // --- Data Management ---
    saveAudiometry() {
        const hn = document.getElementById('history-number').value;
        const dt = document.getElementById('audiometry-date').value;
        if (!hn || !dt) { showToast('Introduce Nº Historia y Fecha', 'error'); return; }
        const data = { id: `${hn}-${dt}-${Date.now()}`, historyNumber: hn, audiometryDate: dt, observations: document.getElementById('observations').value, va: { od: {}, oi: {} }, vo: { od: {}, oi: {} } };
        FREQUENCIES.forEach(f => {
            data.va.od[f] = parseInt(document.getElementById(`va-od-${f}`).value) || 0;
            data.vo.od[f] = parseInt(document.getElementById(`vo-od-${f}`).value) || 0;
            data.va.oi[f] = parseInt(document.getElementById(`va-oi-${f}`).value) || 0;
            data.vo.oi[f] = parseInt(document.getElementById(`vo-oi-${f}`).value) || 0;
        });
        const idx = this.records.findIndex(r => r.historyNumber === hn && r.audiometryDate === dt);
        if (idx > -1) { if (confirm('Ya existe un registro para esta fecha. ¿Sobrescribir?')) this.records[idx] = data; else return; }
        else this.records.push(data);
        this.persistRecords();
        showToast('Audiometría guardada', 'success');
    }

    loadRecords() {
        const stored = localStorage.getItem('audiometryRecords');
        this.records = stored ? JSON.parse(stored) : [];
        this.populateSelects();
    }

    persistRecords() {
        localStorage.setItem('audiometryRecords', JSON.stringify(this.records));
        this.populateSelects();
    }

    populateSelects() {
        const selects = { 'saved-audiometries': 'Seleccionar para cargar...', 'compare-1': 'Seleccionar Basal', 'compare-2': 'Seleccionar Seguimiento' };
        Object.entries(selects).forEach(([id, defaultText]) => {
            const sel = document.getElementById(id);
            if (!sel) return;
            const cv = sel.value;
            sel.innerHTML = `<option value="">${defaultText}</option>`;
            this.records.sort((a, b) => new Date(b.audiometryDate) - new Date(a.audiometryDate)).forEach(r => {
                const opt = document.createElement('option');
                opt.value = r.id; opt.textContent = `Nº ${r.historyNumber} — ${r.audiometryDate}`;
                sel.appendChild(opt);
            });
            sel.value = cv;
        });
    }

    loadSelectedAudiometry() {
        const sel = document.getElementById('saved-audiometries');
        const rec = this.records.find(r => r.id === sel.value);
        if (!rec) { showToast('Selecciona una audiometría', 'info'); return; }
        document.getElementById('history-number').value = rec.historyNumber;
        document.getElementById('audiometry-date').value = rec.audiometryDate;
        document.getElementById('observations').value = rec.observations || '';
        FREQUENCIES.forEach(f => {
            document.getElementById(`va-od-${f}`).value = rec.va.od[f] ?? 0;
            document.getElementById(`vo-od-${f}`).value = rec.vo.od[f] ?? 0;
            document.getElementById(`va-oi-${f}`).value = rec.va.oi[f] ?? 0;
            document.getElementById(`vo-oi-${f}`).value = rec.vo.oi[f] ?? 0;
        });
        this.navigateTo('results');
        this.calculateResults();
        showToast('Audiometría cargada', 'info');
    }

    deleteSelectedAudiometry() {
        const sel = document.getElementById('saved-audiometries');
        if (!sel.value) { showToast('Selecciona una audiometría', 'info'); return; }
        if (confirm('¿Eliminar este registro?')) {
            this.records = this.records.filter(r => r.id !== sel.value);
            this.persistRecords();
            showToast('Eliminado', 'success');
        }
    }

    resetApplication() {
        document.querySelectorAll('input[type="text"], input[type="number"], textarea').forEach(i => i.value = '');
        document.querySelectorAll('input[type="number"]').forEach(i => i.value = '0');
        this.initDate();
        ['audiometry-table', 'audiogram-container', 'basic-report-content', 'detailed-report-content', 'ai-report-content', 'diagnosis-summary'].forEach(id => {
            const el = document.getElementById(id); if (el) { if (el.tagName === 'TABLE') { const tb = el.querySelector('tbody'); if (tb) tb.innerHTML = ''; } else el.innerHTML = ''; }
        });
        document.getElementById('calc-results')?.classList.add('hidden');
        document.getElementById('ai-source-badge').style.display = 'none';
        this.currentResults = null;
        this.changeImage();
        this.navigateTo('capture');
        showToast('Aplicación reiniciada', 'info');
    }

    exportData() {
        if (!this.records.length) { showToast('No hay datos', 'info'); return; }
        const blob = new Blob([JSON.stringify(this.records, null, 2)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `audiometria_backup_${new Date().toISOString().split('T')[0]}.json`;
        a.click(); URL.revokeObjectURL(a.href);
        showToast('Datos exportados', 'success');
    }

    importData(event) {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const imported = JSON.parse(e.target.result);
                if (Array.isArray(imported) && imported.every(r => r.id && r.historyNumber)) {
                    const existing = new Set(this.records.map(r => r.id));
                    const newRecs = imported.filter(r => !existing.has(r.id));
                    this.records.push(...newRecs);
                    this.persistRecords();
                    showToast(`${newRecs.length} registros importados`, 'success');
                } else showToast('Formato incorrecto', 'error');
            } catch { showToast('Error al leer archivo', 'error'); }
        };
        reader.readAsText(file);
        event.target.value = '';
    }

    // --- Comparison ---
    compareAudiometries() {
        const id1 = document.getElementById('compare-1').value;
        const id2 = document.getElementById('compare-2').value;
        if (!id1 || !id2) { showToast('Selecciona dos audiometrías', 'info'); return; }
        const r1 = this.records.find(r => r.id === id1);
        const r2 = this.records.find(r => r.id === id2);
        if (!r1 || !r2) { showToast('Registros no encontrados', 'error'); return; }

        AudiogramRenderer.draw(r1.va.od, r1.vo.od, r1.va.oi, r1.vo.oi, 'comparison-audiogram-1');
        AudiogramRenderer.draw(r2.va.od, r2.vo.od, r2.va.oi, r2.vo.oi, 'comparison-audiogram-2');

        const calc = (rec) => {
            const dOD = AudiometryCalculator.diagnoseHipoacusia(rec.va.od, rec.vo.od);
            const dOI = AudiometryCalculator.diagnoseHipoacusia(rec.va.oi, rec.vo.oi);
            const bd = AudiometryCalculator.calculateBinauralDeficiency(rec.va.od, rec.va.oi);
            const dp = AudiometryCalculator.getDisabilityPercentage(bd.binauralDeficiency, dOD, dOI);
            const dc = AudiometryCalculator.getDisabilityClass(dp);
            return { dOD, dOI, bd, dp, dc };
        };
        const c1 = calc(r1), c2 = calc(r2);
        const diff = c2.dp - c1.dp;
        let trend = diff > 2 ? `Empeoramiento (+${diff.toFixed(2)} puntos)` : diff < -2 ? `Mejora (${diff.toFixed(2)} puntos)` : 'Estable';

        document.getElementById('comparison-report').innerHTML = `
            <h3>Basal (${r1.audiometryDate})</h3><ul><li><strong>Def. Binaural:</strong> ${c1.bd.binauralDeficiency.toFixed(2)}%</li><li><strong>Discapacidad:</strong> ${c1.dp.toFixed(2)}% — ${c1.dc.class}</li></ul>
            <h3>Seguimiento (${r2.audiometryDate})</h3><ul><li><strong>Def. Binaural:</strong> ${c2.bd.binauralDeficiency.toFixed(2)}%</li><li><strong>Discapacidad:</strong> ${c2.dp.toFixed(2)}% — ${c2.dc.class}</li></ul>
            <h3>Evolución</h3><ul><li><strong>Tendencia:</strong> ${trend}</li>${c1.dc.class !== c2.dc.class ? `<li>Cambio de clase: ${c1.dc.class} → ${c2.dc.class}</li>` : `<li>Clase mantenida: ${c2.dc.class}</li>`}</ul>`;
        document.getElementById('comparison-container').classList.remove('hidden');
        showToast('Comparación generada', 'success');
    }

    // --- Settings ---
    openSettings() {
        document.getElementById('settings-modal').classList.remove('hidden');
        // Load saved values into UI
        document.getElementById('backend-select').value = this.api.backend;
        document.getElementById('api-key-input').value = this.api.apiKey;
        document.getElementById('model-select').value = this.api.model;
        document.getElementById('ollama-url').value = this.api.ollamaUrl;
        document.getElementById('ollama-model').value = this.api.ollamaModel;
        document.getElementById('custom-url').value = this.api.customUrl;
        document.getElementById('custom-api-key').value = this.api.customApiKey;
        document.getElementById('custom-model-name').value = this.api.customModel;
        this.onBackendChange();
    }
    closeSettings() { document.getElementById('settings-modal').classList.add('hidden'); }

    onBackendChange() {
        const backend = document.getElementById('backend-select').value;
        document.querySelectorAll('.backend-settings').forEach(el => el.classList.add('hidden'));
        document.getElementById(`settings-${backend}`)?.classList.remove('hidden');
    }

    saveSettings() {
        const backend = document.getElementById('backend-select').value;
        localStorage.setItem('ai_backend', backend);
        this.api.backend = backend;

        // Cloud
        const key = document.getElementById('api-key-input').value.trim();
        const model = document.getElementById('model-select').value;
        localStorage.setItem('gemma_api_key', key);
        localStorage.setItem('gemma_model', model);
        this.api.apiKey = key;
        this.api.model = model;

        // Ollama
        const ollamaUrl = document.getElementById('ollama-url').value.trim();
        const ollamaModel = document.getElementById('ollama-model').value.trim();
        localStorage.setItem('ollama_url', ollamaUrl);
        localStorage.setItem('ollama_model', ollamaModel);
        this.api.ollamaUrl = ollamaUrl;
        this.api.ollamaModel = ollamaModel;

        // Custom
        const customUrl = document.getElementById('custom-url').value.trim();
        const customApiKey = document.getElementById('custom-api-key').value.trim();
        const customModelName = document.getElementById('custom-model-name').value.trim();
        localStorage.setItem('custom_url', customUrl);
        localStorage.setItem('custom_api_key', customApiKey);
        localStorage.setItem('custom_model_name', customModelName);
        this.api.customUrl = customUrl;
        this.api.customApiKey = customApiKey;
        this.api.customModel = customModelName;

        // On-device
        const ondeviceModelUrl = document.getElementById('ondevice-model-url').value.trim();
        localStorage.setItem('ondevice_model_url', ondeviceModelUrl);
        this.api.onDeviceModelUrl = ondeviceModelUrl;

        this.updateAPIStatus();
        this.closeSettings();
        showToast('Configuración guardada', 'success');
    }

    async testAPIConnection() {
        this.saveSettings();
        if (!this.api.isConfigured) {
            if (this.api.backend === 'ondevice' && !this.api.onDeviceReady) {
                showToast('Primero carga el modelo con "Descargar y Cargar Modelo"', 'warning');
            } else {
                showToast('Completa la configuración del backend', 'warning');
            }
            return;
        }
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        try {
            this.openSettings();
            dot.className = 'status-dot'; text.textContent = 'Probando conexión...';
            await this.api.generateText('Responde solo "OK".');
            dot.className = 'status-dot connected';
            text.textContent = `✅ Conectado — ${this.api.backendLabel}`;
            showToast('¡Conexión exitosa!', 'success');
        } catch (err) {
            dot.className = 'status-dot error';
            text.textContent = `❌ ${err.message.substring(0, 80)}`;
            showToast('Error de conexión', 'error', 5000);
        }
    }

    updateAPIStatus() {
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');
        if (this.api.isConfigured) { dot.className = 'status-dot connected'; text.textContent = this.api.backendLabel; }
        else { dot.className = 'status-dot'; text.textContent = 'No configurado'; }
    }

    toggleKeyVisibility() {
        const input = document.getElementById('api-key-input');
        input.type = input.type === 'password' ? 'text' : 'password';
    }

    // --- On-Device Model Loading ---
    async loadOnDeviceModel() {
        const modelUrl = document.getElementById('ondevice-model-url').value.trim();
        if (!modelUrl) { showToast('Introduce la URL del modelo .litertlm', 'warning'); return; }

        // Save URL
        localStorage.setItem('ondevice_model_url', modelUrl);
        this.api.onDeviceModelUrl = modelUrl;

        // Show progress
        const statusEl = document.getElementById('ondevice-status');
        const statusText = document.getElementById('ondevice-status-text');
        const progressBar = document.getElementById('ondevice-progress');
        statusEl.style.display = 'block';
        statusText.textContent = '⏳ Descargando y compilando modelo... Esto puede tardar varios minutos la primera vez.';
        progressBar.style.width = '10%';

        try {
            progressBar.style.width = '30%';
            statusText.textContent = '📦 Inicializando MediaPipe y descargando modelo...';

            await this.api.loadModel(modelUrl);

            progressBar.style.width = '100%';
            statusText.textContent = '✅ ¡Modelo cargado! Listo para inferencia on-device.';

            // Update backend
            this.api.backend = 'ondevice';
            localStorage.setItem('ai_backend', 'ondevice');
            document.getElementById('backend-select').value = 'ondevice';
            this.updateAPIStatus();
            showToast('¡Modelo cargado en el dispositivo!', 'success');
        } catch (err) {
            progressBar.style.width = '0%';
            statusText.textContent = `❌ Error: ${err.message}`;
            showToast(`Error cargando modelo: ${err.message}`, 'error', 5000);
            console.error('On-device model load error:', err);
        }
    }

    // --- Print ---
    printReport() { document.getElementById('print-modal').classList.remove('hidden'); }
    closePrintModal() { document.getElementById('print-modal').classList.add('hidden'); }
    executePrint() { this.closePrintModal(); window.print(); }

    // --- Loading ---
    showLoading(text = 'Procesando...') { document.getElementById('loading-text').textContent = text; document.getElementById('loading-overlay').classList.remove('hidden'); }
    hideLoading() { document.getElementById('loading-overlay').classList.add('hidden'); }
}

// ===== INIT =====
const app = new AppController();
document.addEventListener('DOMContentLoaded', () => app.init());