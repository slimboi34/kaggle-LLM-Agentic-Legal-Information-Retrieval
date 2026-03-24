/* ═══════════════════════════════════════
   JARVIS — Frontend Logic
   Particle system, typewriter, API integration
   ═══════════════════════════════════════ */

// ── Particle Background ──────────────────────────────────
(function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    const PARTICLE_COUNT = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    class Particle {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.3;
            this.vy = (Math.random() - 0.5) * 0.3;
            this.radius = Math.random() * 1.5 + 0.5;
            this.opacity = Math.random() * 0.5 + 0.1;
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 229, 255, ${this.opacity})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(new Particle());

    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 229, 255, ${0.08 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => { p.update(); p.draw(); });
        drawConnections();
        requestAnimationFrame(animate);
    }
    animate();
})();

// ── Loading Messages ─────────────────────────────────────
const LOADING_MESSAGES = [
    "Scanning Swiss Federal Archives...",
    "Cross-referencing BGE decisions...",
    "Analyzing statutory provisions...",
    "Mapping precedent chains...",
    "Evaluating legislative context...",
    "Consulting the Bundesgericht...",
];

// ── DOM References ───────────────────────────────────────
const chatContainer = document.getElementById('chat-container');
const queryInput = document.getElementById('query-input');
const sendBtn = document.getElementById('send-btn');

// ── Event Listeners ──────────────────────────────────────
sendBtn.addEventListener('click', sendQuery);
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
    }
});

function askSuggestion(el) {
    queryInput.value = el.textContent;
    sendQuery();
}

// ── Send Query ───────────────────────────────────────────
async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Remove welcome hero
    const hero = document.getElementById('welcome-hero');
    if (hero) hero.remove();

    // Add user message
    addMessage(query, 'user');
    queryInput.value = '';

    // Show loading
    const loadingId = showLoading();

    // Cycle loading messages
    let msgIdx = 0;
    const loadingInterval = setInterval(() => {
        const textEl = document.querySelector(`#${loadingId} .loading-text`);
        if (textEl) {
            msgIdx = (msgIdx + 1) % LOADING_MESSAGES.length;
            textEl.textContent = LOADING_MESSAGES[msgIdx];
        }
    }, 2000);

    try {
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 10 })
        });

        clearInterval(loadingInterval);
        removeElement(loadingId);

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        // Build Jarvis response
        const analysis = data.analysis || "Here are the most relevant Swiss legal authorities for your query:";
        const citations = data.citations || [];

        addJarvisResponse(analysis, citations);

    } catch (err) {
        clearInterval(loadingInterval);
        removeElement(loadingId);
        console.error(err);
        addJarvisResponse(
            "I apologize — I encountered a technical difficulty accessing the legal database. Please try again.",
            []
        );
    }
}

// ── Add User Message ─────────────────────────────────────
function addMessage(text, sender) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.textContent = text;
    chatContainer.appendChild(div);
    scrollToBottom();
}

// ── Add Jarvis Response with Typewriter ──────────────────
function addJarvisResponse(analysis, citations) {
    const div = document.createElement('div');
    div.className = 'message jarvis';

    const label = document.createElement('span');
    label.className = 'msg-label';
    label.textContent = '⚡ jarvis analysis';
    div.appendChild(label);

    const body = document.createElement('div');
    body.className = 'msg-body';
    div.appendChild(body);

    chatContainer.appendChild(div);
    scrollToBottom();

    // Typewriter effect
    typeWriter(body, analysis, 12, () => {
        // After typewriter, show citations
        if (citations.length > 0) {
            const container = document.createElement('div');
            container.className = 'citations-container';
            citations.forEach((cit, i) => {
                const chip = document.createElement('span');
                chip.className = 'citation-chip';
                const isLaw = cit.startsWith('Art.');
                chip.innerHTML = `<span class="chip-icon">${isLaw ? '§' : '⚖'}</span>${cit}`;
                chip.style.animationDelay = `${i * 0.05}s`;
                chip.style.animation = 'msgSlide 0.3s ease forwards';
                chip.style.opacity = '0';
                container.appendChild(chip);
            });
            div.appendChild(container);
            scrollToBottom();
        }
    });
}

// ── Typewriter Effect ────────────────────────────────────
function typeWriter(element, text, speed, callback) {
    let idx = 0;
    function type() {
        if (idx < text.length) {
            element.textContent += text.charAt(idx);
            idx++;
            scrollToBottom();
            setTimeout(type, speed);
        } else if (callback) {
            callback();
        }
    }
    type();
}

// ── Loading Indicator ────────────────────────────────────
function showLoading() {
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'loading-indicator';
    div.innerHTML = `
        <div class="loading-dots"><span></span><span></span><span></span></div>
        <span class="loading-text">${LOADING_MESSAGES[0]}</span>
    `;
    chatContainer.appendChild(div);
    scrollToBottom();
    return id;
}

// ── Utilities ────────────────────────────────────────────
function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
