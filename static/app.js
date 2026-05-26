const chat = document.getElementById('chat');
const txt = document.getElementById('txt');
const btn = document.getElementById('btn');
const fileIn = document.getElementById('file-in');
const fname = document.getElementById('fname');
const preview = document.getElementById('preview');
const micBtn = document.getElementById('mic-btn');
const empty = document.getElementById('empty');
const inputArea = document.getElementById('input-area');

let currentFile = null;
let mediaRecorder = null;
let audioChunks = [];
let isVoiceMode = false;
let silenceTimer = null;
let vadInterval = null;
const SILENCE_TIMEOUT = 2000;

const SESSION_ID = localStorage.getItem('lightvlm_session') || crypto.randomUUID().slice(0, 8);
localStorage.setItem('lightvlm_session', SESSION_ID);

// Voice overlay
const voiceOverlay = document.createElement('div');
voiceOverlay.id = 'voice-overlay';
voiceOverlay.innerHTML = `
    <div class="voice-avatar">LV</div>
    <div class="voice-status">Tap microphone to speak</div>
    <div class="voice-waves">
        <span></span><span></span><span></span><span></span><span></span>
    </div>
    <button class="voice-close">✕</button>
`;
document.body.appendChild(voiceOverlay);

voiceOverlay.querySelector('.voice-close').onclick = () => toggleVoiceMode();

function toggleVoiceMode() {
    isVoiceMode = !isVoiceMode;
    if (isVoiceMode) {
        voiceOverlay.classList.add('active');
        inputArea.style.display = 'none';
        chat.style.display = 'none';
        document.querySelector('header').style.display = 'none';
        toggleRecording();
    } else {
        voiceOverlay.classList.remove('active');
        inputArea.style.display = '';
        chat.style.display = '';
        document.querySelector('header').style.display = '';
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
    }
}

let clickTimer = null;
micBtn.addEventListener('click', function(e) {
    if (clickTimer) {
        clearTimeout(clickTimer);
        clickTimer = null;
        toggleVoiceMode();
    } else {
        clickTimer = setTimeout(() => {
            clickTimer = null;
            toggleRecording();
        }, 250);
    }
});

txt.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

txt.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
    }
});

// File input: images go to vision chat, PDF/text go to upload
fileIn.addEventListener('change', function () {
    const f = this.files[0];
    if (!f) return;

    if (f.type.startsWith('image/')) {
        currentFile = f;
        fname.textContent = f.name;
        const r = new FileReader();
        r.onload = e => {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };
        r.readAsDataURL(f);
    } else {
        uploadFile(f);
        this.value = '';
    }
});

async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        if (silenceTimer) clearTimeout(silenceTimer);
        if (vadInterval) clearInterval(vadInterval);
        mediaRecorder.stop();
        return;
    }

    try {
        audioChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        let lastSoundTime = Date.now();

        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
                if (e.data.size > 50) {
                    lastSoundTime = Date.now();
                }
            }
        };

        vadInterval = setInterval(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                const silence = Date.now() - lastSoundTime;
                if (silence > SILENCE_TIMEOUT) {
                    clearInterval(vadInterval);
                    vadInterval = null;
                    mediaRecorder.stop();
                }
            } else {
                clearInterval(vadInterval);
                vadInterval = null;
            }
        }, 200);

        mediaRecorder.onstop = async () => {
            if (silenceTimer) clearTimeout(silenceTimer);
            if (vadInterval) clearInterval(vadInterval);
            vadInterval = null;
            stream.getTracks().forEach(t => t.stop());

            micBtn.classList.remove('recording');

            if (audioChunks.length === 0) {
                if (isVoiceMode) {
                    voiceOverlay.querySelector('.voice-status').textContent = 'No speech detected, tap to speak';
                    voiceOverlay.querySelector('.voice-waves').classList.remove('active');
                }
                return;
            }

            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'voice.webm');

            if (isVoiceMode) {
                voiceOverlay.querySelector('.voice-status').textContent = 'Processing...';
                voiceOverlay.querySelector('.voice-waves').classList.remove('active');
            } else {
                txt.placeholder = 'Transcribing...';
            }

            try {
                const resp = await fetch('/transcribe', { method: 'POST', body: formData });
                const data = await resp.json();

                txt.value = data.text || '';
                txt.dispatchEvent(new Event('input'));
                txt.placeholder = 'Message LightVLM...';

                if (data.text && data.text.trim()) {
                    if (isVoiceMode) {
                        voiceOverlay.querySelector('.voice-status').textContent = data.text;
                    }
                    await send(true);
                } else {
                    if (isVoiceMode) {
                        voiceOverlay.querySelector('.voice-status').textContent = 'No speech detected, tap to speak';
                        voiceOverlay.querySelector('.voice-waves').classList.remove('active');
                    }
                }
            } catch (err) {
                if (isVoiceMode) {
                    voiceOverlay.querySelector('.voice-status').textContent = 'Error, try again';
                    voiceOverlay.querySelector('.voice-waves').classList.remove('active');
                } else {
                    txt.placeholder = 'Transcription failed';
                }
                console.error(err);
            }
        };

        mediaRecorder.start(500);

        if (isVoiceMode) {
            voiceOverlay.querySelector('.voice-status').textContent = 'Listening...';
            voiceOverlay.querySelector('.voice-waves').classList.add('active');
        } else {
            micBtn.classList.add('recording');
            txt.placeholder = 'Recording... (click mic to stop)';
        }

        silenceTimer = setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, 10000);

    } catch (err) {
        alert('Microphone access denied or not available');
        console.error(err);
    }
}

function speakText(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();

    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-US';
    u.rate = 1.0;
    u.pitch = 1.0;

    const voices = speechSynthesis.getVoices();
    const voice = voices.find(v =>
        v.lang.includes('en') &&
        (v.name.includes('Samantha') || v.name.includes('Daniel') || v.name.includes('Google'))
    ) || voices.find(v => v.lang.includes('en'));

    if (voice) u.voice = voice;

    u.onstart = () => {
        if (isVoiceMode) {
            voiceOverlay.querySelector('.voice-status').textContent = 'Speaking...';
            voiceOverlay.querySelector('.voice-waves').classList.add('active');
        }
    };
    u.onend = () => {
        if (isVoiceMode) {
            voiceOverlay.querySelector('.voice-status').textContent = 'Tap microphone to speak';
            voiceOverlay.querySelector('.voice-waves').classList.remove('active');
            setTimeout(() => toggleRecording(), 600);
        }
    };

    window.speechSynthesis.speak(u);
}

function timeString() {
    return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function fixSpacing(text) {
    return text
        .replace(/([.!?])([A-Z])/g, '$1 $2')
        .replace(/([.!?])([a-z])/g, '$1 $2')
        .replace(/([,;:])([A-Za-z])/g, '$1 $2')
        .replace(/([.!?])(\d)/g, '$1 $2')
        .replace(/ {2,}/g, ' ')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

function cleanMarkdown(text) {
    return text
        .replace(/###\s*(Bot|User):\s*/g, '')
        .replace(/^(Bot|User):\s*/gi, '')
        .replace(/^#{1,6}\s+/gm, '')
        .trim();
}

function formatText(text) {
    let cleaned = cleanMarkdown(text);
    cleaned = fixSpacing(cleaned);
    return cleaned
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function addMsg(role, text, imgUrl = null, isStreaming = false, generatedImage = null) {
    if (empty) empty.style.display = 'none';

    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg ' + role;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'You' : 'LV';
    msgDiv.appendChild(avatar);

    const content = document.createElement('div');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (isStreaming) {
        bubble.innerHTML = '<span class="typing"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span>';
    } else if (generatedImage) {
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + generatedImage;
        img.className = 'img-preview';
        img.style.maxWidth = '512px';
        bubble.appendChild(img);

        const caption = document.createElement('div');
        caption.textContent = text;
        caption.style.marginTop = '10px';
        caption.style.fontSize = '14px';
        caption.style.color = 'var(--text-muted)';
        bubble.appendChild(caption);

        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'tts-btn';
        downloadBtn.innerHTML = '⬇️';
        downloadBtn.title = 'Download image';
        downloadBtn.style.marginTop = '8px';
        downloadBtn.style.display = 'block';
        downloadBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = 'data:image/png;base64,' + generatedImage;
            link.download = 'lightvlm-' + Date.now() + '.png';
            link.click();
        };
        bubble.appendChild(downloadBtn);
    } else {
        bubble.innerHTML = formatText(text);
    }
    content.appendChild(bubble);

    if (imgUrl && !generatedImage) {
        const img = document.createElement('img');
        img.src = imgUrl;
        img.className = 'img-preview';
        bubble.appendChild(img);
    }

    if (role === 'bot' && !isStreaming && !generatedImage) {
        const ttsBtn = document.createElement('button');
        ttsBtn.className = 'tts-btn';
        ttsBtn.innerHTML = '🔊';
        ttsBtn.title = 'Read aloud';
        ttsBtn.onclick = () => speakText(text);
        bubble.appendChild(document.createTextNode(' '));
        bubble.appendChild(ttsBtn);
    }

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = timeString();
    content.appendChild(meta);

    msgDiv.appendChild(content);
    chat.appendChild(msgDiv);
    chat.scrollTop = chat.scrollHeight;

    return { msgDiv, bubble };
}

async function send(autoPlayVoice = false) {
    const text = txt.value.trim();
    if (!text && !currentFile) return;

    // If we have a document loaded but no active file input, use the text as question
    const imgUrl = preview.style.display !== 'none' ? preview.src : null;
    addMsg('user', text, imgUrl);
    txt.value = '';
    txt.style.height = 'auto';
    fileIn.value = '';
    fname.textContent = '';
    preview.style.display = 'none';

    const botObj = addMsg('bot', '', null, true);
    btn.disabled = true;

    const fd = new FormData();
    if (text) fd.append('text', text);
    if (currentFile) fd.append('image', currentFile);
    currentFile = null;

    try {
        const res = await fetch('/stream', {
            method: 'POST',
            body: fd,
            headers: { 'X-Session-Id': SESSION_ID }
        });

        const newSession = res.headers.get('X-Session-Id');
        if (newSession && newSession !== SESSION_ID) {
            localStorage.setItem('lightvlm_session', newSession);
        }

        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let fullText = '';
        let isGenerated = false;
        let generatedImage = null;

        const contentSpan = document.createElement('span');
        botObj.bubble.innerHTML = '';
        botObj.bubble.appendChild(contentSpan);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += dec.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6).trim();
                if (data === '[DONE]') continue;

                try {
                    const j = JSON.parse(data);
                    if (j.token) {
                        fullText += j.token;

                        if (j.generated && j.image) {
                            isGenerated = true;
                            generatedImage = j.image;
                        } else {
                            contentSpan.textContent = fixSpacing(fullText);
                            chat.scrollTop = chat.scrollHeight;
                        }
                    }
                } catch (e) {}
            }
        }

        botObj.msgDiv.remove();

        if (isGenerated && generatedImage) {
            addMsg('bot', fullText, null, false, generatedImage);
            if (autoPlayVoice) speakText(fullText);
        } else {
            addMsg('bot', fullText);
            if (autoPlayVoice) speakText(fullText);
        }

    } catch (err) {
        botObj.bubble.innerHTML = '<span style="color:#ef4444">Error: ' + err.message + '</span>';
    } finally {
        btn.disabled = false;
    }
}

window.speechSynthesis.getVoices();
window.speechSynthesis.onvoiceschanged = () => {
    window.speechSynthesis.getVoices();
};

// ===== DRAG & DROP / PASTE / UPLOAD =====

const dropZone = document.getElementById('chat');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
    dropZone.addEventListener(e, (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(e => {
    dropZone.addEventListener(e, () => dropZone.classList.add('drag-active'), false);
});

['dragleave', 'drop'].forEach(e => {
    dropZone.addEventListener(e, () => dropZone.classList.remove('drag-active'), false);
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) uploadFile(files[0]);
});

document.addEventListener('paste', (e) => {
    const items = e.clipboardData.items;
    for (let item of items) {
        if (item.kind === 'file') {
            uploadFile(item.getAsFile());
            break;
        }
    }
});

async function uploadFile(file) {
    addMsg('user', `📄 Uploading: ${file.name}...`);

    const fd = new FormData();
    fd.append('file', file);
    fd.append('mode', 'auto');

    const question = txt.value.trim();
    if (question) {
        fd.append('ask_question', question);
        txt.value = '';
        txt.style.height = 'auto';
    }

    try {
            const resp = await fetch('/upload', {
            method: 'POST',
            body: fd,
            headers: { 'X-Session-Id': SESSION_ID }
        });
        const data = await resp.json();

        if (data.error) {
            addMsg('bot', `❌ Error: ${data.error}`);
            return;
        }

        let summary = `📄 **${file.name}**\n`;
        if (data.pages) summary += `Pages: ${data.pages}\n`;
        if (data.scanned_pages) summary += `OCR pages: ${data.scanned_pages}\n`;
        if (data.truncated) summary += `⚠️ Truncated\n`;

        if (data.answer) {
            summary += `\n**Q:** ${data.question}\n**A:** ${data.answer}`;
        } else if (data.text) {
            const preview = data.text.substring(0, 500) + (data.text.length > 500 ? '...' : '');
            summary += `\n\`\`\`\n${preview}\n\`\`\``;
        } else {
            summary += `\nNo text extracted.`;
        }

        addMsg('bot', summary);

        if (!question && data.text) {
            txt.value = `Summarize this: ${file.name}`;
            txt.dispatchEvent(new Event('input'));
        }

    } catch (err) {
        addMsg('bot', `❌ Upload failed: ${err.message}`);
    }
}