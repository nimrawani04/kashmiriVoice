// Kashmiri Voice (کٲشُر آواز) - Interactive Web Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // Tab Switching
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.target);
      if (target) target.classList.add("active");
    });
  });

  // Kashmiri Virtual Keyboard Key Insertion
  const kashmiriInput = document.getElementById("kashmiri-text");
  document.querySelectorAll(".kb-key").forEach(key => {
    key.addEventListener("click", () => {
      const char = key.textContent.trim();
      const start = kashmiriInput.selectionStart;
      const end = kashmiriInput.selectionEnd;
      const val = kashmiriInput.value;
      kashmiriInput.value = val.substring(0, start) + char + val.substring(end);
      kashmiriInput.focus();
      kashmiriInput.selectionStart = kashmiriInput.selectionEnd = start + char.length;
    });
  });

  // Dialect Card Selection
  const dialectCards = document.querySelectorAll(".dialect-card");
  let selectedDialect = "kupwara";
  dialectCards.forEach(card => {
    card.addEventListener("click", () => {
      dialectCards.forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedDialect = card.dataset.dialect;
    });
  });

  // Benchmark Sentence Preset Selector
  const presetSelect = document.getElementById("preset-select");
  presetSelect.addEventListener("change", (e) => {
    if (e.target.value) {
      kashmiriInput.value = e.target.value;
    }
  });

  // Audio Visualizer Setup
  const audioPlayer = document.getElementById("tts-audio-player");
  const canvas = document.getElementById("visualizer-canvas");
  const canvasCtx = canvas.getContext("2d");
  let audioCtx, analyser, sourceNode;

  function initVisualizer() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      sourceNode = audioCtx.createMediaElementSource(audioPlayer);
      sourceNode.connect(analyser);
      analyser.connect(audioCtx.destination);
    }
  }

  function drawWaveform() {
    requestAnimationFrame(drawWaveform);
    if (!analyser) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = "rgba(9, 13, 22, 0.3)";
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;
      
      const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, "#388bfd");
      gradient.addColorStop(0.5, "#a371f7");
      gradient.addColorStop(1, "#f778ba");

      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
      x += barWidth;
    }
  }

  audioPlayer.addEventListener("play", () => {
    initVisualizer();
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }
    drawWaveform();
  });

  // TTS Synthesis Trigger
  const synthBtn = document.getElementById("synth-btn");
  const synthStatus = document.getElementById("synth-status");
  const downloadLink = document.getElementById("download-audio");

  synthBtn.addEventListener("click", async () => {
    const text = kashmiriInput.value.trim();
    if (!text) {
      alert("Please enter Kashmiri text in Perso-Arabic script!");
      return;
    }

    synthBtn.disabled = true;
    synthBtn.innerHTML = `<span>⏳ Synthesizing Voice...</span>`;
    synthStatus.textContent = "Processing Kashmiri phonetics and neural spectrogram...";

    try {
      const resp = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          dialect: selectedDialect
        })
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const data = await resp.json();
      audioPlayer.src = data.audio_url + "?t=" + Date.now();
      audioPlayer.hidden = false;
      audioPlayer.play();

      downloadLink.href = data.audio_url;
      downloadLink.hidden = false;

      synthStatus.innerHTML = `✨ Synthesized <strong>${data.duration.toFixed(2)}s</strong> in <em>${data.dialect_name} (${data.dialect_zone})</em>`;
    } catch (err) {
      synthStatus.textContent = "Error: " + err.message;
    } finally {
      synthBtn.disabled = false;
      synthBtn.innerHTML = `<span>🎙️ Synthesize Kashmiri Speech</span>`;
    }
  });

  // Accent Classification Drag & Drop / Upload
  const dropzone = document.getElementById("dropzone");
  const audioFileInput = document.getElementById("audio-file-input");
  const classifyBtn = document.getElementById("classify-btn");
  const classifyResult = document.getElementById("classify-result");
  let selectedFile = null;

  dropzone.addEventListener("click", () => audioFileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleAudioSelection(e.dataTransfer.files[0]);
    }
  });

  audioFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleAudioSelection(e.target.files[0]);
    }
  });

  function handleAudioSelection(file) {
    if (!file.name.match(/\.(wav|mp3|ogg|flac)$/i)) {
      alert("Please select a valid audio file (WAV, MP3, OGG)!");
      return;
    }
    selectedFile = file;
    document.getElementById("file-name-display").textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    classifyBtn.disabled = false;
  }

  classifyBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    classifyBtn.disabled = true;
    classifyBtn.innerHTML = `<span>⏳ Extracting MFCCs & Classifying Accent...</span>`;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const resp = await fetch("/api/classify", {
        method: "POST",
        body: formData
      });

      if (!resp.ok) throw new Error(await resp.text());
      const res = await resp.json();

      document.getElementById("pred-district").textContent = res.predicted_dialect;
      document.getElementById("pred-zone").textContent = res.dialect_zone;
      document.getElementById("pred-conf").textContent = (res.confidence * 100).toFixed(1) + "%";

      // Populate probability bars
      const probsContainer = document.getElementById("probs-container");
      probsContainer.innerHTML = "";
      for (const [dist, prob] of Object.entries(res.probabilities)) {
        const percent = (prob * 100).toFixed(1);
        probsContainer.innerHTML += `
          <div class="prob-bar">
            <div class="prob-header">
              <span>${dist}</span>
              <span>${percent}%</span>
            </div>
            <div class="prob-track">
              <div class="prob-fill" style="width: ${percent}%"></div>
            </div>
          </div>
        `;
      }

      classifyResult.classList.add("active");
    } catch (err) {
      alert("Classification failed: " + err.message);
    } finally {
      classifyBtn.disabled = false;
      classifyBtn.innerHTML = `<span>🎯 Identify Kashmiri Accent</span>`;
    }
  });
});
