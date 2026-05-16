/* =========================================================
   Kazakh ASR – Frontend JS v3.0
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {

  /* ─── Helpers ─── */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  function formatTime(sec) {
    sec = parseFloat(sec) || 0;
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  /* ─── Drop zone (index page) ─── */
  const dropZone = document.getElementById('dropZoneModern');
  const fileInput = document.getElementById('fileInput');
  const dropContent = document.getElementById('dropZoneContent');

  if (dropZone && fileInput) {
    ['dragenter', 'dragover'].forEach(ev => {
      dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(ev => {
      dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    });
    dropZone.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files.length) { fileInput.files = files; updateDropLabel(files[0]); }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) updateDropLabel(fileInput.files[0]);
    });

    function updateDropLabel(file) {
      if (!dropContent) return;
      const sizeMb = (file.size / 1024 / 1024).toFixed(1);
      dropContent.innerHTML = `
        <div class="drop-zone-icon">✅</div>
        <p class="drop-zone-primary">${escapeHtml(file.name)}</p>
        <p class="drop-zone-secondary">${sizeMb} МБ</p>
      `;
    }
  }

  /* ─── Form submit: show loading overlay ─── */
  const form = document.getElementById('transcribeForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const submitBtn = document.getElementById('submitBtn');

  if (form) {
    form.addEventListener('submit', e => {
      const checked = form.querySelectorAll('input[name="models"]:checked');
      if (checked.length === 0) {
        e.preventDefault();
        alert('Кемінде бір модель таңдаңыз.');
        return;
      }
      if (loadingOverlay) loadingOverlay.style.display = 'flex';
      if (submitBtn) {
        submitBtn.disabled = true;
        const textEl = submitBtn.querySelector('.btn-submit-text');
        if (textEl) textEl.textContent = 'Өңделуде…';
      }
      // Animate loading steps
      const steps = document.querySelectorAll('.loading-step');
      if (steps.length) {
        let i = 0;
        const interval = setInterval(() => {
          steps.forEach((s, idx) => {
            s.classList.toggle('active', idx === i);
            if (idx < i) s.classList.add('done');
          });
          i = (i + 1) % steps.length;
          if (i === 0) clearInterval(interval);
        }, 3000);
      }
    });
  }

  /* ─── Copy button (results page) ─── */
  const copyBtn = document.getElementById('copyIdealBtn');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const targetId = copyBtn.dataset.target;
      const target = document.getElementById(targetId);
      if (!target) return;
      const text = target.innerText || target.textContent;
      navigator.clipboard.writeText(text).then(() => {
        const label = copyBtn.querySelector('.copy-label');
        copyBtn.classList.add('copied');
        if (label) label.textContent = 'Көшірілді!';
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          if (label) label.textContent = 'Көшіру';
        }, 2000);
      }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        const label = copyBtn.querySelector('.copy-label');
        copyBtn.classList.add('copied');
        if (label) label.textContent = 'Көшірілді!';
        setTimeout(() => {
          copyBtn.classList.remove('copied');
          if (label) label.textContent = 'Көшіру';
        }, 2000);
      });
    });
  }

  /* ─── Text tab switcher (results page) ─── */
  document.querySelectorAll('.text-toggle-tabs').forEach(tabsEl => {
    tabsEl.querySelectorAll('.text-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        tabsEl.querySelectorAll('.text-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const targetId = tab.dataset.tab;
        const col = tabsEl.closest('.transcript-col');
        if (!col) return;
        col.querySelectorAll('.segments-list').forEach(sl => {
          sl.style.display = sl.id === targetId ? '' : 'none';
        });
      });
    });
  });

  /* ─── Time-Series Accuracy Chart (results page) ─── */
  const chartCanvas = document.getElementById('accuracyChart');
  if (chartCanvas && typeof Chart !== 'undefined' && typeof CHART_DATA !== 'undefined') {
    buildAccuracyChart(chartCanvas, CHART_DATA);
  }

  function buildAccuracyChart(canvas, results) {
    // Color palette for up to 3 models
    const colors = [
      { line: '#1565C0', fill: 'rgba(21,101,192,.08)' },
      { line: '#6366f1', fill: 'rgba(99,102,241,.08)' },
      { line: '#0f766e', fill: 'rgba(15,118,110,.08)' },
    ];

    const datasets = [];
    results.forEach((r, idx) => {
      const segs = r.segments || [];
      if (!segs.length) return;
      const color = colors[idx % colors.length];

      // Build confidence time series
      const confData = segs.map(s => ({
        x: ((parseFloat(s.start) + parseFloat(s.end)) / 2).toFixed(2),
        y: (s.confidence != null) ? Math.round(s.confidence * 100) : null,
      })).filter(pt => pt.y !== null);

      datasets.push({
        label: `${r.model_name} — Сенімділік`,
        data: confData,
        borderColor: color.line,
        backgroundColor: color.fill,
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        fill: true,
        tension: 0.3,
      });
    });

    if (!datasets.length) {
      canvas.parentElement.innerHTML = '<p style="text-align:center;padding:2rem;color:#94a3b8;font-size:.875rem;">Графикке арналған деректер жоқ</p>';
      return;
    }

    new Chart(canvas, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: {
            type: 'linear',
            title: {
              display: true,
              text: 'Уақыт (секунд)',
              color: '#64748b',
              font: { size: 12, family: 'Inter, sans-serif' },
            },
            ticks: {
              color: '#94a3b8',
              callback: v => formatTime(v),
            },
            grid: { color: '#f1f5f9' },
          },
          y: {
            title: {
              display: true,
              text: 'Сенімділік (%)',
              color: '#64748b',
              font: { size: 12, family: 'Inter, sans-serif' },
            },
            min: 0,
            max: 100,
            ticks: {
              color: '#94a3b8',
              callback: v => v + '%',
            },
            grid: { color: '#f1f5f9' },
          },
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#334155',
              font: { size: 12, family: 'Inter, sans-serif' },
              usePointStyle: true,
              pointStyleWidth: 10,
            },
          },
          tooltip: {
            backgroundColor: '#0d2347',
            titleColor: '#fff',
            bodyColor: 'rgba(255,255,255,.8)',
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              title: items => `Уақыт: ${formatTime(items[0].parsed.x)}`,
              label: item => ` ${item.dataset.label}: ${item.parsed.y}%`,
            },
          },
        },
      },
    });
  }

  /* ─── History table: client-side filter ─── */
  const historyTable = document.getElementById('historyTable');
  if (historyTable) {
    const filterInput = document.createElement('input');
    filterInput.type = 'text';
    filterInput.className = 'filter-input';
    filterInput.placeholder = '🔍 Файл атауы бойынша іздеу…';
    historyTable.parentElement.insertBefore(filterInput, historyTable);

    filterInput.addEventListener('input', () => {
      const query = filterInput.value.toLowerCase();
      historyTable.querySelectorAll('tbody tr').forEach(row => {
        const filename = row.cells[1]?.textContent.toLowerCase() || '';
        row.style.display = filename.includes(query) ? '' : 'none';
      });
    });
  }

});

