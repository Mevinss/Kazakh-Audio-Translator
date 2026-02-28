/* =========================================================
   Kazakh ASR – Frontend JS
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {

  /* ----- Drag-and-drop file zone ----- */
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  if (dropZone && fileInput) {
    ['dragenter', 'dragover'].forEach(event => {
      dropZone.addEventListener(event, e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(event => {
      dropZone.addEventListener(event, e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', e => {
      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files;
        updateDropZoneLabel(files[0].name);
      }
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) {
        updateDropZoneLabel(fileInput.files[0].name);
      }
    });

    function updateDropZoneLabel(name) {
      const prompt = dropZone.querySelector('.drop-zone__prompt');
      if (prompt) {
        prompt.innerHTML = `<span class="fs-1">✅</span><br><strong>${escapeHtml(name)}</strong>`;
      }
    }
  }

  /* ----- Form submit: show loading overlay and validate ----- */
  const form = document.getElementById('transcribeForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const submitBtn = document.getElementById('submitBtn');

  if (form) {
    form.addEventListener('submit', e => {
      // Ensure at least one model is checked
      const checked = form.querySelectorAll('.model-check:checked');
      if (checked.length === 0) {
        e.preventDefault();
        alert('Пожалуйста, выберите хотя бы одну модель.');
        return;
      }
      // Show loading state
      if (loadingOverlay) loadingOverlay.classList.remove('d-none');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '⏳ Обработка…';
      }
    });
  }

  /* ----- Simple HTML escape helper ----- */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  /* ----- History table: basic client-side filter ----- */
  const historyTable = document.getElementById('historyTable');
  if (historyTable) {
    const filterInput = document.createElement('input');
    filterInput.type = 'text';
    filterInput.className = 'form-control form-control-sm mb-2';
    filterInput.placeholder = 'Фильтр по имени файла…';
    historyTable.parentElement.insertBefore(filterInput, historyTable);

    filterInput.addEventListener('input', () => {
      const query = filterInput.value.toLowerCase();
      const rows = historyTable.querySelectorAll('tbody tr');
      rows.forEach(row => {
        const filename = row.cells[1]?.textContent.toLowerCase() || '';
        row.style.display = filename.includes(query) ? '' : 'none';
      });
    });
  }

});
