// Front-end logic: upload files to the API and draw the results.

const COLORS = { known: "#1a7f37", unknown: "#c2410c" };

const folderInput = document.getElementById("folder-input");
const photoInput = document.getElementById("photo-input");
const photoButton = document.getElementById("photo-button");
const galleryStatus = document.getElementById("gallery-status");
const galleryWarnings = document.getElementById("gallery-warnings");
const photoStatus = document.getElementById("photo-status");

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = "status " + kind;
}

function enablePhotoStep(enabled) {
  photoInput.disabled = !enabled;
  photoButton.setAttribute("aria-disabled", String(!enabled));
}

// Turn a FastAPI error response into a readable message.
async function errorMessage(response) {
  try {
    const body = await response.json();
    const detail = body.detail;
    if (typeof detail === "string") return { message: detail, warnings: [] };
    if (detail && detail.message) return { message: detail.message, warnings: detail.warnings || [] };
  } catch (_) { /* response was not JSON */ }
  return { message: `Server error (${response.status}).`, warnings: [] };
}

function showWarnings(list) {
  galleryWarnings.replaceChildren(...list.map((w) => {
    const li = document.createElement("li");
    li.textContent = w;
    return li;
  }));
}

// ---- Step 1: reference folder -------------------------------------------
folderInput.addEventListener("change", async () => {
  const files = Array.from(folderInput.files);
  folderInput.value = ""; // allow choosing the same folder again
  if (files.length === 0) return;

  const form = new FormData();
  // Send only the file name (not the sub-folder path); the name becomes the person label.
  files.forEach((f) => form.append("files", f, f.name));

  showWarnings([]);
  setStatus(galleryStatus, `Reading ${files.length} file(s)… this can take a little while.`);
  try {
    const response = await fetch("/api/gallery", { method: "POST", body: form });
    if (!response.ok) {
      const err = await errorMessage(response);
      setStatus(galleryStatus, err.message, "error");
      showWarnings(err.warnings);
      return;
    }
    const data = await response.json();
    setStatus(
      galleryStatus,
      `Loaded ${data.people.length} people from ${data.reference_photos} photo(s): ${data.people.join(", ")}`,
      "ok",
    );
    showWarnings(data.warnings);
    enablePhotoStep(true);
  } catch (e) {
    setStatus(galleryStatus, "Could not reach the server. Is it running?", "error");
  }
});

// ---- Step 2: group photo -------------------------------------------------
photoInput.addEventListener("change", async () => {
  const file = photoInput.files[0];
  photoInput.value = "";
  if (!file) return;

  const form = new FormData();
  form.append("file", file, file.name);
  setStatus(photoStatus, `Analysing ${file.name}…`);
  document.getElementById("results").hidden = true;

  try {
    const response = await fetch("/api/identify", { method: "POST", body: form });
    if (!response.ok) {
      setStatus(photoStatus, (await errorMessage(response)).message, "error");
      return;
    }
    const data = await response.json();
    setStatus(photoStatus, "");
    await showResults(file, data);
  } catch (e) {
    setStatus(photoStatus, "Could not reach the server. Is it running?", "error");
  }
});

// ---- Results ---------------------------------------------------------------
function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The browser could not display this image."));
    img.src = URL.createObjectURL(file);
  });
}

async function showResults(file, data) {
  const { faces, summary } = data;
  document.getElementById("summary").textContent =
    summary.total === 0
      ? "No faces were detected in this photo."
      : `${summary.total} face(s) found: ${summary.identified} identified, ${summary.unknown} unknown.`;
  document.getElementById("threshold-note").textContent =
    `A name is shown only when similarity is at least ${data.threshold.toFixed(2)}; otherwise the face is marked Unknown.`;

  let img;
  try {
    img = await loadImage(file);
  } catch (e) {
    setStatus(photoStatus, e.message, "error");
    return;
  }
  drawCanvas(img, data);
  fillTable(faces);
  document.getElementById("results").hidden = false;
}

function drawCanvas(img, data) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  // Boxes are in the server's image coordinates; scale in case the browser size differs.
  const sx = img.naturalWidth / data.width;
  const sy = img.naturalHeight / data.height;
  const line = Math.max(2, Math.round(img.naturalWidth / 400));
  const fontSize = Math.max(14, Math.round(img.naturalWidth / 55));
  ctx.font = `600 ${fontSize}px sans-serif`;
  ctx.textBaseline = "top";

  for (const face of data.faces) {
    const [x1, y1, x2, y2] = face.box;
    const x = x1 * sx, y = y1 * sy, w = (x2 - x1) * sx, h = (y2 - y1) * sy;
    const color = face.name === "Unknown" ? COLORS.unknown : COLORS.known;

    ctx.lineWidth = line;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const label = `${face.id}. ${face.name}`;
    const pad = Math.round(fontSize * 0.3);
    const textW = ctx.measureText(label).width + pad * 2;
    const textH = fontSize + pad * 2;
    const labelY = y - textH >= 0 ? y - textH : y + h; // above the box, or below if no room
    ctx.fillStyle = color;
    ctx.fillRect(x - line / 2, labelY, textW, textH);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x - line / 2 + pad, labelY + pad);
  }
}

function fillTable(faces) {
  const body = document.getElementById("results-body");
  body.replaceChildren(...faces.map((face) => {
    const tr = document.createElement("tr");
    const known = face.name !== "Unknown";
    [String(face.id), face.name, face.score.toFixed(2)].forEach((text, i) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (i === 1) td.className = known ? "known" : "unknown";
      tr.appendChild(td);
    });
    return tr;
  }));
}
