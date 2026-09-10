"use strict";

/* The browser client submits validated inputs and monitors one server job. */
const elements = {
  form: document.querySelector("#simulation-form"),
  plotKinds: Array.from(document.querySelectorAll('input[name="plot_kind"]')),
  year: document.querySelector("#launch-year"),
  month: document.querySelector("#launch-month"),
  day: document.querySelector("#launch-day"),
  dateHelp: document.querySelector("#date-help"),
  plotHelp: document.querySelector("#plot-help"),
  lowerAltitude: document.querySelector("#lower-altitude"),
  upperAltitude: document.querySelector("#upper-altitude"),
  quickDemo: document.querySelector("#quick-demo"),
  useCache: document.querySelector("#use-cache"),
  generateButton: document.querySelector("#generate-button"),
  generateButtonLabel: document.querySelector("#generate-button span"),
  resetSpacecraft: document.querySelector("#reset-spacecraft"),
  connectionAlert: document.querySelector("#connection-alert"),
  connectionMessage: document.querySelector("#connection-message"),
  reconnectButton: document.querySelector("#reconnect-button"),
  serverStatus: document.querySelector("#server-status"),
  serverStatusLabel: document.querySelector("#server-status-label"),
  resultImage: document.querySelector("#result-image"),
  resultPanel: document.querySelector(".result-panel"),
  plotOverlay: document.querySelector("#plot-overlay"),
  downloadLink: document.querySelector("#download-link"),
  activePlotLabel: document.querySelector("#active-plot-label"),
  statusMessage: document.querySelector("#status-message"),
  progress: document.querySelector("#job-progress"),
  progressPercent: document.querySelector("#progress-percent"),
  resultNote: document.querySelector("#result-note"),
};

let serverConfiguration = null;
let pollingTimer = null;
let activeJobIdentifier = null;
let initializationInProgress = false;
const rememberedDates = { line: null, heatmap: null };

/* Resolve the selected graph kind without duplicating DOM queries. */
function selectedPlotKind() {
  return elements.plotKinds.find((control) => control.checked).value;
}

/* Create an inclusive numeric sequence for date selector options. */
function integerRange(start, end) {
  return Array.from({ length: Math.max(0, end - start + 1) }, (_, index) => start + index);
}

/* Replace one select menu while preserving a valid preferred value. */
function populateSelect(select, values, preferredValue, width = 2) {
  const normalized = String(preferredValue).padStart(width, "0");
  select.replaceChildren(
    ...values.map((value) => {
      const option = document.createElement("option");
      option.value = String(value).padStart(width, "0");
      option.textContent = option.value;
      return option;
    }),
  );
  const available = Array.from(select.options).some((option) => option.value === normalized);
  select.value = available ? normalized : select.options[select.options.length - 1]?.value;
}

/* Return the date currently represented by the three browser selectors. */
function selectedDateText() {
  return `${elements.year.value}-${elements.month.value}-${elements.day.value}`;
}

/* Normalize selector components before a month or year change reaches the date parser. */
function refreshDateFromControls() {
  const year = Number(elements.year.value);
  const month = Number(elements.month.value);
  const naturalDayCount = new Date(Date.UTC(year, month, 0)).getUTCDate();
  const day = Math.min(Number(elements.day.value) || 1, naturalDayCount);
  refreshDateSelectors(
    `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`,
  );
}

/* Constrain calendar selectors to the data coverage for the active graph kind. */
function refreshDateSelectors(preferredDate = null) {
  if (!serverConfiguration) {
    return;
  }

  const plotKind = selectedPlotKind();
  const minimum = new Date(`${serverConfiguration.minimum_launch_date}T00:00:00Z`);
  const maximum = new Date(`${serverConfiguration.latest_launch_dates[plotKind]}T00:00:00Z`);
  const candidateText = preferredDate || selectedDateText() || serverConfiguration.default_launch_dates[plotKind];
  const candidate = new Date(`${candidateText}T00:00:00Z`);
  const selected = Number.isNaN(candidate.getTime())
    ? new Date(`${serverConfiguration.default_launch_dates[plotKind]}T00:00:00Z`)
    : new Date(Math.min(maximum.getTime(), Math.max(minimum.getTime(), candidate.getTime())));

  populateSelect(elements.year, integerRange(minimum.getUTCFullYear(), maximum.getUTCFullYear()), selected.getUTCFullYear(), 4);
  const year = Number(elements.year.value);
  const maximumMonth = year === maximum.getUTCFullYear() ? maximum.getUTCMonth() + 1 : 12;
  const minimumMonth = year === minimum.getUTCFullYear() ? minimum.getUTCMonth() + 1 : 1;
  populateSelect(elements.month, integerRange(minimumMonth, maximumMonth), selected.getUTCMonth() + 1);

  const month = Number(elements.month.value);
  const naturalDayCount = new Date(Date.UTC(year, month, 0)).getUTCDate();
  const maximumDay = year === maximum.getUTCFullYear() && month === maximum.getUTCMonth() + 1
    ? maximum.getUTCDate()
    : naturalDayCount;
  const minimumDay = year === minimum.getUTCFullYear() && month === minimum.getUTCMonth() + 1
    ? minimum.getUTCDate()
    : 1;
  populateSelect(elements.day, integerRange(minimumDay, maximumDay), selected.getUTCDate());

  const graphName = plotKind === "line" ? "line-graph" : "heat-map";
  elements.dateHelp.textContent = `Latest supported ${graphName} launch date: ${serverConfiguration.latest_launch_dates[plotKind]}.`;
}

/* Populate the advanced form with the documented VELOX-C1 baseline. */
function applySpacecraftDefaults() {
  if (!serverConfiguration) {
    return;
  }
  Object.entries(serverConfiguration.spacecraft).forEach(([name, value]) => {
    const input = elements.form.elements.namedItem(name);
    if (input) {
      input.value = value;
    }
  });
}

/* Reflect API availability through the compact header indicator. */
function setServerStatus(state, label) {
  elements.serverStatus.dataset.state = state;
  elements.serverStatusLabel.textContent = label;
}

/* Present a server or validation problem in both the form and result areas. */
function showError(message, connectionProblem = false) {
  elements.statusMessage.textContent = message;
  elements.resultNote.textContent = message;
  elements.progress.value = 0;
  elements.progressPercent.textContent = "0%";
  elements.plotOverlay.querySelector(".preview-label").textContent = "Unable to generate graph";
  elements.plotOverlay.querySelector("p").textContent = message;
  elements.plotOverlay.hidden = false;
  elements.resultPanel.dataset.state = "error";
  if (connectionProblem) {
    serverConfiguration = null;
    setServerStatus("offline", "Disconnected");
    elements.connectionMessage.textContent = message;
    elements.connectionAlert.hidden = false;
    elements.generateButton.disabled = true;
  }
}

/* Extract a readable message from FastAPI's structured error response. */
function responseError(payload, fallback) {
  if (typeof payload?.detail === "string") {
    return payload.detail;
  }
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg).join(" ");
  }
  return fallback;
}

/* Update the progress meter and its accessible text as a single operation. */
function setProgress(value, message) {
  const bounded = Math.max(0, Math.min(Number(value) || 0, 100));
  elements.progress.value = bounded;
  elements.progress.textContent = `${Math.round(bounded)}%`;
  elements.progressPercent.textContent = `${Math.round(bounded)}%`;
  elements.statusMessage.textContent = message;
}

/* Restore form availability after a terminal job state. */
function finishPolling() {
  if (pollingTimer !== null) {
    window.clearTimeout(pollingTimer);
    pollingTimer = null;
  }
  activeJobIdentifier = null;
  setFormBusy(false);
  elements.generateButtonLabel.textContent = "Generate graph";
}

/* Prevent request parameters from changing while their graph is being computed. */
function setFormBusy(busy) {
  Array.from(elements.form.elements).forEach((control) => {
    const publicDemoLock = control === elements.quickDemo
      && serverConfiguration?.allow_full_runs === false;
    control.disabled = busy || publicDemoLock;
  });
  elements.generateButton.disabled = busy || !serverConfiguration;
}

/* Poll one accepted server job until a graph or an error is available. */
async function pollJob(identifier) {
  if (identifier !== activeJobIdentifier) {
    return;
  }

  try {
    const response = await fetch(`api/jobs/${identifier}`, { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(responseError(payload, "The simulation status could not be read."));
    }

    setProgress(payload.progress, payload.message);
    if (payload.state === "complete") {
      const imageUrl = `${payload.image_url}?completed=${Date.now()}`;
      elements.resultImage.src = imageUrl;
      elements.resultImage.alt = `${selectedPlotKind() === "line" ? "Line graph" : "Heat map"} generated from the selected VELOX-C1 inputs`;
      elements.plotOverlay.hidden = true;
      elements.downloadLink.href = payload.image_url;
      elements.downloadLink.hidden = false;
      elements.resultNote.textContent = payload.message;
      elements.resultPanel.dataset.state = "complete";
      finishPolling();
      return;
    }
    if (payload.state === "failed") {
      showError(payload.error || payload.message);
      finishPolling();
      return;
    }
    pollingTimer = window.setTimeout(() => pollJob(identifier), 800);
  } catch (error) {
    showError(error.message, true);
    finishPolling();
  }
}

/* Convert all browser controls into the JSON schema accepted by the API. */
function requestPayload() {
  const spacecraft = {};
  Object.keys(serverConfiguration.spacecraft).forEach((name) => {
    spacecraft[name] = Number(elements.form.elements.namedItem(name).value);
  });
  return {
    launch_date: selectedDateText(),
    lower_altitude_km: Number(elements.lowerAltitude.value),
    upper_altitude_km: Number(elements.upperAltitude.value),
    plot_kind: selectedPlotKind(),
    use_cache: elements.useCache.checked,
    quick_demo: elements.quickDemo.checked,
    spacecraft,
  };
}

/* Validate and submit a graph request without navigating away from the page. */
async function submitSimulation(event) {
  event.preventDefault();
  if (!serverConfiguration || !elements.form.reportValidity()) {
    return;
  }
  if (Number(elements.upperAltitude.value) <= Number(elements.lowerAltitude.value)) {
    showError("Upper altitude must be greater than lower altitude.");
    return;
  }

  const submission = requestPayload();
  setFormBusy(true);
  elements.generateButtonLabel.textContent = "Generating…";
  elements.resultPanel.dataset.state = "running";
  elements.downloadLink.hidden = true;
  elements.plotOverlay.hidden = false;
  elements.plotOverlay.querySelector(".preview-label").textContent = "Simulation running";
  elements.plotOverlay.querySelector("p").textContent = "The graph will appear here when the numerical worker finishes.";
  setProgress(0, "Submitting simulation");

  try {
    const response = await fetch("api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(submission),
    });
    const responsePayload = await response.json();
    if (!response.ok) {
      throw new Error(responseError(responsePayload, "The simulation request was rejected."));
    }
    activeJobIdentifier = responsePayload.job_id;
    setProgress(responsePayload.progress, responsePayload.message);
    await pollJob(responsePayload.job_id);
  } catch (error) {
    showError(error.message, error instanceof TypeError);
    finishPolling();
  }
}

/* Keep per-mode dates, help text, and result descriptions synchronized. */
function changePlotKind(event) {
  const otherKind = event.target.value === "line" ? "heatmap" : "line";
  rememberedDates[otherKind] = selectedDateText();
  const plotKind = selectedPlotKind();
  const selectedDate = rememberedDates[plotKind] || serverConfiguration?.default_launch_dates[plotKind];
  refreshDateSelectors(selectedDate);
  elements.plotHelp.textContent = plotKind === "line"
    ? "One-year mean decay rate across the selected altitude interval."
    : "Instantaneous mean decay rate across altitude and solar-cycle phase.";
  elements.activePlotLabel.textContent = plotKind === "line" ? "Line graph" : "Heat map";
}

/* Fetch model limits before enabling requests from the page. */
async function initializeApplication() {
  if (initializationInProgress) {
    return;
  }

  if (window.location.protocol === "file:") {
    elements.reconnectButton.hidden = true;
    showError("This page needs the Python model. Run python run_web.py, then open http://127.0.0.1:8000.", true);
    return;
  }

  initializationInProgress = true;
  elements.reconnectButton.hidden = false;
  elements.reconnectButton.disabled = true;
  elements.reconnectButton.textContent = "Checking…";
  elements.generateButton.disabled = true;
  setServerStatus("connecting", "Connecting");

  try {
    const response = await fetch("api/config", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(responseError(payload, "Model configuration is unavailable."));
    }
    serverConfiguration = payload;
    if (payload.allow_full_runs === false) {
      elements.quickDemo.checked = true;
      elements.quickDemo.disabled = true;
      elements.quickDemo.closest("label").querySelector("small").textContent =
        "Required on this public service · 8 realizations · coarse 30-day step";
    }
    refreshDateSelectors(payload.default_launch_dates.line);
    applySpacecraftDefaults();
    elements.connectionAlert.hidden = true;
    elements.generateButton.disabled = false;
    elements.statusMessage.textContent = "Ready to generate";
    if (elements.resultPanel.dataset.state === "error") {
      elements.resultPanel.dataset.state = "idle";
      elements.plotOverlay.querySelector(".preview-label").textContent = "Example output";
      elements.plotOverlay.querySelector("p").textContent = "Choose the graph type and parameters, then generate a new result.";
      elements.resultNote.textContent = "Ready. The generated PNG will fill this panel and remain available for download.";
    }
    setServerStatus("online", "Model online");
  } catch (error) {
    const isLocalHost = ["127.0.0.1", "localhost", "::1"].includes(window.location.hostname);
    const message = isLocalHost
      ? "The local Python service is not running. Open start_web.command and leave its Terminal window open, then try again."
      : "The simulation service is temporarily unavailable. Please try again shortly.";
    showError(message, true);
  } finally {
    initializationInProgress = false;
    elements.reconnectButton.disabled = false;
    elements.reconnectButton.textContent = "Try again";
  }
}

/* Calendar changes may alter the valid months and days at the coverage boundary. */
elements.year.addEventListener("change", refreshDateFromControls);
elements.month.addEventListener("change", refreshDateFromControls);
elements.plotKinds.forEach((control) => control.addEventListener("change", changePlotKind));
elements.resetSpacecraft.addEventListener("click", applySpacecraftDefaults);
elements.reconnectButton.addEventListener("click", initializeApplication);
elements.form.addEventListener("submit", submitSimulation);
window.addEventListener("beforeunload", finishPolling);
window.addEventListener("online", initializeApplication);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && !serverConfiguration) {
    initializeApplication();
  }
});

initializeApplication();
