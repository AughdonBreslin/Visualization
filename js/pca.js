document.addEventListener('DOMContentLoaded', () => {
  const dataContainer = document.getElementById('pcaDataViz');
  const operatorContainer = document.getElementById('pcaOperatorViz');
  if (!dataContainer || !operatorContainer) return;

  const dimensionInput = document.getElementById('pcaDimension');
  const presetInput = document.getElementById('pcaPreset');
  const samplesInput = document.getElementById('pcaSamples');
  const spread1Input = document.getElementById('pcaSpread1');
  const spread1ValueEl = document.getElementById('pcaSpread1Value');
  const spread2Input = document.getElementById('pcaSpread2');
  const spread2ValueEl = document.getElementById('pcaSpread2Value');
  const spread3ControlEl = document.getElementById('pcaSpread3Control');
  const spread3Input = document.getElementById('pcaSpread3');
  const spread3ValueEl = document.getElementById('pcaSpread3Value');
  const angleInput = document.getElementById('pcaAngle');
  const angleValueEl = document.getElementById('pcaAngleValue');
  const elevationControlEl = document.getElementById('pcaElevationControl');
  const elevationInput = document.getElementById('pcaElevation');
  const elevationValueEl = document.getElementById('pcaElevationValue');
  const noiseInput = document.getElementById('pcaNoise');
  const noiseValueEl = document.getElementById('pcaNoiseValue');
  const stepInput = document.getElementById('pcaStep');
  const rankInput = document.getElementById('pcaRank');
  const showVectorsInput = document.getElementById('pcaShowVectors');
  const showLabelsInput = document.getElementById('pcaShowLabels');
  const showRank1Input = document.getElementById('pcaShowRank1');
  const useCubeAspectInput = document.getElementById('pcaUseCubeAspect');
  const syncCamerasInput = document.getElementById('pcaSyncCameras');
  const randomizeBtn = document.getElementById('pcaRandomize');
  const controlTabButtons = Array.from(document.querySelectorAll('.pca-control-tab'));
  const controlTabPanels = Array.from(document.querySelectorAll('.pca-control-panel'));

  const stepSummaryEl = document.getElementById('pcaStepSummary');
  const dataLegendEl = document.getElementById('pcaDataLegend');
  const operatorLegendEl = document.getElementById('pcaOperatorLegend');
  const matrixXEl = document.getElementById('pcaMatrixX');
  const matrixFactorsEl = document.getElementById('pcaMatrixFactors');
  const matrixCovEl = document.getElementById('pcaMatrixCov');
  const matrixCurrentEl = document.getElementById('pcaMatrixCurrent');

  let seed = 7;
  let shared3DCamera = { eye: { x: 1.45, y: 1.35, z: 1.15 } };
  let last3DCameraSource = dataContainer;

  function syncAngleLabel() {
    if (!angleValueEl || !angleInput) return;
    const angle = Number(angleInput.value);
    angleValueEl.textContent = `${Number.isFinite(angle) ? Math.round(angle) : 0}°`;
  }

  function syncNumericLabel(input, output, digits = 2, suffix = '') {
    if (!input || !output) return;
    const value = Number(input.value);
    output.textContent = Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : `0.00${suffix}`;
  }

  function clampPositive(x, minValue = 1e-6) {
    if (!Number.isFinite(x)) return minValue;
    return Math.max(minValue, x);
  }

  function get3DAspectMode() {
    return useCubeAspectInput?.checked ? 'cube' : 'data';
  }

  function shouldSync3DCameras() {
    return !!syncCamerasInput?.checked;
  }

  function getLiveCamera(container) {
    return cloneCamera(container?._fullLayout?.scene?.camera) || cloneCamera(shared3DCamera);
  }

  function getVectorLength(vector) {
    if (!vector) return 0;
    const x = Number(vector.x) || 0;
    const y = Number(vector.y) || 0;
    const z = Number(vector.z) || 0;
    return Math.sqrt(x * x + y * y + z * z);
  }

  function normalizeVector(vector, fallback = { x: 1.45, y: 1.35, z: 1.15 }) {
    const length = getVectorLength(vector);
    if (length < 1e-6) {
      return normalizeVector(fallback, { x: 1, y: 0, z: 0 });
    }
    return {
      x: (Number(vector.x) || 0) / length,
      y: (Number(vector.y) || 0) / length,
      z: (Number(vector.z) || 0) / length,
    };
  }

  function scaleVector(vector, length) {
    return {
      x: vector.x * length,
      y: vector.y * length,
      z: vector.z * length,
    };
  }

  function mergeCameraOrientation(sourceCamera, targetCamera) {
    const nextTargetCamera = cloneCamera(targetCamera) || {};
    const sourceEye = sourceCamera?.eye || shared3DCamera?.eye;
    const targetEye = nextTargetCamera.eye || sourceEye || { x: 1.45, y: 1.35, z: 1.15 };
    const direction = normalizeVector(sourceEye);
    const targetDistance = Math.max(getVectorLength(targetEye), 1e-6);

    nextTargetCamera.eye = scaleVector(direction, targetDistance);
    if (sourceCamera?.up) {
      nextTargetCamera.up = cloneCamera(sourceCamera.up);
    }
    if (!nextTargetCamera.center && sourceCamera?.center) {
      nextTargetCamera.center = cloneCamera(sourceCamera.center);
    }

    return nextTargetCamera;
  }

  function applyCameraOrientation(sourceContainer, targetContainer) {
    if (!window.Plotly || !sourceContainer || !targetContainer) return;
    if (sourceContainer.dataset.plotlyInitialized !== 'true' || targetContainer.dataset.plotlyInitialized !== 'true') return;

    const sourceCamera = getLiveCamera(sourceContainer);

    targetContainer.dataset.suppressCameraSync = 'true';
    Plotly.relayout(targetContainer, { 'scene.camera': cloneCamera(sourceCamera) })
      .catch(() => {})
      .finally(() => {
        delete targetContainer.dataset.suppressCameraSync;
      });
  }

  function debounce(fn, delay = 180) {
    let timer = null;
    return (...args) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function setActiveControlTab(tabName) {
    controlTabButtons.forEach((button) => {
      const isActive = button.dataset.tabTarget === tabName;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    controlTabPanels.forEach((panel) => {
      panel.hidden = panel.dataset.tabPanel !== tabName;
    });
  }

  function clear(el) {
    if (window.Plotly && el?.dataset?.plotlyInitialized === 'true') {
      Plotly.purge(el);
    }
    if (el) {
      el.innerHTML = '';
      delete el.dataset.plotlyInitialized;
      delete el.dataset.renderer;
      delete el.dataset.cameraSyncBound;
      delete el.dataset.suppressCameraSync;
    }
  }

  function renderPlotly(container, traces, layout) {
    if (!window.Plotly) {
      throw new Error('Plotly failed to load, so the 3D view is unavailable.');
    }

    if (container.dataset.renderer && container.dataset.renderer !== 'plotly') {
      clear(container);
    }

    const config = {
      responsive: true,
      displayModeBar: false,
    };

    if (container.dataset.plotlyInitialized === 'true') {
      Plotly.react(container, traces, layout, config);
    } else {
      Plotly.newPlot(container, traces, layout, config);
      container.dataset.plotlyInitialized = 'true';
    }
    container.dataset.renderer = 'plotly';
  }

  function cloneCamera(camera) {
    if (!camera) return null;
    return JSON.parse(JSON.stringify(camera));
  }

  function getCameraFromEvent(sourceContainer, eventData) {
    const eventCamera = eventData?.['scene.camera'];
    if (eventCamera) return cloneCamera(eventCamera);

    const cameraKeys = Object.keys(eventData || {}).filter((key) => key.startsWith('scene.camera.'));
    if (cameraKeys.length > 0) {
      const baseCamera = cloneCamera(sourceContainer?._fullLayout?.scene?.camera) || {};
      cameraKeys.forEach((key) => {
        const segments = key.split('.').slice(1);
        let cursor = baseCamera;
        for (let index = 0; index < segments.length - 1; index += 1) {
          const segment = segments[index];
          if (!cursor[segment] || typeof cursor[segment] !== 'object') {
            cursor[segment] = {};
          }
          cursor = cursor[segment];
        }
        cursor[segments[segments.length - 1]] = eventData[key];
      });
      return baseCamera;
    }

    const liveCamera = sourceContainer?._fullLayout?.scene?.camera;
    if (liveCamera) return cloneCamera(liveCamera);

    return null;
  }

  function syncPlotlyCamera(sourceContainer, targetContainer) {
    if (!window.Plotly || !sourceContainer || !targetContainer) return;
    if (sourceContainer.dataset.cameraSyncBound === 'true') return;

    let syncFrameId = null;
    let queuedCamera = null;
    let relayoutInFlight = false;

    function flushQueuedCamera() {
      if (relayoutInFlight || !queuedCamera) return;
      if (targetContainer.dataset.plotlyInitialized !== 'true') return;

      const cameraToApply = cloneCamera(queuedCamera);
      queuedCamera = null;
      relayoutInFlight = true;
      targetContainer.dataset.suppressCameraSync = 'true';

      Plotly.relayout(targetContainer, { 'scene.camera': cloneCamera(cameraToApply) })
        .catch(() => {})
        .finally(() => {
          relayoutInFlight = false;
          delete targetContainer.dataset.suppressCameraSync;
          if (queuedCamera) {
            flushQueuedCamera();
          }
        });
    }

    function queueCameraSync(eventData) {
      if (sourceContainer.dataset.suppressCameraSync === 'true') return;
      const nextCamera = getCameraFromEvent(sourceContainer, eventData);
      if (!nextCamera) return;

      last3DCameraSource = sourceContainer;
      shared3DCamera = cloneCamera(nextCamera);

      if (!shouldSync3DCameras()) return;

      queuedCamera = cloneCamera(nextCamera);

      if (syncFrameId !== null) return;
      syncFrameId = window.requestAnimationFrame(() => {
        syncFrameId = null;
        flushQueuedCamera();
      });
    }

    sourceContainer.on('plotly_relayouting', queueCameraSync);
    sourceContainer.on('plotly_relayout', queueCameraSync);

    sourceContainer.dataset.cameraSyncBound = 'true';
  }

  function sync3DPlots() {
    syncPlotlyCamera(dataContainer, operatorContainer);
    syncPlotlyCamera(operatorContainer, dataContainer);

    if (!shouldSync3DCameras()) return;

    const sourceContainer = last3DCameraSource === operatorContainer ? operatorContainer : dataContainer;
    const targetContainer = sourceContainer === dataContainer ? operatorContainer : dataContainer;
    applyCameraOrientation(sourceContainer, targetContainer);
  }

  function mulberry32(seedValue) {
    let a = seedValue >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleStandardNormal(rng) {
    const u1 = Math.max(1e-12, rng());
    const u2 = Math.max(1e-12, rng());
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function roundSmall(x) {
    if (!Number.isFinite(x)) return 0;
    return Math.abs(x) < 1e-9 ? 0 : x;
  }

  function degToRad(degrees) {
    return degrees * Math.PI / 180;
  }

  function matrixIdentity(size) {
    return Array.from({ length: size }, (_, rowIndex) => (
      Array.from({ length: size }, (_, colIndex) => (rowIndex === colIndex ? 1 : 0))
    ));
  }

  function matrixTranspose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map((row) => row[colIndex]));
  }

  function matrixMultiply(a, b) {
    return a.map((row) => (
      b[0].map((_, colIndex) => row.reduce((sum, value, innerIndex) => sum + value * b[innerIndex][colIndex], 0))
    ));
  }

  function matrixDiagonal(values) {
    return values.map((value, rowIndex) => values.map((_, colIndex) => (rowIndex === colIndex ? value : 0)));
  }

  function applyMatrix(matrix, vector) {
    return matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));
  }

  function rotationMatrix2(theta) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return [
      [c, -s],
      [s, c],
    ];
  }

  function rotationMatrixX(theta) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return [
      [1, 0, 0],
      [0, c, -s],
      [0, s, c],
    ];
  }

  function rotationMatrixZ(theta) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return [
      [c, -s, 0],
      [s, c, 0],
      [0, 0, 1],
    ];
  }

  function rotationMatrix3(theta, elevation) {
    return matrixMultiply(rotationMatrixZ(theta), rotationMatrixX(elevation));
  }

  function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
  }

  function centerPoints(points) {
    const dimension = points[0]?.length || 2;
    const means = Array.from({ length: dimension }, (_, axisIndex) => mean(points.map((point) => point[axisIndex])));
    return {
      centered: points.map((point) => point.map((value, axisIndex) => value - means[axisIndex])),
      mean: means,
    };
  }

  function getDimension() {
    return Math.max(2, Math.min(3, Number(dimensionInput?.value) || 3));
  }

  function syncDimensionControls() {
    const is3D = getDimension() === 3;
    if (spread3ControlEl) spread3ControlEl.hidden = !is3D;
    if (elevationControlEl) elevationControlEl.hidden = !is3D;
  }

  function buildDataset() {
    const dimension = getDimension();
    const preset = presetInput.value;
    const samples = Math.max(8, Math.floor(Number(samplesInput.value) || 24));
    const spread1 = clampPositive(Number(spread1Input.value), 0.1);
    const spread2 = clampPositive(Number(spread2Input.value), 0.05);
    const spread3 = clampPositive(Number(spread3Input.value), 0.05);
    const noise = Math.max(0, Number(noiseInput.value) || 0);
    const theta = degToRad(Number(angleInput.value) || 0);
    const elevation = degToRad(Number(elevationInput?.value) || 0);
    const rotation = dimension === 3 ? rotationMatrix3(theta, elevation) : rotationMatrix2(theta);
    const rng = mulberry32(seed);
    const raw = [];

    for (let i = 0; i < samples; i++) {
      let localPoint;
      if (dimension === 3) {
        if (preset === 'line') {
          const t = -2.8 + 5.6 * (i / Math.max(1, samples - 1));
          localPoint = [
            t * spread1 * 0.75,
            sampleStandardNormal(rng) * spread2 * 0.35,
            sampleStandardNormal(rng) * spread3 * 0.28,
          ];
        } else if (preset === 'clusters') {
          const clusterSign = i < samples / 2 ? -1 : 1;
          localPoint = [
            clusterSign * spread1 * 0.85 + sampleStandardNormal(rng) * spread1 * 0.4,
            sampleStandardNormal(rng) * spread2,
            sampleStandardNormal(rng) * spread3,
          ];
        } else {
          localPoint = [
            sampleStandardNormal(rng) * spread1,
            sampleStandardNormal(rng) * spread2,
            sampleStandardNormal(rng) * spread3,
          ];
        }

        const rotated = applyMatrix(rotation, localPoint);
        raw.push(rotated.map((value) => value + sampleStandardNormal(rng) * noise));
      } else {
        if (preset === 'line') {
          const t = -2.5 + 5 * (i / Math.max(1, samples - 1));
          localPoint = [t * spread1 * 0.7, sampleStandardNormal(rng) * spread2 * 0.4];
        } else if (preset === 'clusters') {
          const clusterSign = i < samples / 2 ? -1 : 1;
          localPoint = [
            clusterSign * spread1 * 0.8 + sampleStandardNormal(rng) * spread1 * 0.45,
            sampleStandardNormal(rng) * spread2,
          ];
        } else {
          localPoint = [sampleStandardNormal(rng) * spread1, sampleStandardNormal(rng) * spread2];
        }

        const rotated = applyMatrix(rotation, localPoint);
        raw.push(rotated.map((value) => value + sampleStandardNormal(rng) * noise));
      }
    }

    return centerPoints(raw).centered;
  }

  function formatNumber(x, digits = 3) {
    const rounded = roundSmall(x);
    return `${rounded >= 0 ? ' ' : ''}${rounded.toFixed(digits)}`;
  }

  function formatMatrix(matrix, maxRows = matrix.length) {
    const rows = matrix.slice(0, maxRows).map((row) => `[${row.map((value) => formatNumber(value)).join(', ')}]`);
    if (matrix.length > maxRows) rows.push('...');
    return rows.join('\n');
  }

  function getPlotSize(container, minHeight = 300) {
    const rect = container.getBoundingClientRect();
    return { width: Math.max(320, rect.width), height: minHeight };
  }

  function computePCA(points) {
    if (!window.numeric || typeof numeric.svd !== 'function') {
      throw new Error('numeric.js failed to load, so SVD is unavailable.');
    }

    const dimension = points[0]?.length || 2;
    const X = points.map((point) => point.slice(0, dimension));
    const svd = numeric.svd(X);
    const singularValues = svd.S.slice(0, dimension).map((value) => roundSmall(value));
    const V = svd.V.map((row) => row.slice(0, dimension).map(roundSmall));
    const VT = matrixTranspose(V);
    const lambda = singularValues.map((value) => (value * value) / Math.max(1, points.length - 1));
    const Lambda = matrixDiagonal(lambda);
    const covariance = matrixMultiply(V, matrixMultiply(Lambda, VT));
    const scores = points.map((point) => applyMatrix(VT, point));

    return {
      X,
      U: svd.U.map((row) => row.slice(0, dimension).map(roundSmall)),
      singularValues,
      Sigma: matrixDiagonal(singularValues),
      V,
      VT,
      lambda,
      Lambda,
      covariance,
      scores,
      dimension,
    };
  }

  function getRankReconstruction(decomposition, rank) {
    const safeRank = Math.max(1, Math.min(rank, decomposition.dimension));
    const truncatedScores = decomposition.scores.map((score) => score.map((value, index) => (index < safeRank ? value : 0)));
    return truncatedScores.map((score) => applyMatrix(decomposition.V, score));
  }

  function getCurrentTransform(step, decomposition) {
    if (step === 1) return decomposition.VT;
    if (step === 2) return matrixMultiply(decomposition.Lambda, decomposition.VT);
    if (step === 3) return decomposition.covariance;
    return matrixIdentity(decomposition.dimension);
  }

  function formatSpectrum(values) {
    return values.map((value, index) => `${index + 1}: ${value.toFixed(3)}`).join(' | ');
  }

  function getStepMeta(step, decomposition) {
    if (step === 1) {
      return {
        name: 'Rotate by V^T',
        explanation: `The ${decomposition.dimension}D cloud is rotated into principal coordinates, so the directions of largest, second-largest, and remaining variance line up with the coordinate axes.`,
        motivation: 'This is the operational heart of PCA. By rotating into the eigenvector basis, correlation between coordinates is removed and variance can be read off axis by axis.',
        dataView: 'The left plot shows the same centered samples in a new coordinate system. The cloud has not been stretched yet; it has only been re-expressed so its longest spread lies along pc1, the next along pc2, and so on.',
        operatorView: 'The right plot shows a unit circle or sphere after the same rotation. Rotations preserve shape, so the object should still look round; only its orientation changes.',
        interpretation: 'If one axis is much longer than the others after this rotation, PCA has found a strong low-dimensional structure. Projection now becomes easy: keep the first few rotated coordinates and discard the rest.',
        matrixLabel: 'Current matrix action',
        matrixAction: 'Apply V^T to rotate feature coordinates into the principal basis.',
        transform: decomposition.VT,
      };
    }
    if (step === 2) {
      return {
        name: 'Stretch by Λ',
        explanation: 'After rotating into the eigenbasis, the covariance operator scales each principal axis by its eigenvalue, so high-variance directions expand more than low-variance ones.',
        motivation: 'This step isolates what covariance contributes beyond rotation: anisotropic scaling. Large eigenvalues mean the data varies strongly in that principal direction, while small eigenvalues mean that direction contributes little.',
        dataView: 'The left plot now applies the diagonal matrix Λ after the rotation. Axes are no longer treated equally: directions with large variance are stretched more strongly than directions with small variance.',
        operatorView: 'The right plot turns the unit circle or sphere into an ellipse or ellipsoid. Its semiaxis lengths are the eigenvalues, so the deformation makes the variance profile visible as geometry.',
        interpretation: 'This is why PCA ranks components. The first few eigenvalues tell you how much structure lives along each principal axis, and tiny eigenvalues signal directions that can often be dropped with little loss.',
        matrixLabel: 'Current matrix action',
        matrixAction: 'Apply Λ after the rotation so each principal coordinate is scaled by its variance weight.',
        transform: matrixMultiply(decomposition.Lambda, decomposition.VT),
      };
    }
    if (step === 3) {
      return {
        name: 'Rotate back by V',
        explanation: 'Applying V returns the stretched result to feature coordinates. This completes the covariance action VΛV^T.',
        motivation: 'Covariance is not just a diagonal scaling in the original basis. The rotate-back step shows how the operator acts in the ambient feature coordinates where the data was originally observed.',
        dataView: 'The left plot shows the final covariance action on the sample cloud. The deformation is now expressed in the original feature axes, so it may look tilted again even though its principal structure is unchanged.',
        operatorView: 'The right plot shows the transformed circle or sphere rotated back into the ambient coordinates. This is the full covariance operator viewed as a single linear map on the original space.',
        interpretation: 'This stage explains why covariance eigenvectors matter: they are precisely the directions that survive this full map without changing direction, only length. Their lengths are scaled by the eigenvalues.',
        matrixLabel: 'Current matrix action',
        matrixAction: 'Apply VΛV^T, the full covariance operator written in the original feature basis.',
        transform: decomposition.covariance,
      };
    }
    return {
      name: 'Original centered data',
      explanation: `The ${decomposition.dimension}D cloud has been centered so PCA measures variation around the mean rather than around an arbitrary offset. The principal directions come from the right singular vectors in V.`,
      motivation: 'Subtracting the mean from all samples centers the data on 0. This neutralizes the location of the data in space, allowing PCA to isolate principal components of variation rather than just following a biased direction from the origin.',
      dataView: 'The left plot shows the centered samples in their original feature coordinates. Any tilt or elongation here is the geometric structure PCA will try to summarize.',
      operatorView: 'The right plot shows the unit circle or sphere before any covariance-induced deformation. It is the reference object that later stages rotate and stretch.',
      interpretation: 'Use this stage to compare the raw shape of the cloud with the later aligned and stretched versions. The singular values and eigenvalues reported below already tell you how much variance each principal direction will capture.',
      matrixLabel: 'Current matrix action',
      matrixAction: 'Identity: no transformation yet, only centering of the raw data matrix X.',
      transform: matrixIdentity(decomposition.dimension),
    };
  }

  function renderStepSummary(stepMeta, decomposition) {
    if (!stepSummaryEl) return;

    const sections = [
      { label: 'What is happening', text: stepMeta.explanation },
      { label: 'Why PCA uses this step', text: stepMeta.motivation },
      { label: 'Data / Principal Components', text: stepMeta.dataView },
      { label: 'Covariance Operator', text: stepMeta.operatorView },
      { label: 'How to read it', text: stepMeta.interpretation },
      { label: stepMeta.matrixLabel, text: stepMeta.matrixAction },
    ];

    stepSummaryEl.innerHTML = `
      <div class="pca-step-summary-card">
        <div class="pca-step-summary-header">
          <strong>${stepMeta.name}</strong>
          <span>${decomposition.dimension}D walkthrough stage</span>
        </div>
        <div class="pca-step-summary-grid">
          ${sections.map((section) => `
            <div class="pca-step-summary-section">
              <div class="pca-step-summary-label">${section.label}</div>
              <p>${section.text}</p>
            </div>
          `).join('')}
        </div>
        <div class="pca-step-summary-stats">
          <div>
            <span>Singular values</span>
            <strong>${formatSpectrum(decomposition.singularValues)}</strong>
            <p>These are the axis lengths of the centered data matrix under SVD, so larger values indicate directions where the data cloud extends more strongly.</p>
          </div>
          <div>
            <span>Covariance eigenvalues</span>
            <strong>${formatSpectrum(decomposition.lambda)}</strong>
            <p>These are the variances along the principal directions, so they quantify how much spread each retained PCA component explains.</p>
          </div>
        </div>
      </div>
    `;
  }

  function extentWithPadding(values) {
    const minValue = d3.min(values);
    const maxValue = d3.max(values);
    const span = Math.max(1e-6, maxValue - minValue);
    const pad = span * 0.18;
    return [minValue - pad, maxValue + pad];
  }

  function makeVectorTrace(vector, color, label) {
    return {
      type: 'scatter3d',
      mode: 'lines+text',
      x: [-vector[0], vector[0]],
      y: [-vector[1], vector[1]],
      z: [-vector[2], vector[2]],
      text: ['', label],
      textposition: 'top center',
      textfont: { color, size: 14 },
      line: { color, width: 7 },
      hoverinfo: 'skip',
      showlegend: false,
    };
  }

  function makeAxisTrace(axisVector, color, label) {
    return {
      type: 'scatter3d',
      mode: 'lines+text',
      x: [0, axisVector[0]],
      y: [0, axisVector[1]],
      z: [0, axisVector[2]],
      text: ['', label],
      textposition: 'top center',
      textfont: { color: 'rgba(255,255,255,0.72)', size: 13 },
      line: { color, width: 5 },
      hoverinfo: 'skip',
      showlegend: false,
    };
  }

  function drawScatterPlot2D({ container, points, principalVectors, showVectors, showLabels, overlayPoints, title, axisLabels, basisLabels }) {
    clear(container);
    container.dataset.renderer = 'svg';
    const { width, height } = getPlotSize(container, 460);
    const margin = { top: 6, right: 18, bottom: 34, left: 44 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const vectorPoints = showVectors ? principalVectors.flatMap((vector) => [vector, vector.map((value) => -value)]) : [];
    const allPoints = (overlayPoints ? points.concat(overlayPoints) : points.slice()).concat(vectorPoints);
    const xs = allPoints.map((point) => point[0]).concat([0]);
    const ys = allPoints.map((point) => point[1]).concat([0]);
    const xScale = d3.scaleLinear().domain(extentWithPadding(xs)).range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(extentWithPadding(ys)).range([innerHeight, 0]);

    const svg = d3.select(container).append('svg').attr('width', '100%').attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g').attr('transform', `translate(0,${yScale(0)})`).call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
    g.append('g').attr('transform', `translate(${xScale(0)},0)`).call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));

    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.75)')
      .attr('font-size', 14)
      .text(axisLabels[0]);

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -34)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.75)')
      .attr('font-size', 14)
      .text(axisLabels[1]);

    g.selectAll('circle.data-point')
      .data(points)
      .enter()
      .append('circle')
      .attr('class', 'data-point')
      .attr('cx', (point) => xScale(point[0]))
      .attr('cy', (point) => yScale(point[1]))
      .attr('r', 4)
      .attr('fill', 'rgba(74, 163, 255, 0.88)');

    if (overlayPoints) {
      g.selectAll('circle.overlay-point')
        .data(overlayPoints)
        .enter()
        .append('circle')
        .attr('class', 'overlay-point')
        .attr('cx', (point) => xScale(point[0]))
        .attr('cy', (point) => yScale(point[1]))
        .attr('r', 3)
        .attr('fill', 'rgba(255, 196, 86, 0.86)');
    }

    if (showLabels) {
      g.selectAll('text.point-label')
        .data(points)
        .enter()
        .append('text')
        .attr('class', 'point-label')
        .attr('x', (point) => xScale(point[0]) + 6)
        .attr('y', (point) => yScale(point[1]) - 6)
        .attr('fill', 'rgba(255,255,255,0.72)')
        .attr('font-size', 13)
        .text((_, index) => index + 1);
    }

    if (showVectors) {
      const colors = ['rgba(125, 255, 178, 0.92)', 'rgba(255, 196, 86, 0.92)'];
      const axisGroup = g.append('g');
      principalVectors.forEach((vector, index) => {
        axisGroup.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(vector[0]))
          .attr('y2', yScale(vector[1]))
          .attr('stroke', colors[index])
          .attr('stroke-width', 2.5);

        axisGroup.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(-vector[0]))
          .attr('y2', yScale(-vector[1]))
          .attr('stroke', colors[index].replace('0.92', '0.55'))
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6 4');

        axisGroup.append('text')
          .attr('x', xScale(vector[0]) + 6)
          .attr('y', yScale(vector[1]) - 6)
          .attr('fill', colors[index])
          .attr('font-size', 14)
          .text(basisLabels[index]);
      });
    }
  }

  function drawScatterPlot3D({ container, points, principalVectors, showVectors, showLabels, overlayPoints, title, axisLabels, basisLabels }) {
    const combinedPoints = (overlayPoints ? points.concat(overlayPoints) : points.slice()).concat([[0, 0, 0]]);
    const bound = Math.max(1, d3.max(combinedPoints.flatMap((point) => point.map((value) => Math.abs(value)))) || 1);
    const axisLength = bound * 1.15;

    const traces = [
      {
        type: 'scatter3d',
        mode: showLabels ? 'markers+text' : 'markers',
        x: points.map((point) => point[0]),
        y: points.map((point) => point[1]),
        z: points.map((point) => point[2]),
        text: showLabels ? points.map((_, index) => String(index + 1)) : undefined,
        textposition: 'top center',
        textfont: {
          color: 'rgba(255,255,255,0.9)',
          size: 13,
        },
        marker: {
          size: 4,
          color: 'rgba(74, 163, 255, 0.9)',
        },
        hovertemplate: '(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
        showlegend: false,
      },
    ];

    if (overlayPoints) {
      traces.push({
        type: 'scatter3d',
        mode: 'markers',
        x: overlayPoints.map((point) => point[0]),
        y: overlayPoints.map((point) => point[1]),
        z: overlayPoints.map((point) => point[2]),
        marker: {
          size: 3.5,
          color: 'rgba(255, 196, 86, 0.9)',
        },
        hovertemplate: 'recon (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>',
        showlegend: false,
      });
    }

    traces.push(
      makeAxisTrace([axisLength, 0, 0], 'rgba(255,255,255,0.22)', axisLabels[0]),
      makeAxisTrace([0, axisLength, 0], 'rgba(255,255,255,0.18)', axisLabels[1]),
      makeAxisTrace([0, 0, axisLength], 'rgba(255,255,255,0.14)', axisLabels[2])
    );

    if (showVectors) {
      const colors = ['rgba(125, 255, 178, 0.95)', 'rgba(255, 196, 86, 0.95)', 'rgba(255, 122, 122, 0.95)'];
      principalVectors.forEach((vector, index) => {
        traces.push(makeVectorTrace(vector, colors[index], basisLabels[index]));
      });
    }

    renderPlotly(container, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 0, r: 0, b: 0, t: 0 },
      uirevision: `pca-data-3d-${get3DAspectMode()}`,
      scene: {
        aspectmode: get3DAspectMode(),
        camera: getLiveCamera(container),
        xaxis: {
          title: axisLabels[0],
          range: [-axisLength, axisLength],
          backgroundcolor: 'rgba(255,255,255,0.02)',
          gridcolor: 'rgba(255,255,255,0.08)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          color: 'rgba(255,255,255,0.72)',
        },
        yaxis: {
          title: axisLabels[1],
          range: [-axisLength, axisLength],
          backgroundcolor: 'rgba(255,255,255,0.02)',
          gridcolor: 'rgba(255,255,255,0.08)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          color: 'rgba(255,255,255,0.72)',
        },
        zaxis: {
          title: axisLabels[2],
          range: [-axisLength, axisLength],
          backgroundcolor: 'rgba(255,255,255,0.02)',
          gridcolor: 'rgba(255,255,255,0.08)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          color: 'rgba(255,255,255,0.72)',
        },
      },
    });
  }

  function drawScatterPlot(args) {
    const dimension = args.points[0]?.length || 2;
    if (dimension === 3) {
      drawScatterPlot3D(args);
      return;
    }
    drawScatterPlot2D(args);
  }

  function buildSphereWireframe(transform) {
    const traces = [];
    const step = 12;
    const latitudes = [-60, -30, 0, 30, 60];
    const longitudes = [0, 30, 60, 90, 120, 150];

    latitudes.forEach((latDeg) => {
      const lat = degToRad(latDeg);
      const points = [];
      for (let lonDeg = 0; lonDeg <= 360; lonDeg += step) {
        const lon = degToRad(lonDeg);
        const base = [
          Math.cos(lat) * Math.cos(lon),
          Math.cos(lat) * Math.sin(lon),
          Math.sin(lat),
        ];
        points.push(applyMatrix(transform, base));
      }
      traces.push({
        type: 'scatter3d',
        mode: 'lines',
        x: points.map((point) => point[0]),
        y: points.map((point) => point[1]),
        z: points.map((point) => point[2]),
        line: { color: 'rgba(74, 163, 255, 0.58)', width: 4 },
        hoverinfo: 'skip',
        showlegend: false,
      });
    });

    longitudes.forEach((lonDeg) => {
      const lon = degToRad(lonDeg);
      const points = [];
      for (let latDeg = -90; latDeg <= 90; latDeg += step) {
        const lat = degToRad(latDeg);
        const base = [
          Math.cos(lat) * Math.cos(lon),
          Math.cos(lat) * Math.sin(lon),
          Math.sin(lat),
        ];
        points.push(applyMatrix(transform, base));
      }
      traces.push({
        type: 'scatter3d',
        mode: 'lines',
        x: points.map((point) => point[0]),
        y: points.map((point) => point[1]),
        z: points.map((point) => point[2]),
        line: { color: 'rgba(255,255,255,0.26)', width: 3 },
        hoverinfo: 'skip',
        showlegend: false,
      });
    });

    return traces;
  }

  function drawOperatorPlot2D({ container, transform, principalVectors, lambda, showVectors, title }) {
    clear(container);
    container.dataset.renderer = 'svg';
    const { width, height } = getPlotSize(container, 460);
    const margin = { top: 6, right: 18, bottom: 34, left: 44 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const circle = d3.range(0, 361).map((deg) => {
      const theta = degToRad(deg);
      const point = [Math.cos(theta), Math.sin(theta)];
      return applyMatrix(transform, point);
    });

    const transformedVectors = principalVectors.map((vector) => applyMatrix(transform, vector));
    const xs = circle.map((point) => point[0]).concat(transformedVectors.map((point) => point[0]), [0]);
    const ys = circle.map((point) => point[1]).concat(transformedVectors.map((point) => point[1]), [0]);
  const xDomain = extentWithPadding(xs);
  const yDomain = extentWithPadding(ys);
  const xCenter = (xDomain[0] + xDomain[1]) / 2;
  const yCenter = (yDomain[0] + yDomain[1]) / 2;
  const xSpan = Math.max(1e-6, xDomain[1] - xDomain[0]);
  const ySpan = Math.max(1e-6, yDomain[1] - yDomain[0]);
  const unitsPerPixel = Math.max(xSpan / innerWidth, ySpan / innerHeight);
  const halfWidthUnits = unitsPerPixel * innerWidth / 2;
  const halfHeightUnits = unitsPerPixel * innerHeight / 2;
  const xScale = d3.scaleLinear().domain([xCenter - halfWidthUnits, xCenter + halfWidthUnits]).range([0, innerWidth]);
  const yScale = d3.scaleLinear().domain([yCenter - halfHeightUnits, yCenter + halfHeightUnits]).range([innerHeight, 0]);

    const svg = d3.select(container).append('svg').attr('width', '100%').attr('height', height);
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g').attr('transform', `translate(0,${yScale(0)})`).call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
    g.append('g').attr('transform', `translate(${xScale(0)},0)`).call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));

    const line = d3.line().x((point) => xScale(point[0])).y((point) => yScale(point[1])).curve(d3.curveLinearClosed);

    g.append('path')
      .datum(circle)
      .attr('fill', 'rgba(255,255,255,0.06)')
      .attr('stroke', 'rgba(74, 163, 255, 0.9)')
      .attr('stroke-width', 2)
      .attr('d', line);

    if (showVectors) {
      transformedVectors.forEach((vector, index) => {
        const colors = ['rgba(125, 255, 178, 0.92)', 'rgba(255, 196, 86, 0.92)'];
        g.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(vector[0]))
          .attr('y2', yScale(vector[1]))
          .attr('stroke', colors[index])
          .attr('stroke-width', 2.5);

        g.append('text')
          .attr('x', xScale(vector[0]) + 6)
          .attr('y', yScale(vector[1]) - 6)
          .attr('fill', colors[index])
          .attr('font-size', 12)
          .text(`λ${index + 1}=${lambda[index].toFixed(2)}`);
      });
    }
  }

  function drawOperatorPlot3D({ container, transform, principalVectors, lambda, showVectors, title }) {
    const traces = buildSphereWireframe(transform);
    if (showVectors) {
      const colors = ['rgba(125, 255, 178, 0.95)', 'rgba(255, 196, 86, 0.95)', 'rgba(255, 122, 122, 0.95)'];
      principalVectors.forEach((vector, index) => {
        const transformed = applyMatrix(transform, vector);
        traces.push(makeVectorTrace(transformed, colors[index], `λ${index + 1}=${lambda[index].toFixed(2)}`));
      });
    }

    renderPlotly(container, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 0, r: 0, b: 0, t: 0 },
      uirevision: `pca-operator-3d-${get3DAspectMode()}`,
      scene: {
        aspectmode: get3DAspectMode(),
        camera: getLiveCamera(container),
        xaxis: {
          title: 'x₁',
          backgroundcolor: 'rgba(255,255,255,0.02)',
          gridcolor: 'rgba(255,255,255,0.08)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          color: 'rgba(255,255,255,0.72)',
        },
        yaxis: {
          title: 'x₂',
          backgroundcolor: 'rgba(255,255,255,0.02)',
          gridcolor: 'rgba(255,255,255,0.08)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          color: 'rgba(255,255,255,0.72)',
        },
        zaxis: {
          title: 'x₃',
          backgroundcolor: 'rgba(255,255,255,0.02)',
          gridcolor: 'rgba(255,255,255,0.08)',
          zerolinecolor: 'rgba(255,255,255,0.18)',
          color: 'rgba(255,255,255,0.72)',
        },
      },
    });
  }

  function drawOperatorPlot(args) {
    if (args.principalVectors[0]?.length === 3) {
      drawOperatorPlot3D(args);
      return;
    }
    drawOperatorPlot2D(args);
  }

  function getAxisLabels(step, dimension) {
    if (step === 1 || step === 2) {
      return Array.from({ length: dimension }, (_, index) => `pc${index + 1}`);
    }
    return Array.from({ length: dimension }, (_, index) => `x${index + 1}`);
  }

  function getDisplayVectors(step, decomposition, principalVectors) {
    if (step === 1) {
      return principalVectors.map((_, index) => Array.from({ length: decomposition.dimension }, (_, axisIndex) => (index === axisIndex ? 1 : 0)));
    }

    if (step === 2) {
      return decomposition.lambda.map((value, index) => (
        Array.from({ length: decomposition.dimension }, (_, axisIndex) => (index === axisIndex ? value : 0))
      ));
    }

    if (step === 3) {
      return principalVectors.map((vector, index) => vector.map((value) => value * decomposition.lambda[index]));
    }

    return principalVectors;
  }

  function getBasisLabels(step, dimension) {
    if (step === 1 || step === 2) {
      return Array.from({ length: dimension }, (_, index) => `pc${index + 1}`);
    }
    if (step === 3) {
      return Array.from({ length: dimension }, (_, index) => `λ${index + 1}v${index + 1}`);
    }
    return Array.from({ length: dimension }, (_, index) => `v${index + 1}`);
  }

  function syncRankOptions() {
    if (!rankInput) return;
    const dimension = getDimension();
    const current = Number(rankInput.value) || dimension;
    const ranks = dimension === 3 ? [3, 2, 1] : [2, 1];
    const labels = {
      3: 'Keep 3 components',
      2: 'Keep 2 components',
      1: 'Keep 1 component',
    };
    rankInput.innerHTML = ranks.map((rank) => `<option value="${rank}">${labels[rank]}</option>`).join('');
    rankInput.value = String(ranks.includes(current) ? current : ranks[0]);
  }

  function renderLegend(container, items) {
    if (!container) return;
    container.innerHTML = items.map((item) => {
      const swatch = item.color
        ? `<span class="pca-legend-swatch" style="--legend-color: ${item.color}"></span>`
        : '<span class="pca-legend-swatch pca-legend-swatch-muted"></span>';
      return `<div class="pca-legend-item">${swatch}<div><strong>${item.label}</strong><p>${item.description}</p></div></div>`;
    }).join('');
  }

  function getDataLegendItems({ dimension, step, showVectors, showLabels, showOverlay, rank }) {
    const items = [
      {
        color: 'rgba(74, 163, 255, 0.9)',
        label: 'Blue data cloud',
        description: step === 0
          ? `Centered sample points in ${dimension}D before any PCA transform is applied.`
          : `The sample points after the current PCA stage is applied in ${dimension}D.`,
      },
      {
        color: 'rgba(255,255,255,0.22)',
        label: 'White coordinate axes',
        description: step <= 2
          ? 'Reference axes for the current coordinate system shown in the plot.'
          : 'Reference axes for the ambient feature coordinates after rotating back.',
      },
    ];

    if (showOverlay) {
      items.push({
        color: 'rgba(255, 196, 86, 0.9)',
        label: 'Gold reconstruction',
        description: `Low-rank reconstruction keeping the first ${rank} principal ${rank === 1 ? 'coordinate' : 'coordinates'}.`,
      });
    }

    if (showVectors) {
      items.push({
        color: 'rgba(125, 255, 178, 0.95)',
        label: 'Green dominant axis',
        description: step <= 1
          ? 'First principal direction, the axis along which variance is largest.'
          : 'First transformed direction after the current operator is applied.',
      });
      items.push({
        color: 'rgba(255, 196, 86, 0.95)',
        label: 'Gold secondary axis',
        description: dimension === 3
          ? 'Second principal direction, orthogonal to the dominant axis but still variance-carrying.'
          : 'Second principal direction, orthogonal to the dominant axis.',
      });
      if (dimension === 3) {
        items.push({
          color: 'rgba(255, 122, 122, 0.95)',
          label: 'Red tertiary axis',
          description: 'Third principal direction, capturing the remaining orthogonal variation.',
        });
      }
    }

    if (showLabels) {
      items.push({
        label: 'Point numbers',
        description: 'Sample indices shown to help track how individual points move through the transform.',
      });
    }

    return items;
  }

  function getOperatorLegendItems({ dimension, showVectors }) {
    const items = [
      {
        color: 'rgba(74, 163, 255, 0.7)',
        label: dimension === 3 ? 'Blue latitude bands' : 'Blue transformed circle',
        description: dimension === 3
          ? 'Latitude-like curves from the unit sphere after the current covariance-related operator is applied.'
          : 'A unit circle after the current covariance-related operator is applied.',
      },
      {
        color: 'rgba(255,255,255,0.26)',
        label: dimension === 3 ? 'White meridian bands' : 'White reference axes',
        description: dimension === 3
          ? 'Longitude-like curves on the same transformed sphere, included to reveal the surface geometry.'
          : 'Coordinate axes used as a fixed frame for reading the operator action.',
      },
    ];

    if (showVectors) {
      items.push({
        color: 'rgba(125, 255, 178, 0.95)',
        label: 'Green λ₁ direction',
        description: 'The first principal direction after transformation, annotated by its eigenvalue.',
      });
      items.push({
        color: 'rgba(255, 196, 86, 0.95)',
        label: 'Gold λ₂ direction',
        description: 'The second principal direction after transformation, annotated by its eigenvalue.',
      });
      if (dimension === 3) {
        items.push({
          color: 'rgba(255, 122, 122, 0.95)',
          label: 'Red λ₃ direction',
          description: 'The third principal direction after transformation, annotated by its eigenvalue.',
        });
      }
    }

    return items;
  }

  function render() {
    syncDimensionControls();
    syncRankOptions();

    let points;
    let decomposition;
    try {
      points = buildDataset();
      decomposition = computePCA(points);
    } catch (error) {
      clear(dataContainer);
      clear(operatorContainer);
      const message = error && error.message ? error.message : String(error);
      stepSummaryEl.textContent = message;
      matrixXEl.textContent = '-';
      matrixFactorsEl.textContent = '-';
      matrixCovEl.textContent = '-';
      matrixCurrentEl.textContent = '-';
      return;
    }

    const step = Math.max(0, Math.min(3, Number(stepInput.value) || 0));
    const stepMeta = getStepMeta(step, decomposition);
    const currentTransform = stepMeta.transform;
    const transformedPoints = points.map((point) => applyMatrix(currentTransform, point));
    const principalVectors = Array.from({ length: decomposition.dimension }, (_, index) => decomposition.V.map((row) => row[index]));
    const showVectors = !!showVectorsInput.checked;
    const showLabels = !!showLabelsInput.checked;
    const targetRank = Math.max(1, Math.min(Number(rankInput.value) || decomposition.dimension, decomposition.dimension));
    const overlayPoints = showRank1Input.checked ? getRankReconstruction(decomposition, targetRank).map((point) => applyMatrix(currentTransform, point)) : null;
    const axisLabels = getAxisLabels(step, decomposition.dimension);
    const displayVectors = getDisplayVectors(step, decomposition, principalVectors);
    const basisLabels = getBasisLabels(step, decomposition.dimension);

    drawScatterPlot({
      container: dataContainer,
      points: transformedPoints,
      principalVectors: displayVectors,
      showVectors,
      showLabels,
      overlayPoints,
      title: stepMeta.name,
      axisLabels,
      basisLabels,
    });

    drawOperatorPlot({
      container: operatorContainer,
      transform: currentTransform,
      principalVectors,
      lambda: decomposition.lambda,
      showVectors,
      title: 'How the covariance operator acts on directions',
    });

    if (decomposition.dimension === 3) {
      sync3DPlots();
    }

    renderLegend(dataLegendEl, getDataLegendItems({
      dimension: decomposition.dimension,
      step,
      showVectors,
      showLabels,
      showOverlay: !!overlayPoints,
      rank: targetRank,
    }));
    renderLegend(operatorLegendEl, getOperatorLegendItems({
      dimension: decomposition.dimension,
      showVectors,
    }));

    renderStepSummary(stepMeta, decomposition);

    matrixXEl.textContent = formatMatrix(decomposition.X, 8);
    matrixFactorsEl.textContent = `Σ =\n${formatMatrix(decomposition.Sigma)}\n\nV =\n${formatMatrix(decomposition.V)}\n\nV^T =\n${formatMatrix(decomposition.VT)}`;
    matrixCovEl.textContent = `Λ =\n${formatMatrix(decomposition.Lambda)}\n\nC = X^T X / (n-1) =\n${formatMatrix(decomposition.covariance)}`;
    matrixCurrentEl.textContent = `${stepMeta.name}\n\n${formatMatrix(currentTransform)}${showRank1Input.checked ? `\n\nLow-rank overlay keeps the first ${targetRank} principal ${targetRank === 1 ? 'coordinate' : 'coordinates'}.` : ''}`;
  }

  const debouncedRender = debounce(render, 220);
  [
    dimensionInput,
    presetInput,
    samplesInput,
    stepInput,
    rankInput,
    showVectorsInput,
    showLabelsInput,
    showRank1Input,
    useCubeAspectInput,
    syncCamerasInput,
  ].forEach((element) => {
    element?.addEventListener('input', debouncedRender);
    element?.addEventListener('change', render);
  });

  randomizeBtn?.addEventListener('click', () => {
    seed += 1;
    render();
  });

  controlTabButtons.forEach((button) => {
    button.addEventListener('click', () => {
      setActiveControlTab(button.dataset.tabTarget || 'dataset');
    });
  });

  spread1Input?.addEventListener('input', () => {
    syncNumericLabel(spread1Input, spread1ValueEl);
    render();
  });
  spread1Input?.addEventListener('change', () => {
    syncNumericLabel(spread1Input, spread1ValueEl);
    render();
  });

  spread2Input?.addEventListener('input', () => {
    syncNumericLabel(spread2Input, spread2ValueEl);
    render();
  });
  spread2Input?.addEventListener('change', () => {
    syncNumericLabel(spread2Input, spread2ValueEl);
    render();
  });

  spread3Input?.addEventListener('input', () => {
    syncNumericLabel(spread3Input, spread3ValueEl);
    render();
  });
  spread3Input?.addEventListener('change', () => {
    syncNumericLabel(spread3Input, spread3ValueEl);
    render();
  });

  angleInput?.addEventListener('input', syncAngleLabel);
  angleInput?.addEventListener('input', render);
  angleInput?.addEventListener('change', syncAngleLabel);
  angleInput?.addEventListener('change', render);

  elevationInput?.addEventListener('input', () => {
    syncNumericLabel(elevationInput, elevationValueEl, 0, '°');
    render();
  });
  elevationInput?.addEventListener('change', () => {
    syncNumericLabel(elevationInput, elevationValueEl, 0, '°');
    render();
  });

  noiseInput?.addEventListener('input', () => {
    syncNumericLabel(noiseInput, noiseValueEl);
    render();
  });
  noiseInput?.addEventListener('change', () => {
    syncNumericLabel(noiseInput, noiseValueEl);
    render();
  });

  window.addEventListener('resize', () => {
    if (render._t) clearTimeout(render._t);
    render._t = setTimeout(render, 120);
  });

  syncAngleLabel();
  syncNumericLabel(spread1Input, spread1ValueEl);
  syncNumericLabel(spread2Input, spread2ValueEl);
  syncNumericLabel(spread3Input, spread3ValueEl);
  syncNumericLabel(elevationInput, elevationValueEl, 0, '°');
  syncNumericLabel(noiseInput, noiseValueEl);
  setActiveControlTab('dataset');
  syncDimensionControls();
  syncRankOptions();
  render();
});