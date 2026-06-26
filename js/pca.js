import { createDataPlot3D, createOperatorPlot3D } from './pca-3d.js';

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
  const syncCamerasInput = document.getElementById('pcaSyncCameras');
  const randomizeBtn = document.getElementById('pcaRandomize');
  const controlTabButtons = Array.from(document.querySelectorAll('.pca-control-tab'));
  const controlTabPanels = Array.from(document.querySelectorAll('.pca-control-panel'));

  const dataLegendEl = document.getElementById('pcaDataLegend');
  const operatorLegendEl = document.getElementById('pcaOperatorLegend');
  const matrixFactorsEl = document.getElementById('pcaMatrixFactors');
  const matrixCovEl = document.getElementById('pcaMatrixCov');

  let seed = 7;
  const GRAPH_FONT_SIZE = 14;

  let dataPlot = null;
  let operatorPlot = null;

  function ensureContextsCreated() {
    if (dataPlot && operatorPlot) return;
    try {
      dataPlot = createDataPlot3D(dataContainer);
      operatorPlot = createOperatorPlot3D(operatorContainer);
      dataPlot.onCameraChange((phi, theta) => {
        if (shouldSync3DCameras()) operatorPlot.applyCameraDir(phi, theta);
      });
      operatorPlot.onCameraChange((phi, theta) => {
        if (shouldSync3DCameras()) dataPlot.applyCameraDir(phi, theta);
      });
    } catch (_err) {
      if (dataPlot) { dataPlot.destroy(); }
      dataContainer.textContent = 'WebGL is not available in this browser.';
      operatorContainer.textContent = '';
      dataPlot = null;
      operatorPlot = null;
    }
  }

  function ensureContextsDestroyed() {
    if (dataPlot) { dataPlot.destroy(); dataPlot = null; }
    if (operatorPlot) { operatorPlot.destroy(); operatorPlot = null; }
  }

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

  function shouldSync3DCameras() {
    return !!syncCamerasInput?.checked;
  }

  function debounce(fn, delay = 180) {
    let timer = null;
    const d = (...args) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
    d.cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
    return d;
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
    if (el) {
      el.innerHTML = '';
      delete el.dataset.renderer;
    }
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

  function sampleUniformDirection(dimension, rng) {
    const direction = Array.from(
      { length: dimension },
      () => sampleStandardNormal(rng),
    );
    const length = Math.sqrt(
      direction.reduce((sum, value) => sum + value * value, 0),
    );

    if (length < 1e-9) {
      return sampleUniformDirection(dimension, rng);
    }

    return direction.map((value) => value / length);
  }

  function sampleUniformBallPoint(radii, rng) {
    const direction = sampleUniformDirection(radii.length, rng);
    const radius = Math.pow(rng(), 1 / radii.length);
    return direction.map((value, index) => value * radius * radii[index]);
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
        } else if (preset === 'sphere') {
          localPoint = sampleUniformBallPoint(
            [spread1, spread2, spread3],
            rng,
          );
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
        } else if (preset === 'sphere') {
          localPoint = sampleUniformBallPoint([spread1, spread2], rng);
        } else {
          localPoint = [
            sampleStandardNormal(rng) * spread1,
            sampleStandardNormal(rng) * spread2,
          ];
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
    // Follow the container's definite height (.pca-viz is height:400) so the 2D SVG plot fills
    // its box exactly and is not clipped; fall back to minHeight if the box has no height yet.
    return { width: Math.max(320, rect.width), height: Math.round(rect.height) || minHeight };
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

  function getStepName(step) {
    if (step === 1) return 'Rotate by V^T';
    if (step === 2) return 'Stretch by Λ';
    if (step === 3) return 'Rotate back by V';
    return 'Original centered data';
  }

  function extentWithPadding(values) {
    const minValue = d3.min(values);
    const maxValue = d3.max(values);
    const span = Math.max(1e-6, maxValue - minValue);
    const pad = span * 0.18;
    return [minValue - pad, maxValue + pad];
  }

  function styleSvgAxis(axisSelection) {
    axisSelection.selectAll('text').attr('font-size', GRAPH_FONT_SIZE);
  }

  function ensureSvgPlotFrame(container, minHeight = 460) {
    if (container.dataset.renderer && container.dataset.renderer !== 'svg') {
      clear(container);
    }

    container.dataset.renderer = 'svg';
    const { width, height } = getPlotSize(container, minHeight);
    const margin = { top: 6, right: 18, bottom: 34, left: 44 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = d3.select(container)
      .selectAll('svg.pca-svg-root')
      .data([null])
      .join('svg')
      .attr('class', 'pca-svg-root')
      .attr('width', '100%')
      .attr('height', height);

    const root = svg.selectAll('g.plot-root')
      .data([null])
      .join('g')
      .attr('class', 'plot-root')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const axisX = root.selectAll('g.axis-x')
      .data([null])
      .join('g')
      .attr('class', 'axis-x');

    const axisY = root.selectAll('g.axis-y')
      .data([null])
      .join('g')
      .attr('class', 'axis-y');

    return {
      svg,
      root,
      axisX,
      axisY,
      margin,
      width,
      height,
      innerWidth,
      innerHeight,
    };
  }

  function setSvgAxisTitle(selection, attrs) {
    if (!attrs) {
      selection.remove();
      return;
    }

    selection
      .attr('x', attrs.x)
      .attr('y', attrs.y)
      .attr('transform', attrs.transform || null)
      .attr('text-anchor', attrs.anchor || 'middle')
      .attr('fill', 'rgba(255,255,255,0.75)')
      .attr('font-size', GRAPH_FONT_SIZE)
      .text(attrs.text);
  }

  function drawScatterPlot2D({ container, points, principalVectors, showVectors, showLabels, overlayPoints, title, axisLabels, basisLabels }) {
    const {
      root,
      axisX,
      axisY,
      innerWidth,
      innerHeight,
    } = ensureSvgPlotFrame(container, 460);

    const vectorPoints = showVectors ? principalVectors.flatMap((vector) => [vector, vector.map((value) => -value)]) : [];
    const allPoints = (overlayPoints ? points.concat(overlayPoints) : points.slice()).concat(vectorPoints);
    const xs = allPoints.map((point) => point[0]).concat([0]);
    const ys = allPoints.map((point) => point[1]).concat([0]);
    const xScale = d3.scaleLinear().domain(extentWithPadding(xs)).range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(extentWithPadding(ys)).range([innerHeight, 0]);

    axisX
      .attr('transform', `translate(0,${yScale(0)})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
    styleSvgAxis(axisX);

    axisY
      .attr('transform', `translate(${xScale(0)},0)`)
      .call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
    styleSvgAxis(axisY);

    setSvgAxisTitle(
      root.selectAll('text.axis-title-x').data([null]).join('text').attr('class', 'axis-title-x'),
      {
        x: innerWidth / 2,
        y: innerHeight + 30,
        text: axisLabels[0],
      }
    );

    setSvgAxisTitle(
      root.selectAll('text.axis-title-y').data([null]).join('text').attr('class', 'axis-title-y'),
      {
        x: -innerHeight / 2,
        y: -34,
        transform: 'rotate(-90)',
        text: axisLabels[1],
      }
    );

    const scatterLayer = root.selectAll('g.scatter-layer')
      .data([null])
      .join('g')
      .attr('class', 'scatter-layer');

    scatterLayer
      .selectAll('circle.data-point')
      .data(points)
      .join('circle')
      .attr('class', 'data-point')
      .attr('cx', (point) => xScale(point[0]))
      .attr('cy', (point) => yScale(point[1]))
      .attr('r', 4)
      .attr('fill', 'rgba(74, 163, 255, 0.88)');

    const overlayLayer = root.selectAll('g.overlay-layer')
      .data([null])
      .join('g')
      .attr('class', 'overlay-layer');

    overlayLayer
      .selectAll('circle.overlay-point')
      .data(overlayPoints || [])
      .join('circle')
        .attr('class', 'overlay-point')
        .attr('cx', (point) => xScale(point[0]))
        .attr('cy', (point) => yScale(point[1]))
        .attr('r', 3)
        .attr('fill', 'rgba(255, 196, 86, 0.86)');

    const pointLabelLayer = root.selectAll('g.point-label-layer')
      .data([null])
      .join('g')
      .attr('class', 'point-label-layer');

    pointLabelLayer
      .selectAll('text.point-label')
      .data(showLabels ? points : [], (_, index) => index)
      .join('text')
        .attr('class', 'point-label')
        .attr('x', (point) => xScale(point[0]) + 6)
        .attr('y', (point) => yScale(point[1]) - 6)
        .attr('fill', 'rgba(255,255,255,0.72)')
        .attr('font-size', GRAPH_FONT_SIZE)
        .text((_, index) => index + 1);

    const colors = ['rgba(125, 255, 178, 0.92)', 'rgba(255, 196, 86, 0.92)'];
    const vectorGroups = root.selectAll('g.vector-layer').data([null]).join('g').attr('class', 'vector-layer')
      .selectAll('g.principal-vector')
      .data(showVectors ? principalVectors : [], (_, index) => index)
      .join(
        (enter) => {
          const group = enter.append('g').attr('class', 'principal-vector');
          group.append('line').attr('class', 'vector-positive');
          group.append('line').attr('class', 'vector-negative');
          group.append('text').attr('class', 'vector-label');
          return group;
        },
        (update) => update,
        (exit) => exit.remove()
      );

    vectorGroups.each(function updateVector(vector, index) {
      const group = d3.select(this);
      const color = colors[index];

      group.select('line.vector-positive')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(vector[0]))
        .attr('y2', yScale(vector[1]))
        .attr('stroke', color)
        .attr('stroke-width', 2.5);

      group.select('line.vector-negative')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(-vector[0]))
        .attr('y2', yScale(-vector[1]))
        .attr('stroke', color.replace('0.92', '0.55'))
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '6 4');

      group.select('text.vector-label')
        .attr('x', xScale(vector[0]) + 6)
        .attr('y', yScale(vector[1]) - 6)
        .attr('fill', color)
        .attr('font-size', GRAPH_FONT_SIZE)
        .text(basisLabels[index]);
    });

    vectorGroups
      .filter((_, index) => index === 0)
      .raise();

    scatterLayer.raise();
    overlayLayer.raise();
    pointLabelLayer.raise();
  }

  function drawOperatorPlot2D({ container, transform, principalVectors, lambda, showVectors, title }) {
    const {
      root,
      axisX,
      axisY,
      innerWidth,
      innerHeight,
    } = ensureSvgPlotFrame(container, 460);

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
    const xScale = d3.scaleLinear()
      .domain([xCenter - halfWidthUnits, xCenter + halfWidthUnits])
      .range([0, innerWidth]);
    const yScale = d3.scaleLinear()
      .domain([yCenter - halfHeightUnits, yCenter + halfHeightUnits])
      .range([innerHeight, 0]);

    axisX
      .attr('transform', `translate(0,${yScale(0)})`)
      .call(d3.axisBottom(xScale).ticks(6).tickSizeOuter(0));
    styleSvgAxis(axisX);

    axisY
      .attr('transform', `translate(${xScale(0)},0)`)
      .call(d3.axisLeft(yScale).ticks(6).tickSizeOuter(0));
    styleSvgAxis(axisY);

    const line = d3.line().x((point) => xScale(point[0])).y((point) => yScale(point[1])).curve(d3.curveLinearClosed);

    root.selectAll('path.operator-shape')
      .data([circle])
      .join('path')
      .attr('class', 'operator-shape')
      .attr('fill', 'rgba(255,255,255,0.06)')
      .attr('stroke', 'rgba(74, 163, 255, 0.9)')
      .attr('stroke-width', 2)
      .attr('d', line);

    const colors = ['rgba(125, 255, 178, 0.92)', 'rgba(255, 196, 86, 0.92)'];
    const operatorVectors = root.selectAll('g.operator-vector-layer').data([null]).join('g').attr('class', 'operator-vector-layer')
      .selectAll('g.operator-vector')
      .data(showVectors ? transformedVectors : [], (_, index) => index)
      .join(
        (enter) => {
          const group = enter.append('g').attr('class', 'operator-vector');
          group.append('line').attr('class', 'operator-vector-line');
          group.append('text').attr('class', 'operator-vector-label');
          return group;
        },
        (update) => update,
        (exit) => exit.remove()
      );

    operatorVectors.each(function updateOperatorVector(vector, index) {
      const group = d3.select(this);
      const color = colors[index];

      group.select('line.operator-vector-line')
        .attr('x1', xScale(0))
        .attr('y1', yScale(0))
        .attr('x2', xScale(vector[0]))
        .attr('y2', yScale(vector[1]))
        .attr('stroke', color)
        .attr('stroke-width', 2.5);

      group.select('text.operator-vector-label')
        .attr('x', xScale(vector[0]) + 6)
        .attr('y', yScale(vector[1]) - 6)
        .attr('fill', color)
        .attr('font-size', GRAPH_FONT_SIZE)
        .text(`λ${index + 1}=${lambda[index].toFixed(2)}`);
    });

    operatorVectors
      .filter((_, index) => index === 0)
      .raise();
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
      matrixFactorsEl.textContent = '-';
      matrixCovEl.textContent = '-';
      return;
    }

    const step = Math.max(0, Math.min(3, Number(stepInput.value) || 0));
    const stepName = getStepName(step);
    const currentTransform = getCurrentTransform(step, decomposition);
    const transformedPoints = points.map((point) => applyMatrix(currentTransform, point));
    const principalVectors = Array.from({ length: decomposition.dimension }, (_, index) => decomposition.V.map((row) => row[index]));
    const showVectors = !!showVectorsInput.checked;
    const showLabels = !!showLabelsInput.checked;
    const targetRank = Math.max(1, Math.min(Number(rankInput.value) || decomposition.dimension, decomposition.dimension));
    const overlayPoints = showRank1Input.checked ? getRankReconstruction(decomposition, targetRank).map((point) => applyMatrix(currentTransform, point)) : null;
    const axisLabels = getAxisLabels(step, decomposition.dimension);
    const displayVectors = getDisplayVectors(step, decomposition, principalVectors);
    const basisLabels = getBasisLabels(step, decomposition.dimension);

    const combinedForBound = (overlayPoints ? transformedPoints.concat(overlayPoints) : transformedPoints).concat([[0,0,0]]);
    const bound = Math.max(1, d3.max(combinedForBound.flatMap(p => p.map(Math.abs))) || 1);

    if (decomposition.dimension === 3) {
      ensureContextsCreated();
      if (dataPlot) {
        dataPlot.update({
          points: transformedPoints,
          principalVectors: displayVectors,
          showVectors,
          showLabels,
          overlayPoints,
          axisLabels,
          basisLabels,
          bound,
        });
      }
      if (operatorPlot) {
        operatorPlot.update({
          transform: currentTransform,
          principalVectors,
          lambda: decomposition.lambda,
          showVectors,
        });
      }
    } else {
      ensureContextsDestroyed();
      drawScatterPlot2D({
        container: dataContainer,
        points: transformedPoints,
        principalVectors: displayVectors,
        showVectors,
        showLabels,
        overlayPoints,
        axisLabels,
        basisLabels,
      });
      drawOperatorPlot2D({
        container: operatorContainer,
        transform: currentTransform,
        principalVectors,
        lambda: decomposition.lambda,
        showVectors,
        title: 'How the covariance operator acts on directions',
      });
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

    matrixFactorsEl.textContent = `X =\n${formatMatrix(decomposition.X, 8)}\n\nΣ =\n${formatMatrix(decomposition.Sigma, 8)}`;
    matrixCovEl.textContent = `V =\n${formatMatrix(decomposition.V, 8)}\n\nΛ =\n${formatMatrix(decomposition.Lambda, 8)}\n\nV^T =\n${formatMatrix(decomposition.VT, 8)}\n\nC =\n${formatMatrix(decomposition.covariance, 8)}`;
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

  angleInput?.addEventListener('input', () => { syncAngleLabel(); render(); });
  angleInput?.addEventListener('change', () => { syncAngleLabel(); render(); });

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