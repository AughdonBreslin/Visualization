document.addEventListener('DOMContentLoaded', () => {
  const dataContainer = document.getElementById('pcaDataViz');
  const operatorContainer = document.getElementById('pcaOperatorViz');
  if (!dataContainer || !operatorContainer) return;

  const presetInput = document.getElementById('pcaPreset');
  const samplesInput = document.getElementById('pcaSamples');
  const spread1Input = document.getElementById('pcaSpread1');
  const spread1ValueEl = document.getElementById('pcaSpread1Value');
  const spread2Input = document.getElementById('pcaSpread2');
  const spread2ValueEl = document.getElementById('pcaSpread2Value');
  const angleInput = document.getElementById('pcaAngle');
  const angleValueEl = document.getElementById('pcaAngleValue');
  const noiseInput = document.getElementById('pcaNoise');
  const noiseValueEl = document.getElementById('pcaNoiseValue');
  const stepInput = document.getElementById('pcaStep');
  const rankInput = document.getElementById('pcaRank');
  const showVectorsInput = document.getElementById('pcaShowVectors');
  const showLabelsInput = document.getElementById('pcaShowLabels');
  const showRank1Input = document.getElementById('pcaShowRank1');
  const randomizeBtn = document.getElementById('pcaRandomize');

  const stepSummaryEl = document.getElementById('pcaStepSummary');
  const matrixXEl = document.getElementById('pcaMatrixX');
  const matrixFactorsEl = document.getElementById('pcaMatrixFactors');
  const matrixCovEl = document.getElementById('pcaMatrixCov');
  const matrixCurrentEl = document.getElementById('pcaMatrixCurrent');

  let seed = 7;

  function syncAngleLabel() {
    if (!angleValueEl || !angleInput) return;
    const angle = Number(angleInput.value);
    angleValueEl.textContent = `${Number.isFinite(angle) ? Math.round(angle) : 0}°`;
  }

  function syncNumericLabel(input, output, digits = 2) {
    if (!input || !output) return;
    const value = Number(input.value);
    output.textContent = Number.isFinite(value) ? value.toFixed(digits) : '0.00';
  }

  function clampPositive(x, minValue = 1e-6) {
    if (!Number.isFinite(x)) return minValue;
    return Math.max(minValue, x);
  }

  function debounce(fn, delay = 180) {
    let timer = null;
    return (...args) => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function clear(el) {
    d3.select(el).selectAll('*').remove();
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

  function rotationMatrix(theta) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return [
      [c, -s],
      [s, c],
    ];
  }

  function transpose2(matrix) {
    return [
      [matrix[0][0], matrix[1][0]],
      [matrix[0][1], matrix[1][1]],
    ];
  }

  function multiply2(a, b) {
    return [
      [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
      [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ];
  }

  function diagonal2(a, b) {
    return [
      [a, 0],
      [0, b],
    ];
  }

  function apply2(matrix, point) {
    return [
      matrix[0][0] * point[0] + matrix[0][1] * point[1],
      matrix[1][0] * point[0] + matrix[1][1] * point[1],
    ];
  }

  function mean(values) {
    return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
  }

  function centerPoints(points) {
    const meanX = mean(points.map((point) => point[0]));
    const meanY = mean(points.map((point) => point[1]));
    return {
      centered: points.map((point) => [point[0] - meanX, point[1] - meanY]),
      mean: [meanX, meanY],
    };
  }

  function buildDataset() {
    const preset = presetInput.value;
    const samples = Math.max(8, Math.floor(Number(samplesInput.value) || 24));
    const spread1 = clampPositive(Number(spread1Input.value), 0.1);
    const spread2 = clampPositive(Number(spread2Input.value), 0.05);
    const noise = Math.max(0, Number(noiseInput.value) || 0);
    const theta = degToRad(Number(angleInput.value) || 0);
    const rotation = rotationMatrix(theta);
    const rng = mulberry32(seed);
    const raw = [];

    for (let i = 0; i < samples; i++) {
      let localPoint;
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

      const rotated = apply2(rotation, localPoint);
      raw.push([
        rotated[0] + sampleStandardNormal(rng) * noise,
        rotated[1] + sampleStandardNormal(rng) * noise,
      ]);
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

    const X = points.map((point) => [point[0], point[1]]);
    const svd = numeric.svd(X);
    const singularValues = svd.S.map((value) => roundSmall(value));
    const V = svd.V.map((row) => row.map(roundSmall));
    const VT = transpose2(V);
    const lambda = singularValues.map((value) => (value * value) / Math.max(1, points.length - 1));
    const Lambda = diagonal2(lambda[0], lambda[1]);
    const covariance = multiply2(multiply2(V, Lambda), VT);
    const scores = points.map((point) => apply2(VT, point));
    const rank1Scores = scores.map((point) => [point[0], 0]);
    const rank1Reconstruction = rank1Scores.map((point) => apply2(V, point));

    return {
      X,
      U: svd.U.map((row) => row.map(roundSmall)),
      singularValues,
      Sigma: diagonal2(singularValues[0], singularValues[1]),
      V,
      VT,
      lambda,
      Lambda,
      covariance,
      scores,
      rank1Reconstruction,
    };
  }

  function getCurrentTransform(step, decomposition) {
    if (step === 1) return decomposition.VT;
    if (step === 2) return multiply2(decomposition.Lambda, decomposition.VT);
    if (step === 3) return multiply2(decomposition.V, multiply2(decomposition.Lambda, decomposition.VT));
    return diagonal2(1, 1);
  }

  function getStepMeta(step, decomposition) {
    if (step === 1) {
      return {
        name: 'Rotate by V^T',
        explanation: 'The data is rotated into the principal-axis coordinate system. This is the key PCA coordinate change: the cloud is now aligned with the directions of maximal and minimal variance.',
        transform: decomposition.VT,
      };
    }
    if (step === 2) {
      return {
        name: 'Stretch by Λ',
        explanation: 'After rotating into the eigenbasis, the covariance operator scales each coordinate by its eigenvalue. The first axis is stretched more when the first principal component explains more variance.',
        transform: multiply2(decomposition.Lambda, decomposition.VT),
      };
    }
    if (step === 3) {
      return {
        name: 'Rotate back by V',
        explanation: 'Applying V rotates the stretched result back into the original feature coordinates. This completes the covariance action VΛV^T.',
        transform: decomposition.covariance,
      };
    }
    return {
      name: 'Original centered data',
      explanation: 'The cloud has been centered so PCA measures variation around the mean. The principal directions shown here come from the right singular vectors in V.',
      transform: diagonal2(1, 1),
    };
  }

  function extentWithPadding(values) {
    const minValue = d3.min(values);
    const maxValue = d3.max(values);
    const span = Math.max(1e-6, maxValue - minValue);
    const pad = span * 0.18;
    return [minValue - pad, maxValue + pad];
  }

  function drawScatterPlot({
    container,
    points,
    principalVectors,
    showVectors,
    showLabels,
    overlayPoints,
    title,
    basisLabel,
  }) {
    clear(container);
    const { width, height } = getPlotSize(container, 320);
    const margin = { top: 18, right: 18, bottom: 34, left: 44 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const allPoints = overlayPoints ? points.concat(overlayPoints) : points.slice();
    const xs = allPoints.map((point) => point[0]);
    const ys = allPoints.map((point) => point[1]);
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
      .attr('font-size', 12)
      .text('x₁');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerHeight / 2)
      .attr('y', -34)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.75)')
      .attr('font-size', 12)
      .text('x₂');

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
        .attr('font-size', 11)
        .text((_, index) => index + 1);
    }

    if (showVectors) {
      const axisGroup = g.append('g');
      principalVectors.forEach((vector, index) => {
        axisGroup.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(vector[0]))
          .attr('y2', yScale(vector[1]))
          .attr('stroke', index === 0 ? 'rgba(125, 255, 178, 0.92)' : 'rgba(255, 196, 86, 0.92)')
          .attr('stroke-width', 2.5);

        axisGroup.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(-vector[0]))
          .attr('y2', yScale(-vector[1]))
          .attr('stroke', index === 0 ? 'rgba(125, 255, 178, 0.55)' : 'rgba(255, 196, 86, 0.55)')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '6 4');

        axisGroup.append('text')
          .attr('x', xScale(vector[0]) + 6)
          .attr('y', yScale(vector[1]) - 6)
          .attr('fill', index === 0 ? 'rgba(125, 255, 178, 0.92)' : 'rgba(255, 196, 86, 0.92)')
          .attr('font-size', 12)
          .text(index === 0 ? `${basisLabel}₁` : `${basisLabel}₂`);
      });
    }

    svg.append('text')
      .attr('x', margin.left)
      .attr('y', 14)
      .attr('fill', 'rgba(255,255,255,0.88)')
      .attr('font-size', 12)
      .attr('font-weight', 650)
      .text(title);
  }

  function drawOperatorPlot({ container, transform, principalVectors, lambda, showVectors, title }) {
    clear(container);
    const { width, height } = getPlotSize(container, 320);
    const margin = { top: 18, right: 18, bottom: 34, left: 44 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const circle = d3.range(0, 361).map((deg) => {
      const theta = degToRad(deg);
      const point = [Math.cos(theta), Math.sin(theta)];
      return apply2(transform, point);
    });

    const transformedVectors = principalVectors.map((vector) => apply2(transform, vector));
    const xs = circle.map((point) => point[0]).concat(transformedVectors.map((point) => point[0]), [0]);
    const ys = circle.map((point) => point[1]).concat(transformedVectors.map((point) => point[1]), [0]);
    const xScale = d3.scaleLinear().domain(extentWithPadding(xs)).range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain(extentWithPadding(ys)).range([innerHeight, 0]);

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
        g.append('line')
          .attr('x1', xScale(0))
          .attr('y1', yScale(0))
          .attr('x2', xScale(vector[0]))
          .attr('y2', yScale(vector[1]))
          .attr('stroke', index === 0 ? 'rgba(125, 255, 178, 0.92)' : 'rgba(255, 196, 86, 0.92)')
          .attr('stroke-width', 2.5);

        g.append('text')
          .attr('x', xScale(vector[0]) + 6)
          .attr('y', yScale(vector[1]) - 6)
          .attr('fill', index === 0 ? 'rgba(125, 255, 178, 0.92)' : 'rgba(255, 196, 86, 0.92)')
          .attr('font-size', 12)
          .text(index === 0 ? `λ₁=${lambda[0].toFixed(2)}` : `λ₂=${lambda[1].toFixed(2)}`);
      });
    }

    svg.append('text')
      .attr('x', margin.left)
      .attr('y', 14)
      .attr('fill', 'rgba(255,255,255,0.88)')
      .attr('font-size', 12)
      .attr('font-weight', 650)
      .text(title);
  }

  function render() {
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
    const transformedPoints = points.map((point) => apply2(currentTransform, point));
    const principalVectors = [
      [decomposition.V[0][0], decomposition.V[1][0]],
      [decomposition.V[0][1], decomposition.V[1][1]],
    ];
    const showVectors = !!showVectorsInput.checked;
    const showLabels = !!showLabelsInput.checked;
    const showRank1 = !!showRank1Input.checked || Number(rankInput.value) === 1;
    const overlayPoints = showRank1
      ? decomposition.rank1Reconstruction.map((point) => apply2(currentTransform, point))
      : null;

    let displayVectors;
    let basisLabel;
    if (step === 1) {
      displayVectors = [[1, 0], [0, 1]];
      basisLabel = 'pc';
    } else if (step === 2) {
      displayVectors = [[decomposition.lambda[0], 0], [0, decomposition.lambda[1]]];
      basisLabel = 'λ';
    } else if (step === 3) {
      displayVectors = principalVectors.map((vector, index) => [
        vector[0] * decomposition.lambda[index],
        vector[1] * decomposition.lambda[index],
      ]);
      basisLabel = 'v';
    } else {
      displayVectors = principalVectors;
      basisLabel = 'v';
    }

    drawScatterPlot({
      container: dataContainer,
      points: transformedPoints,
      principalVectors: displayVectors,
      showVectors,
      showLabels,
      overlayPoints,
      title: stepMeta.name,
      basisLabel,
    });

    drawOperatorPlot({
      container: operatorContainer,
      transform: currentTransform,
      principalVectors,
      lambda: decomposition.lambda,
      showVectors,
      title: 'How the covariance operator acts on directions',
    });

    stepSummaryEl.textContent = `${stepMeta.explanation} Singular values: ${decomposition.singularValues.map((value) => value.toFixed(3)).join(', ')}. Eigenvalues: ${decomposition.lambda.map((value) => value.toFixed(3)).join(', ')}.`;

    matrixXEl.textContent = formatMatrix(decomposition.X, 8);
    matrixFactorsEl.textContent = `Σ =\n${formatMatrix(decomposition.Sigma)}\n\nV =\n${formatMatrix(decomposition.V)}\n\nV^T =\n${formatMatrix(decomposition.VT)}`;
    matrixCovEl.textContent = `Λ =\n${formatMatrix(decomposition.Lambda)}\n\nC = X^T X / (n-1) =\n${formatMatrix(decomposition.covariance)}`;
    matrixCurrentEl.textContent = `${stepMeta.name}\n\n${formatMatrix(currentTransform)}${Number(rankInput.value) === 1 ? '\n\nRank-1 reconstruction uses only the first principal coordinate.' : ''}`;
  }

  const debouncedRender = debounce(render, 220);
  [
    presetInput,
    samplesInput,
    stepInput,
    rankInput,
    showVectorsInput,
    showLabelsInput,
    showRank1Input,
  ].forEach((element) => {
    element.addEventListener('input', debouncedRender);
    element.addEventListener('change', render);
  });

  randomizeBtn.addEventListener('click', () => {
    seed += 1;
    render();
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

  angleInput?.addEventListener('input', syncAngleLabel);
  angleInput?.addEventListener('input', render);
  angleInput?.addEventListener('change', syncAngleLabel);
  angleInput?.addEventListener('change', render);

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
  syncNumericLabel(noiseInput, noiseValueEl);
  render();
});