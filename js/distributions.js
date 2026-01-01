document.addEventListener('DOMContentLoaded', () => {
    // Guard: only run on the distributions visualizer page
    if (!document.querySelector('#pdf') || !document.querySelector('#forms')) return;
    const colors = d3.schemeTableau10.concat(d3.schemeSet3).flat();
    const distributionInfo = {
        normal: { title: 'Normal', params: ['Mean (μ)', 'Std (σ)'], defaults: [0, 1] },
        uniform: { title: 'Uniform', params: ['Min (a)', 'Max (b)'], defaults: [0, 1] },
        exponential: { title: 'Exponential', params: ['Rate (λ)'], defaults: [1] },
        laplace: { title: 'Laplace', params: ['Mean (μ)', 'Scale (b)'], defaults: [0, 1] },
        dirac: { title: 'Dirac', params: ['Location (μ)'], defaults: [0] },
        mixture: {
            title: 'Mixture',
            params: ['Components'],
            defaults: [[]],
            isSpecial: true  // Flag to handle this distribution type differently
        }
    };

    let distributions = JSON.parse(localStorage.getItem('distributions') || '[]');
    let nextId = distributions.length ? Math.max(...distributions.map(d => d.id)) + 1 : 1;

    const formsContainer = d3.select('#forms');
    const xMinInput = d3.select('#xMin');
    const xMaxInput = d3.select('#xMax');
    const addBtn = d3.select('#addDist');
    const resetBtn = d3.select('#resetAll');

    function saveState() {
        localStorage.setItem('distributions', JSON.stringify(distributions));
        localStorage.setItem('xRange', JSON.stringify({ xMin: xMinInput.property('value'), xMax: xMaxInput.property('value') }));
    }

    function loadRange() {
        const range = JSON.parse(localStorage.getItem('xRange') || '{}');
        if (range.xMin) xMinInput.property('value', range.xMin);
        if (range.xMax) xMaxInput.property('value', range.xMax);
    }

    const margin = { top: 18, right: 18, bottom: 36, left: 54 };
    let pdfSvg, cdfSvg, pdfInner, cdfInner, pdfX, pdfY, cdfX, cdfY, width, height;

    function createSvg(container) {
        const rect = container.node().getBoundingClientRect();
        const w = Math.max(300, rect.width);
        const h = 340;
        const svg = container.append('svg').attr('width', '100%').attr('height', h);
        const innerW = w - margin.left - margin.right;
        const innerH = h - margin.top - margin.bottom;
        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        return { svg, g, innerW, innerH };
    }

    function setupCharts() {
        d3.select('#pdf').selectAll('*').remove();
        d3.select('#cdf').selectAll('*').remove();
        const pdfBox = createSvg(d3.select('#pdf'));
        const cdfBox = createSvg(d3.select('#cdf'));
        pdfSvg = pdfBox.svg; pdfInner = pdfBox.g; width = pdfBox.innerW; height = pdfBox.innerH;
        cdfSvg = cdfBox.svg; cdfInner = cdfBox.g;
        pdfX = d3.scaleLinear().range([0, width]);
        pdfY = d3.scaleLinear().range([height, 0]);
        cdfX = d3.scaleLinear().range([0, width]);
        cdfY = d3.scaleLinear().domain([0, 1]).range([height, 0]);
        pdfInner.append('g').attr('class', 'x axis').attr('transform', `translate(0,${height})`);
        pdfInner.append('g').attr('class', 'y axis');
        pdfInner.append('g').attr('class', 'pdf-lines');
        cdfInner.append('g').attr('class', 'x axis').attr('transform', `translate(0,${height})`);
        cdfInner.append('g').attr('class', 'y axis');
        cdfInner.append('g').attr('class', 'cdf-lines');
    }

    function pdf(x, k, p) {
        const [a = 0, b = 1] = p;
        switch (k) {
            case 'normal': return (1 / (b * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - a) / b) ** 2);
            case 'uniform': return (x >= a && x <= b) ? 1 / (b - a) : 0;
            case 'exponential': return x >= 0 ? a * Math.exp(-a * x) : 0;
            case 'laplace': return (1 / (2 * b)) * Math.exp(-Math.abs(x - a) / b);
            case 'dirac': const s = 0.03; return (1 / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - a) / s) ** 2);
            case 'mixture':
                if (!p.components || !p.weights) return 0;
                return p.components.reduce((sum, comp, i) =>
                    sum + (p.weights[i] * pdf(x, comp.distKey, comp.params)), 0);
            default: return 0;
        }
    }

    function cdf(x, k, p) {
        const [a = 0, b = 1] = p;
        switch (k) {
            case 'normal': return 0.5 * (1 + erf((x - a) / (b * Math.sqrt(2))));
            case 'uniform': if (x < a) return 0; if (x > b) return 1; return (x - a) / (b - a);
            case 'exponential': return x >= 0 ? 1 - Math.exp(-a * x) : 0;
            case 'laplace': return x < a ? 0.5 * Math.exp((x - a) / b) : 1 - 0.5 * Math.exp(-(x - a) / b);
            case 'dirac': return x < a ? 0 : 1;
            case 'mixture':
                if (!p.components || !p.weights) return 0;
                return p.components.reduce((sum, comp, i) =>
                    sum + (p.weights[i] * cdf(x, comp.distKey, comp.params)), 0);
            default: return 0;
        }
    }
    function erf(x) {
        const a1 = .254829592, a2 = -.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = .3275911;
        const sign = x >= 0 ? 1 : -1; x = Math.abs(x);
        const t = 1 / (1 + p * x);
        const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
    }
    function sharedXs(a, b, n = 300) {
        return Array.from({ length: n + 1 }, (_, i) => a + (b - a) * i / n);
    }

    function redrawAll() {
        if (!pdfInner) setupCharts();
        const xMin = parseFloat(xMinInput.property('value')), xMax = parseFloat(xMaxInput.property('value'));
        const xs = sharedXs(xMin, xMax, 400);
        let maxPdf = 0;
        const series = distributions.map((d, i) => {
            const color = d.color || colors[i % colors.length];
            let pdfs, cdfs;

            if (d.distKey === 'mixture') {
                // For mixture distributions, use the components and weights stored in params
                pdfs = xs.map(x => {
                    if (!d.params.components || !d.params.weights) return 0;
                    return d.params.components.reduce((sum, comp, j) =>
                        sum + (d.params.weights[j] * pdf(x, comp.distKey, comp.params)), 0);
                });

                cdfs = xs.map(x => {
                    if (!d.params.components || !d.params.weights) return 0;
                    return d.params.components.reduce((sum, comp, j) =>
                        sum + (d.params.weights[j] * cdf(x, comp.distKey, comp.params)), 0);
                });
            } else {
                // For regular distributions
                pdfs = xs.map(x => pdf(x, d.distKey, d.params));
                cdfs = xs.map(x => cdf(x, d.distKey, d.params));
            }

            maxPdf = Math.max(maxPdf, d3.max(pdfs));
            return { id: d.id, key: d.distKey, color, pdfs, cdfs };
        });
        pdfX.domain([xMin, xMax]); cdfX.domain([xMin, xMax]); pdfY.domain([0, maxPdf === 0 ? 1 : maxPdf * 1.1]);
        pdfInner.select(".x.axis").call(d3.axisBottom(pdfX).ticks(8).tickSizeOuter(0));
        pdfInner.select(".y.axis").call(d3.axisLeft(pdfY).ticks(6).tickSizeOuter(0));
        cdfInner.select(".x.axis").call(d3.axisBottom(cdfX).ticks(8).tickSizeOuter(0));
        cdfInner.select(".y.axis").call(d3.axisLeft(cdfY).ticks(6).tickSizeOuter(0));
        const pdfLine = d3.line().x((d, i) => pdfX(xs[i])).y(d => pdfY(d)).curve(d3.curveMonotoneX);
        const cdfLine = d3.line().x((d, i) => cdfX(xs[i])).y(d => cdfY(d)).curve(d3.curveMonotoneX);
        
        // Update PDF plot
        const pdfG = pdfInner.select('.pdf-lines');
        const pdfSel = pdfG.selectAll('.series-pdf').data(series, d => d.id);

        // Handle entering PDF paths
        const pdfEnter = pdfSel.enter().append('g')
            .attr('class', 'series-pdf');
        pdfEnter.append('path')
            .attr('class', 'line')
            .attr('fill', 'none')
            .attr('stroke-width', 2);

        // Remove old PDF paths
        pdfSel.exit().remove();

        // Update all PDF paths (both new and existing)
        pdfG.selectAll('.series-pdf path.line')
            .data(series, d => d.id)
            .attr('stroke', d => d.color)
            .attr('d', d => pdfLine(d.pdfs));

        // Update CDF plot
        const cdfG = cdfInner.select('.cdf-lines');
        const cdfSel = cdfG.selectAll('.series-cdf').data(series, d => d.id);

        // Handle entering CDF paths
        const cdfEnter = cdfSel.enter().append('g')
            .attr('class', 'series-cdf');
        cdfEnter.append('path')
            .attr('class', 'line')
            .attr('fill', 'none')
            .attr('stroke-width', 2);

        // Remove old CDF paths
        cdfSel.exit().remove();

        // Update all CDF paths (both new and existing)
        cdfG.selectAll('.series-cdf path.line')
            .data(series, d => d.id)
            .attr('stroke', d => d.color)
            .attr('d', d => cdfLine(d.cdfs));
        saveState();
        updateFormSwatches();
    }

    function addDistributionForm(pref = { dist: 'normal', defaults: [] }) {
        const id = nextId++;
        const wrapper = formsContainer.append('div')
            .attr('class', 'form-card')
            .attr('data-id', id);
        const selWrap = wrapper.append('div')
            .attr('class', 'form-col');
        selWrap.append('label')
            .attr('for', 'distributionSelect')
            .text('Distribution');
        const sel = selWrap.append('select');
        sel.attr('name', 'distributionSelect');
        Object.keys(distributionInfo).forEach(
            k => sel.append('option')
                .attr('value', k)
                .text(distributionInfo[k].title)
        );
        sel.property('value', pref.dist || 'normal');

        // create distribution object with default/custom color and push to state
        const defaultColor = pref.color || colors[distributions.length % colors.length];
        const formObj = { id, distKey: sel.property('value'), params: [], color: defaultColor };
        distributions.push(formObj);

        // insert a color input as the left-most child so user can pick any color
        const colorInput = wrapper.insert('input', ':first-child')
            .attr('type', 'color')
            .attr('class', 'form-swatch')
            .property('value', formObj.color);

        // sync color input -> model
        colorInput.on('input', function () {
            formObj.color = this.value;
            redrawAll();
            saveState();
        });

        const paramsWrap = wrapper.append('div')
            .attr('class', 'form-col params');
        const actions = wrapper.append('div')
            .attr('class', 'form-col actions');
        actions.append('label')
            .attr('for', 'removeButton')
            .text('');
        const removeBtn = actions.append('button')
            .attr('name', 'removeButton')
            .text('Remove');

        function buildParams() {
            paramsWrap.selectAll('*').remove();
            const info = distributionInfo[formObj.distKey];

            if (formObj.distKey === 'mixture') {
                // Initialize mixture parameters if not set
                if (!formObj.params.weights) formObj.params.weights = [];
                if (!formObj.params.components) formObj.params.components = [];

                // Clear existing mixture components display
                wrapper.selectAll('.mixture-components').remove();

                const buttonDiv = paramsWrap.append('div')
                    .attr('class', 'param-item');
                buttonDiv.append('label')
                    .attr('for', 'add-component-button')
                    .text('');

                const addButton = buttonDiv.append('button')
                    .attr('name', 'add-component-button')
                    .attr('class', 'add-component-button')
                    .text('Add Component')
                    .on('click', () => {
                        // Default to the first non-mixture distribution type
                        const selectedDist = Object.keys(distributionInfo).find(k => k !== 'mixture') || 'normal';
                        const distInfo = distributionInfo[selectedDist];
                        const newComponent = {
                            distKey: selectedDist,
                            params: [...distInfo.defaults]
                        };

                        // Add component with equal weight
                        formObj.params.components.push(newComponent);
                        const numComponents = formObj.params.components.length;
                        formObj.params.weights = formObj.params.components.map(() => 1 / numComponents);

                        buildParams(); // Rebuild the form
                        redrawAll();
                        saveState();
                    });

                // Show existing components
                if (formObj.params.components.length > 0) {

                    const componentsDiv = wrapper.append('div')
                        .attr('class', 'mixture-components');

                    formObj.params.components.forEach((comp, idx) => {
                        const componentInfo = distributionInfo[comp.distKey];
                        const componentDiv = componentsDiv.append('div')
                            .attr('class', 'mixture-component');
                        // Component header with type selector and weight/remove controls
                        const headerDiv = componentDiv.append('div')
                            .attr('class', 'mixture-component-header');

                        // Add a select inside each component so the user can change its type
                        const compSelect = headerDiv.append('select')
                            .attr('class', 'component-type-select');

                        Object.keys(distributionInfo).forEach(k => {
                            if (k !== 'mixture') compSelect.append('option').attr('value', k).text(distributionInfo[k].title);
                        });
                        compSelect.property('value', comp.distKey);
                        compSelect.on('change', function () {
                            const newKey = this.value;
                            comp.distKey = newKey;
                            // reset params to defaults for the newly selected type
                            comp.params = [...(distributionInfo[newKey].defaults || [])];
                            buildParams();
                            redrawAll();
                            saveState();
                        });



                        // Build body: left column for weight + remove button, right column for params
                        const body = componentDiv.append('div')
                            .attr('class', 'mixture-component-body');

                        const leftCol = body.append('div')
                            .attr('class', 'mixture-component-left');

                        const rightCol = body.append('div')
                            .attr('class', 'mixture-component-right');

                        // Move weight input into left column (stacked)
                        const weightWrap = leftCol.append('div')
                            .attr('class', 'mixture-input');

                        weightWrap.append('label')
                            .attr('for', 'weightInput')
                            .text('Weight:');

                        weightWrap.append('input')
                            .attr('name', 'weightInput')
                            .attr('type', 'number')
                            .attr('min', '0')
                            .attr('max', '1')
                            .attr('step', '0.1')
                            .property('value', formObj.params.weights[idx])
                            .on('input', function () {
                                formObj.params.weights[idx] = +this.value;
                                // Normalize weights
                                const sum = formObj.params.weights.reduce((a, b) => a + b, 0);
                                formObj.params.weights = formObj.params.weights.map(w => w / sum);
                                buildParams();
                                redrawAll();
                                saveState();
                            });

                        // Remove button below weight in left column
                        leftCol.append('button')
                            .attr('class', 'mixture-remove-btn')
                            .text('Remove')
                            .on('click', () => {
                                formObj.params.components.splice(idx, 1);
                                formObj.params.weights.splice(idx, 1);
                                if (formObj.params.weights.length > 0) {
                                    // Renormalize remaining weights
                                    const sum = formObj.params.weights.reduce((a, b) => a + b, 0);
                                    formObj.params.weights = formObj.params.weights.map(w => w / sum);
                                }
                                buildParams();
                                redrawAll();
                                saveState();
                            });

                        componentInfo.params.forEach((paramName, paramIdx) => {
                            const paramDiv = rightCol.append('div').attr('class', 'mixture-input');
                            paramDiv.append('label')
                                .attr('for', 'paramInput')
                                .text(paramName);

                            paramDiv.append('input')
                                .attr('name', 'paramInput')
                                .attr('type', 'number')
                                .attr('step', '0.1')
                                .property('value', comp.params[paramIdx])
                                .on('input', function () {
                                    comp.params[paramIdx] = +this.value;
                                    redrawAll();
                                    saveState();
                                });
                        });
                    });
                } else {
                    // Show message when no components exist
                    paramsWrap.append('div')
                        .attr('class', 'mixture-empty-message')
                        .text('Add components to create a mixture distribution');
                }
            } else {
                // For regular distributions
                formObj.params = [];
                info.params.forEach((label, idx) => {
                    const f = paramsWrap.append('div').attr('class', 'param-item');
                    f.append('label').attr('for', 'input').text(label);
                    const inp = f.append('input')
                        .attr('name', 'input')
                        .attr('type', 'number')
                        .attr('step', label.toLowerCase().includes('prob') ? '0.01' : '0.1')
                        .property('value', (pref.defaults && pref.defaults[idx] !== undefined) ? pref.defaults[idx] : info.defaults[idx]);
                    formObj.params[idx] = +inp.property('value');
                    inp.on('input', () => { formObj.params[idx] = +inp.property('value'); redrawAll(); saveState(); });
                });
            }
            redrawAll(); saveState();
        }

        sel.on('change', () => { formObj.distKey = sel.property('value'); buildParams(); });
        removeBtn.on('click', () => { distributions = distributions.filter(d => d.id !== id); wrapper.remove(); redrawAll(); saveState(); });
        buildParams();
    }

    function updateFormSwatches() {
        // Update the small color swatch on each form based on the distribution's position
        const cards = document.querySelectorAll('.form-card');
        cards.forEach(card => {
            const id = +card.getAttribute('data-id');
            const idx = distributions.findIndex(d => d.id === id);
            const swatch = card.querySelector('.form-swatch');
            if (swatch) {
                const distObj = distributions[idx];
                const color = distObj ? distObj.color : (idx >= 0 ? colors[idx % colors.length] : 'transparent');
                if (swatch.tagName === 'INPUT') {
                    swatch.value = color;
                    swatch.style.background = color;
                } else swatch.style.background = color;
            }
        });
    }

    function clearAll() {
        distributions = []; nextId = 1;
        formsContainer.selectAll('*').remove();
        redrawAll(); saveState();
    }

    // Mixture functionality
    const mixtureSection = d3.select('#mixtureSection');
    const mixtureWeights = d3.select('#mixtureWeights');
    const createMixtureBtn = d3.select('#createMixture');
    const applyMixtureBtn = d3.select('#applyMixture');
    const cancelMixtureBtn = d3.select('#cancelMixture');

    function showMixtureUI() {
        if (distributions.length < 2) {
            alert('Add at least 2 distributions to create a mixture');
            return;
        }

        // Show mixture UI by toggling a CSS class (styling lives in styles.css)
        mixtureSection.classed('visible', true);
        mixtureWeights.selectAll('*').remove();

        distributions.forEach((dist, i) => {
            const weightDiv = mixtureWeights.append('div')
                .attr('class', 'weight-input')
                // set CSS variable for the swatch color; actual border set in CSS
                .style('--swatch-color', dist.color);

            weightDiv.append('span')
                .text(`${distributionInfo[dist.distKey].title} ${i + 1}`);

            weightDiv.append('input')
                .attr('type', 'number')
                .attr('min', '0')
                .attr('max', '1')
                .attr('step', '0.1')
                .attr('value', (1 / distributions.length).toFixed(2))
                .attr('data-idx', i);
        });
    }

    function hideMixtureUI() {
        // Hide mixture UI by toggling the CSS class
        mixtureSection.classed('visible', false);
    }

    function createMixture() {
        const weights = Array.from(document.querySelectorAll('.weight-input input'))
            .map(input => parseFloat(input.value));

        // Normalize weights
        const sum = weights.reduce((a, b) => a + b, 0);
        const normalizedWeights = weights.map(w => w / sum);

        // Create mixture distribution
        const mixtureComponents = distributions.map((dist, i) => ({
            distKey: dist.distKey,
            params: dist.params
        }));

        const mixtureColor = d3.interpolateRgb(distributions[0].color, distributions[1].color)(0.5);

        const mixtureObj = {
            id: nextId++,
            distKey: 'mixture',
            params: {
                components: mixtureComponents,
                weights: normalizedWeights
            },
            color: mixtureColor
        };

        distributions.push(mixtureObj);
        addDistributionForm({
            dist: 'mixture',
            defaults: [],
            color: mixtureColor
        });

        hideMixtureUI();
    }

    // Event listeners
    addBtn.on('click', () => addDistributionForm());
    resetBtn.on('click', clearAll);
    createMixtureBtn.on('click', showMixtureUI);
    applyMixtureBtn.on('click', createMixture);
    cancelMixtureBtn.on('click', hideMixtureUI);
    xMinInput.on('input', () => { redrawAll(); saveState(); });
    xMaxInput.on('input', () => { redrawAll(); saveState(); });

    setupCharts(); loadRange();

    // Clear existing forms on load
    formsContainer.selectAll('*').remove();
    const saved = JSON.parse(localStorage.getItem('distributions') || '[]');
    distributions = [];
    nextId = 1;

    if (saved.length) {
        saved.forEach(d => addDistributionForm({
            dist: d.distKey,
            defaults: d.params,
            color: d.color
        }));
    } else addDistributionForm();

    window.addEventListener('resize', () => { setupCharts(); redrawAll(); });

    // Distribution information
    const distributionDetails = {
        normal: {
            formula: "f(x|μ,σ) = \\frac{1}{σ\\sqrt{2π}} e^{-\\frac{(x-μ)^2}{2σ^2}}",
            expectedValue: "E[X] = μ",
            variance: "Var(X) = σ^2",
            useCases: [
                "Weight initialization in neural networks (Xavier/Glorot, He initialization)",
                "Modeling noise in VAEs and diffusion models",
                "Output distribution in regression tasks",
                "Sampling latent vectors in generative models"
            ]
        },
        uniform: {
            formula: "f(x|a,b) = \\frac{1}{b-a} \\text{ for } x \\in [a,b]",
            expectedValue: "E[X] = \\frac{a+b}{2}",
            variance: "Var(X) = \\frac{(b-a)^2}{12}",
            useCases: [
                "Random weight initialization",
                "Dropout mask generation",
                "Noise generation in adversarial training",
                "Exploration in reinforcement learning"
            ]
        },
        exponential: {
            formula: "f(x|λ) = λe^{-λx} \\text{ for } x ≥ 0",
            expectedValue: "E[X] = \\frac{1}{λ}",
            variance: "Var(X) = \\frac{1}{λ^2}",
            useCases: [
                "Modeling waiting times in sequence models",
                "Attention mechanism decay rates",
                "Time-to-event prediction in survival analysis",
                "Rate parameters in Poisson processes"
            ]
        },
        laplace: {
            formula: "f(x|μ,b) = \\frac{1}{2b}e^{-\\frac{|x-μ|}{b}}",
            expectedValue: "E[X] = μ",
            variance: "Var(X) = 2b^2",
            useCases: [
                "Robust alternatives to normal distribution in neural networks",
                "Modeling sparse gradients in optimization",
                "Noise distribution in adversarial training",
                "Parameter regularization in Bayesian deep learning"
            ]
        },
        dirac: {
            formula: "δ(x-μ) = \\begin{cases} ∞ & \\text{if } x = μ \\\\ 0 & \\text{otherwise} \\end{cases}",
            expectedValue: "E[X] = μ",
            variance: "Var(X) = 0",
            useCases: [
                "Deterministic latent representations",
                "One-hot encoded targets",
                "Perfect prediction scenarios in theoretical analysis",
                "Modeling exact values in mixture models"
            ]
        },
        mixture: {
            formula: "f(x) = \\sum_{i=1}^n w_i f_i(x)",
            expectedValue: "E[X] = \\sum_{i=1}^n w_i E[X_i]",
            variance: "Var(X) = \\sum_{i=1}^n w_i(Var(X_i) + (E[X_i] - E[X])^2)",
            useCases: [
                "Modeling multi-modal data distributions",
                "Combining multiple component distributions",
                "Creating complex probability densities",
                "Representing heterogeneous data sources"
            ]
        },
    };

    function updateDistributionInfo() {
        const detailsContainer = document.getElementById('distributions-details');
        detailsContainer.innerHTML = '';

        if (distributions.length === 0) {
            detailsContainer.innerHTML = '<p class="no-distributions">Add distributions above to see their details here.</p>';
            return;
        }

        // Only show one description per unique distribution type (preserve order of first occurrence)
        const uniqueKeys = Array.from(new Set(distributions.map(d => d.distKey)));
        uniqueKeys.forEach((key, idx) => {
            const info = distributionDetails[key];
            // Use color of the first occurrence of this distribution, if available
            const firstIndex = distributions.findIndex(d => d.distKey === key);
            const color = (firstIndex >= 0 && distributions[firstIndex] && distributions[firstIndex].color)
                ? distributions[firstIndex].color
                : colors[firstIndex >= 0 ? firstIndex % colors.length : idx % colors.length];

            const detailElement = document.createElement('div');
            detailElement.className = 'distribution-detail';

            // Create content with MathJax formatting
            detailElement.innerHTML = `
            <h3>
                <span class="color-indicator" style="background: ${color}"></span>
                ${distributionInfo[key].title}
            </h3>
            <div class="formula">\\[${info.formula}\\]</div>
            <div class="properties">
                <div class="property">
                <strong>Expected Value:</strong><br>
                \\(${info.expectedValue}\\)
                </div>
                <div class="property">
                <strong>Variance:</strong><br>
                \\(${info.variance}\\)
                </div>
            </div>
            <div class="use-cases">
                <strong>Common Use Cases in Deep Learning:</strong>
                ${info.useCases.map(use => `<div class="use-case">${use}</div>`).join('')}
            </div>
            `;

            detailsContainer.appendChild(detailElement);

            // Trigger MathJax to render the new content
            if (window.MathJax) {
                MathJax.typesetPromise([detailElement]);
            }
        });
    }

    // Add MathJax script for formula rendering
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    script.async = true;
    document.head.appendChild(script);

    // Update distribution info when distributions change
    const originalRedrawAll = redrawAll;
    redrawAll = function () {
        originalRedrawAll();
        updateDistributionInfo();
    };
});