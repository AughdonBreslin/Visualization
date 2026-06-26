# Visualization

Interactive machine learning explainers built with vanilla JavaScript, D3, Three.js, and MathJax.
Each page pairs a live sandbox with step-by-step notes, pseudocode, and worked examples.

## Pages

**Distribution Visualizer** (`pages/distributions.html`)
Add and compare probability distributions side by side.
Adjust parameters in real time and see the PDF, CDF, and key statistics update instantly.

**Estimation** (`pages/estimation.html`)
Interactive demonstration of bias, variance, MSE, and MLE.
Sample repeatedly from a true distribution and watch the estimator's sampling distribution form.

**Bayesian Inference** (`pages/bayesian.html`)
Step through prior, likelihood, and posterior for common conjugate families.
Includes MAP estimation and an introduction to approximate inference.

**Linear Regression Regularization** (`pages/regularization.html`)
Visualize how L1 and L2 regularization shape the loss surface and shrink coefficients.
Drag data points to see how the fit responds.

**Principal Component Analysis** (`pages/pca.html`)
3D scatter plot with interactive camera, PC arrows, and a linked operator view showing the covariance ellipsoid.
Controls for dataset shape, noise, and dimensionality.

**Manifold Learning** (`pages/manifold.html`)
Step-by-step comparison of LLE and Isomap on a shared dataset.
Each step shows the algorithm state as a 3D visualization with intuition, formula, and pseudocode panels.

**Fourier Image Decomposition** (`pages/fourier.html`)
Upload an image and watch it reconstructed frequency by frequency.
Explains periodicity and the checkerboard artifact in the frequency domain.

**Unsupervised Supervised Learning** (`pages/generative_classification.html`)
Builds intuition for generative classifiers by showing how class-conditional densities produce decision boundaries.

**Glossary** (`pages/glossary.html`)
Reference definitions for terms used across the site.

## Running locally

The pages use ES modules and must be served over HTTP, not opened as `file://` URLs.

```bash
npx serve .
```

Then open `http://localhost:3000/pages/<page>.html`.

## Tech stack

- D3 v7 for 2D charts and SVG visualizations
- Three.js for 3D scenes on the PCA page
- MathJax 3 for typeset mathematics
- No build step; all pages are static HTML with ES module scripts
