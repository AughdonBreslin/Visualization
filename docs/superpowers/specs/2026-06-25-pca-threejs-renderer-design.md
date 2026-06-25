# PCA Three.js Renderer Design

Date: 2026-06-25
Status: approved for implementation

## Problem

The PCA page's 3D plots are rendered by Plotly.
Plotly's chart-diffing abstraction costs ~87ms per `Plotly.react()` call, and a previous render blocks the main thread while the next slider `input` event waits.
This produces a 329ms input delay (measured at `input#pcaSpread1`) and makes the Shape tab sliders feel sluggish.

Three.js writes directly to WebGL and updates geometry buffers in-place, bringing per-frame render time from ~87ms to 2--10ms.

## Scope

Replace Plotly with Three.js for the 3D rendering path only.
The 2D fallback (`drawScatterPlot2D`, `drawOperatorPlot2D`) is D3 SVG and is already fast; it remains completely untouched.
Plotly is removed from the page entirely.

## Loading

An importmap is added to `pca.html` before all script tags:

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.min.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"
  }
}
</script>
```

The Plotly script tag is removed.
The `pca.js` script tag gains `type="module"`.
`pca-3d.js` is not loaded directly; `pca.js` imports from it.
D3 and numeric.js remain as globals loaded by regular script tags before the module.
The `DOMContentLoaded` wrapper in `pca.js` is removed -- ES modules are deferred automatically.

## File structure

`js/pca-3d.js` is a new ES module that owns all Three.js scene construction and geometry management.
`pca.js` is the orchestrator: PCA math, control wiring, render dispatch.
The 2D draw functions stay in `pca.js` unchanged.

## Module API

`pca-3d.js` exports two factory functions:

```js
export function createDataPlot3D(container) { ... }
export function createOperatorPlot3D(container) { ... }
```

Each returns a plot context:

```js
{
  update(state),            // called by render() with new data each frame
  destroy(),                // disposes renderer, geometries, ResizeObserver
  onCameraChange(fn),       // registers callback fired when user orbits
  applyCameraDir(phi, theta), // receives orientation sync from sibling plot
}
```

`pca.js` initialises both contexts once at startup and wires camera sync:

```js
const dataPlot    = createDataPlot3D(dataContainer);
const operatorPlot = createOperatorPlot3D(operatorContainer);

dataPlot.onCameraChange((phi, theta) => {
  if (shouldSync3DCameras()) operatorPlot.applyCameraDir(phi, theta);
});
operatorPlot.onCameraChange((phi, theta) => {
  if (shouldSync3DCameras()) dataPlot.applyCameraDir(phi, theta);
});
```

`drawScatterPlot3D` and `drawOperatorPlot3D` in `pca.js` are replaced by calls to `dataPlot.update(state)` and `operatorPlot.update(state)`.
The 2D branch is unchanged.

## Scene internals

Each plot context creates:
- One `WebGLRenderer` targeting a `<canvas>` appended to the container.
  `alpha: true`, `clearColor(0,0,0,0)`, `setPixelRatio(devicePixelRatio)`.
- One `Scene`.
- One `PerspectiveCamera` (fov 45, near 0.1, far 1000).
- One `OrbitControls` (target at origin, damping off).
- One `CSS2DRenderer` for HTML labels, overlaid on the canvas (`position: absolute`, `pointer-events: none`).

Rendering is on-demand.
There is no continuous RAF loop.
A render is triggered when `update()` is called or when OrbitControls fires `change`.

### Data plot geometry

| Object | Type | Update strategy |
|---|---|---|
| 3 axis lines | `Line` (2-point geometry) | `setLength` when `bound` changes |
| 3 axis labels | `CSS2DObject` | text content updated when `axisLabels` changes |
| Scatter points | `Points` + `BufferGeometry` | buffer overwrite when N fits current capacity; dispose + recreate only when N exceeds allocated capacity |
| Overlay points | `Points` + `BufferGeometry` | same as scatter; shown/hidden when overlay is present/absent |
| PC vectors | 3 `ArrowHelper` | `setDirection` + `setLength` each frame |
| Vector labels | 3 `CSS2DObject` | text updated when `basisLabels` changes; visibility follows `showVectors` |
| Point number labels | pool of `CSS2DObject` | pool grows to max-seen N; hidden instances have zero cost |

Axis lines scale to `bound * 1.15`.
Camera initial distance is set to `bound * 3` on the first render call.
User zoom is preserved across subsequent data updates.

### Operator plot geometry

| Object | Type | Update strategy |
|---|---|---|
| Latitude wireframe (5 circles) | `LineSegments` | position buffer overwrite on transform change |
| Longitude wireframe (6 circles) | `LineSegments` | position buffer overwrite on transform change |
| 3 transformed PC vectors | 3 `ArrowHelper` | `setDirection` + `setLength` each frame |
| Eigenvalue labels | 3 `CSS2DObject` | text updated each frame; visibility follows `showVectors` |
| 3 axis labels | 3 `CSS2DObject` | static text "x₁" / "x₂" / "x₃" |

Latitude circles: 5 lines × 31 points each (0--360° in 12° steps, positions transformed by covariance matrix).
Color: blue (#4aa3ff, opacity 0.58).
Longitude circles: 6 lines × 31 points each.
Color: white, opacity 0.10.

Line width greater than 1 is not supported in WebGL.
All lines render at 1px.
Visual distinction between latitude and longitude circles is maintained through color and opacity.

### Visual changes from Plotly

- PC vectors gain an arrowhead at the positive tip (Plotly drew bidirectional lines with no arrowhead). This is a deliberate improvement.
- Line width is always 1px (Plotly used width 3--7). Not visually significant at this scale.
- Hover tooltips showing `(x, y, z)` coordinates are dropped.
- Plotly's built-in axis tick marks and tick labels are replaced by the three axis lines and CSS2D axis labels. Numeric tick marks along axes are not reproduced.

## Camera sync

OrbitControls expresses camera position in spherical coordinates (φ, θ, r).
Sync shares orientation (φ, θ) while each plot preserves its own zoom distance r.

When the user drags plot A:
1. `OrbitControls` fires `change`.
2. Plot A reads `phi` and `theta` from `new THREE.Spherical().setFromVector3(camera.position)`.
3. Plot A calls its registered `onCameraChange(phi, theta)` callback.
4. `pca.js` calls `plotB.applyCameraDir(phi, theta)`.
5. Plot B reconstructs camera position: `camera.position.setFromSphericalCoords(r, phi, theta)` where `r = camera.position.length()`.
6. `controls.update()` and `render()` are called on plot B.

A `syncing` boolean on each context prevents echo: plot B does not fire its own `onCameraChange` while applying an external update.

The "Sync 3D camera angles" checkbox is evaluated in the `onCameraChange` callback in `pca.js`; if unchecked the callback returns early.

## Labels (CSS2DRenderer)

`CSS2DRenderer` appends a `position: absolute` div to the container, sized to match the canvas.
Both `renderer.render()` and `css2dRenderer.render()` are called on each frame.

All label elements receive class `pca-label`, styled in `pca.css`:
- Small mono font, muted color
- `white-space: nowrap`
- `pointer-events: none`

No new stylesheet file is needed; a few lines are added to the existing `pca.css`.

## Removal from pca.js

The following are removed entirely:

- `renderPlotly()`
- `cloneCamera()`, `getLiveCamera()`, `applyCameraOrientation()`, `mergeCameraOrientation()`
- `syncPlotlyCamera()`, `sync3DPlots()`
- `makeVectorTrace()`, `makeAxisTrace()`, `makeSceneAnnotation()`
- `buildSphereWireframe()`
- `drawScatterPlot3D()`, `drawOperatorPlot3D()`
- `container.dataset.plotlyInitialized` and `container.dataset.renderer` tracking
- `Plotly.purge()` call inside `clear()`

`clear()` is removed.
On a dimension switch (2D to 3D or back), `pca.js` calls `dataPlot.destroy()` and `operatorPlot.destroy()` then reinitialises the contexts, rather than relying on a generic clear helper.

## ResizeObserver

Each plot context attaches a `ResizeObserver` to the container.
On resize: `renderer.setSize(w, h)`, `css2dRenderer.setSize(w, h)`, `camera.aspect = w / h`, `camera.updateProjectionMatrix()`, then a render call.
`destroy()` disconnects the observer and calls `renderer.dispose()`.

## Error handling

If `createDataPlot3D` or `createOperatorPlot3D` throws (e.g. WebGL not available), `pca.js` catches the error and falls back to a visible error message in the container, consistent with the existing `numeric` check pattern.
