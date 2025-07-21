# Kinone

> 木の根  
> Ki no ne  
> Roots of a tree

Kinone is a re-implementation of essential componenets
of modern gradient based learning,
written from first principles in pure NumPy.
By design it is:

- Deterministic: No hidden kernels or device dependent determinism.
- Transparent: No black boxes, every mathematical transformation is
explicit and auditable.
- Extensible: New operators require the forward definition and its analytic gradient.
(Could be changed if we introduce [JAX](https://docs.jax.dev/en/latest/quickstart.html))
- Reliable: Full test coverage with finite-difference checks for all primitives.

Kinone is in no way a competetor to JAX, PyTorch or TensorFlow,
it is a learning project and a scaffold for experiments
that require total algorithmic control and reportability.

## Mathematical Foundations

Let

$$
\mathcal{T} = (\mathbf{X},\,\text{grad}) ,\quad
\mathbf{X}\in\mathbb{R}^{n_1\times\cdots\times n_d}
$$

denote a Tensor coupled with a gradient accumulator.
Every primitive operator

$$
f: \mathbb{R}^{m}\rightarrow\mathbb{R}^{k},\qquad
\mathbf{y} = f(\mathbf{x})
$$

is registered with a $C^{1}$ map

$$
\partial f:\mathbb{R}^{m}\rightarrow\mathbb{R}^{k\times m}
$$

allowing Kinone’s reverse-mode automatic differentiation
to compute for a scalar loss $L$.

$$
\nabla_{\mathbf{x}}L=\bigl(\partial f(\mathbf{x})\bigr)^{\top}\nabla_{\mathbf{y}}L
$$

in $O\left(\sum_i \text{ops}_i\right)$ memory.
The design obeys the
[define-by-run paradigm](https://docs.chainer.org/en/stable/guides/define_by_run.html):
the dynamic graph $G=(V,E)$ is recorded during forward execution with backward
traversal unfolding in reverse [topological order](https://en.wikipedia.org/wiki/Topological_sorting).

---

##  Implemented Operator Set

| Category | Symbolic Definition |
| - | - |
| Elementwise| $y_i = \phi(x_i)\quad(\phi\in\{+,−,\times,\div,etc.})$ |
| Matrix Multiplication | $\mathbf{Y}=\mathbf{A}\mathbf{B}$ |
| Convolution 2-D | $Y_{c,j,k}=\sum_{p,q}X_{c,j+p,k+q}W_{c,p,q}$ |
| Pooling | $\max, \text{avg}$ |
| Norms | BatchNorm ($\mu$, $\sigma^2$ estimated) |

> Future plans include other missing operations such as
`ConvTranspose`, `1D-3D convolutions`, `attention` style operatations

---

##  Canonical Models

The modules in `src/core/` cover the following function classes:

$$
\mathcal{F}_{\text{CNN}} = \{ f\circ g\_L\circ\cdots\circ g\_1 \;\bigl|\;
g\_i\in\{\text{Conv},\text{BN},\sigma,\text{Pool}\},\; f\in\text{Linear} \},
$$

which subsumes a sample ResNet - $d$ with
$d\in\{18,34,50,101,152\}$ implementation
for multi-label classification of NIH ChestX-ray dataset.

---

## Project Layout

```
Kinone
├── src/
│   ├── core/          # tensor, autograd, nn layers
│   ├── data/          # NIH ChestX-ray14 loader
│   └── scripts/       # dataset download, ONNX export
├── train.py           # SGD/Adam loop
├── evaluate.py        # metrics + ROC-AUC
├── dashboard.py       # Streamlit inference UI
├── tests/             # finite-difference checks
└── README.md          # you are here
```

---

## Installation

The recommended way to work with this repository is to install and use [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/pseudofractal/kinone.git
cd kinone
uv sync # Or equivalent if you are not using uv
```

Dependencies (`pyproject.toml` pinned):

- `numpy`
- `opencv-python`
- `albumentations`
- `onnxruntime`
- `streamlit`
- `pytest`

---

## 7 Quick Start

```bash
# 1. Fetch NIH CXR dataset (≈ 42 GB)
python -m src.scripts.download_nih_dataset --out-dir data/nih

# 2. Train ResNet-18 baseline
python train.py --dataset nih --epochs 30 --batch-size 32 --lr 1e-3

# 3. Evaluate and compute ROC-AUC
python evaluate.py --dataset nih --weights path/to/weights.npz

# 4. ONNX export (ops-set 13)
python -m src.scripts.export_onnx --weights path/to/weights.npz --out resnet18.onnx

# 5. Launch live inference demo
streamlit run dashboard.py
```

---

## Testing Process

Each primitive $f$ satisfies

$$
\frac{\| \hat{\nabla}f - \nabla f \|_2}{\| \nabla f \|_2 + \epsilon} < 10^{-4},
$$

where $\hat{\nabla}f$ is the central-difference estimate with step $h=10^{-3}$.
Run all checks:

```bash
pytest -q
```

---
