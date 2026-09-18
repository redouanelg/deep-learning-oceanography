# Overfitting, and the split that hides it

A worked example of overfitting on a real ocean problem, and of the mistake that
makes overfitting invisible in gridded geophysical data: **splitting cells at
random**.

The task is step 2 of THOR — predict which of six barotropic vorticity regimes
an ocean grid cell belongs to, from local physical properties.

## Run it

```sh
pip install -r requirements.txt
jupyter notebook overfitting_ocean_regimes.ipynb
```

That is the whole setup. The data ships with the notebook as a 1.8 MB `.npz`,
so there is nothing to download and it runs offline. About four minutes on CPU.

## What it shows

**1. Accuracy is an insensitive overfitting detector.** An oversized network
trained on 3000 labelled cells sends its held-out cross-entropy from 0.49 to
1.23 while held-out *accuracy* slips only four points, 0.808 to 0.767. Four
points are easy to dismiss as noise; a loss that more than doubles is not. And
what the loss is catching is invisible to accuracy by construction: when the
model is wrong it becomes far more confident about it, and the probability it
leaves on the true regime falls from 0.21 to 0.09. Monitor a proper scoring
rule.

**2. A random split measures the wrong thing.** Same model, same data, and the
generalisation gap depends only on how the test cells were chosen:

| split | train | held-out | gap | test cells adjacent to a training cell |
| --- | --- | --- | --- | --- |
| random | 0.911 | 0.911 | −0.000 | **99.9 %** |
| hold out the Atlantic | 0.925 | 0.821 | +0.105 | 0.7 % |

With a random split nearly every "held-out" cell is directly next to a cell the
model trained on, at a grid spacing where neighbours are nearly the same water.
The model is interpolating, not generalising, and the gap disappears entirely —
it even comes out a hair negative. Hold out a basin instead and a gap of
`+0.105` appears, from the same model and the same data.

This also punishes you for improving the model. Adding THOR's three SSH inputs
to the other five lifts the random-split score from 0.890 to 0.911, which looks
like progress, while the Atlantic score goes from 0.825 to 0.821 — very slightly
down. Better features bought interpolation skill and no generalisation at all,
and only the geographic split can tell you that.

**3. Not every gap is overfitting.** Early stopping against a *third*,
spatially separate region does the real work: it lifts Atlantic accuracy from
0.769 to 0.816 and roughly halves the gap. Dropout and weight decay then make
things slightly *worse*, by 2.2 and 1.7 points, because they let training run
longer before the validation loss turns and the extra epochs fit the training
basin harder. What remains is distribution shift between basins, and no
regulariser removes it. The two failures look identical in a single number and
need completely different responses.

## Layout

| file | |
| --- | --- |
| `overfitting_ocean_regimes.ipynb` | the notebook, outputs included so it reads on GitHub |
| `data/thor_step2_fields.npz` | the four input fields, float32, 1.8 MB |
| `make_dataset.py` | rebuilds that `.npz` from the original sources; you do not need to run it |
| `requirements.txt` | what the notebook needs |
| `requirements-dataset.txt` | the above plus what `make_dataset.py` needs |

## Data

`data/thor_step2_fields.npz` holds four fields on a 0.5° global grid
(360 × 720), stored as float32:

| array | field | source |
| --- | --- | --- |
| `curl` | wind stress curl | DNN4Cli |
| `bathy` | bathymetry | DNN4Cli |
| `label` | the six regime labels, `-1` over land | DNN4Cli |
| `ssh` | 20-year mean sea-surface height | ECCO v4r3, averaged from 240 monthly fields |

The notebook derives the SSH and bathymetry gradients and the Coriolis
parameter from these, giving **THOR's eight inputs**: wind stress curl, SSH,
∂SSH/∂x, ∂SSH/∂y, bathymetry, ∂H/∂x, ∂H/∂y and *f*.

The file is committed rather than downloaded because the alternative is 580 MB
of yearly ECCO netCDFs from a Google Drive folder — slow, and a link that has
to keep working for as long as this example does. Pre-averaging turns that into
1.8 MB, which is small enough to version. float32 costs about 6 × 10⁻⁸ m on
SSH, and the notebook casts to float32 before training anyway.

`make_dataset.py` rebuilds the file from scratch, so the provenance is
auditable (`pip install -r requirements-dataset.txt` first): it fetches the three DNN4Cli arrays over HTTPS and verifies their
SHA-256, then pulls the twenty yearly SSH files one at a time, folding each
into a running mean and deleting it so peak disk stays near 30 MB.

### Two details that matter

**The SSH mean has to be NaN-aware.** THOR computes it as
`monthlySSH['SSH'].mean(axis=0)`, and xarray skips NaN, so a cell that is NaN
in all 240 months stays NaN. Summing with `np.nansum` and dividing by 240
instead gives land a sea surface of exactly 0 m, which puts a large false
gradient in every ocean cell next to a coast — and those cells survive the
mask, because THOR builds the mask from the gradients:

```python
missingdataindex = np.isnan(curlTau*SSH20mean*gradSSH_x*gradSSH_y*Bathm*gradBathm_x*gradBathm_y)
```

`make_dataset.py` divides by the per-cell count of finite months, which
reproduces xarray's result exactly — same NaN mask, maximum absolute difference
0.0 on the finite cells.

**The grid metric differs from THOR's in one row.** THOR derives the zonal
spacing with `np.roll(lat, shift=-1)` rather than the grid step. That wraps at
the north pole, so the single row at 89.75° N gets −179.5° instead of 0.5°: a
spacing 359 times too large, with the wrong sign, which suppresses and flips
∂/∂x there. We use the grid step directly. It affects 720 of ~149,600 cells,
all in that row.

Latitude itself is not stored, because ECCO's latitude field turns out to be
exactly `linspace(-89.75, 89.75, 360)` broadcast across longitude — identical
to the last bit — so the notebook reconstructs it. Two checks in the notebook
confirm the orientation: row 0 is Antarctic and mostly land, and the Coriolis
parameter changes sign at the equator.

## Known install snag

On conda/macOS, importing `torch` alongside a conda-installed `scipy` or
`scikit-learn` can abort with `OMP: Error #15` — two copies of the OpenMP
runtime. The fix is a consistent set of packages, easiest in a fresh
environment with `pip install -r requirements.txt`. Setting
`KMP_DUPLICATE_LIB_OK=TRUE` silences it but is explicitly unsupported and can
give wrong numerical results, so prefer the clean environment.

## Credit

The problem, the data and the model architecture come from THOR:

> Sonnewald, M. and Lguensat, R. (2021). Revealing the impact of global heating
> on North Atlantic circulation using transparent machine learning.
> *Journal of Advances in Modeling Earth Systems* **13**, e2021MS002496.
> [doi:10.1029/2021MS002496](https://doi.org/10.1029/2021MS002496)

Sea-surface height is from the ECCO Consortium's v4 release 3 state estimate
(JPL/UT/MIT/AER), averaged here over 1992–2011:

> ECCO Consortium, Fukumori, I., Wang, O., Fenty, I., Forget, G., Heimbach, P.
> and Ponte, R. M. *ECCO Central Estimate (Version 4 Release 3)*.
> [ecco-group.org](https://ecco-group.org/)

The regime labels come from the unsupervised step that precedes THOR:

> Sonnewald, M., Wunsch, C. and Heimbach, P. (2019). Unsupervised learning
> reveals geography of global ocean dynamical regions.
> *Earth and Space Science* **6**, 784–794.
> [doi:10.1029/2018EA000519](https://doi.org/10.1029/2018EA000519)

On spatial cross-validation generally:

> Roberts, D. R. et al. (2017). Cross-validation strategies for data with
> temporal, spatial, hierarchical, or phylogenetic structure.
> *Ecography* **40**, 913–929.
> [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)

This example reuses THOR's problem setup, inputs and network architecture for
teaching purposes. It is not part of the THOR publication, and the experiments
here — deliberately undertrained, deliberately oversized, deliberately
mis-split — are not THOR's published results. For those, see the paper and
[`THOR/Step2`](https://github.com/maikejulie/DNN4Cli/tree/main/THOR/Step2).
