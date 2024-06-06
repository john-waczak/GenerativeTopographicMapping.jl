module GenerativeTopographicMapping


using Random
using LinearAlgebra
using Statistics, MultivariateStats
using Combinatorics
using Distances
using LogExpFunctions
using ProgressMeter
using Distributions
using MLJModelInterface
import MLJBase


include("coords.jl")

mk_rng(rng::AbstractRNG) = rng
mk_rng(int::Integer) = Random.MersenneTwister(int)

include("gtm-base.jl")
include("gsm-log-base.jl")
include("gsm-base.jl")


include("gtm-mlj.jl")
include("gsm-log-mlj.jl")
include("gsm-mlj.jl")


export GTM
export GSMLog
export GSM

export DataMeans, DataModes
export responsibility
export predict_responsibility



# ---------------------------------------------------------------------------
# ---------- DOCS -----------------------------------------------------------
# ---------------------------------------------------------------------------



MLJModelInterface.metadata_pkg.(
    [GTM, GSMLog, GSM],
    name = "GenerativeTopographicMapping",
    uuid = "110c1e60-17ba-4aeb-8cee-444277a6d160", # see your Project.toml
    url  = "https://github.com/john-waczak/GenerativeTopographicMapping.jl",
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)


MLJModelInterface.metadata_model(
    GTM,
    input_scitype = MLJModelInterface.Table(MLJModelInterface.Continuous),
    output_scitype  = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
	  load_path    = "GenerativeTopographicMapping.GTM"
)

MLJModelInterface.metadata_model(
    GSMLog,
    input_scitype = MLJModelInterface.Table(MLJModelInterface.Continuous),
    output_scitype  = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
	  load_path    = "GenerativeTopographicMapping.GSMLog"
)


MLJModelInterface.metadata_model(
    GSM,
    input_scitype = MLJModelInterface.Table(MLJModelInterface.Continuous),
    output_scitype  = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
	  load_path    = "GenerativeTopographicMapping.GSM"
)



const DOC_GTM = ""*
    ""*
    ""



"""
$(MLJModelInterface.doc_header(GTM))

GenerativeTopographicMapping implements [Generative Topographic Mapping](https://direct.mit.edu/neco/article/10/1/215-234/6127),
  Neural Computation; Bishop, C.; (1998):
  \"GTM: The Generative Topographic Mapping\"


# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)
where
- `X`: `Table` of input features whose columns are of scitype `Continuous.`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters
- `k=16`: Number of nodes along once side of GTM latent grid. There are `k²` total nodes.
- `m=4`:  There are `m²` total RBFs.
- `s=2.0`: Scale factor for RBF variance.
- `α=0.1`:  Model weight regularization parameter (0.0 for no regularization)
- `topology=:square`: Topology of latent space. One of `:square`, `:cylinder`, or `:torus`
- `tol=0.1`: Tolerance used for determining convergence during expectation-maximization fitting.
- `nepochs=100`: Maximum number of training epochs.
- `nrepeats=4`: Number of steps to repeat at/below `tol` before GTM is considered converged.
- `rand_init=false`: Whether or not to randomly initialize weight matrix `W`. If false, `W` is initialized using first two principal components of the dataset.

# Operations
- `transform(mach, X)`: returns the coordinates `ξ₁` and `ξ₂` cooresponding to the mean of the latent node responsibility distribution.
- `predict(mach, X)`: returns the index of the node corresponding to the mode of the latent node responsibility distribution.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `gtm`: The `GenerativeTopographicMap` object fit by the `GTM` model. Contains node coordinates, RBF means, RBF variance, weights, etc.


# Report
The fields of `report(mach)` are:
- `W`: the fitted GTM weight matrix
- `β⁻¹`: the fitted GTM variance
- `Φ`: the node RBF activation matrix
- `Ξ`: the node coordinates in the latent space
- `Ψ`: the projected node means in the data space
- `llhs`: the log-likelihood values from each iteration of the training loop
- `converged`: is `true` if the convergence critera were met before reaching `niter`
- `AIC`: the Aikike Information Criterion
- `BIC`: the Bayesian Information Criterion

# Examples
```
using MLJ
gtm = @load GTM pkg=GenerativeTopographicMapping
model = gtm()
X, y = make_blob(100, 10; centers=5) # synthetic data
mach = machine(model, X) |> fit!
X̃ = transform(mach, X)

rpt = report(mach)
classes = rpt.classes
```
"""
GTM






const DOC_GSMLog= ""*
    ""*
    ""



"""
$(MLJModelInterface.doc_header(GSMLog))

Generative Simplex Mapping based based on [Generative Topographic Mapping](https://direct.mit.edu/neco/article/10/1/215-234/6127),
  Neural Computation; Bishop, C.; (1998):
  \"GTM: The Generative Topographic Mapping\"
where the latent points are configured as a gridded n-simplex and the data space is given by a lor-normal distribution.

# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)
where
- `X`: `Table` of input features whose columns are of scitype `Continuous.`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters
- `k=10`: Number of nodes along once side of GTM latent grid. There are `k²` total nodes.
- `m=5`:  There are `m²` total RBFs.
- `s=1.0`: Scale factor for RBF variance.
- `Nv=3`: Number of vertices for the simplex. Alternatively, the number of model endmembers.
- `λ=0.1`:  Model weight regularization parameter (0.0 for no regularization)
- `tol=0.1`: Tolerance used for determining convergence during expectation-maximization fitting.
- `nepochs=100`: Maximum number of training epochs.
- `nrepeats=4`: Number of steps to repeat at/below `tol` before GTM is considered converged.
- `rand_init=false`: Whether or not to randomly initialize weight matrix `W`. If false, `W` is initialized using first two principal components of the dataset.

# Operations
- `transform(mach, X)`: returns the coordinates `ξᵢ` cooresponding to the mean of the latent node responsibility distribution.
- `predict(mach, X)`: returns the index of the node corresponding to the mode of the latent node responsibility distribution.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `gsm`: The `GenerativeSimplexMap` object fit by the `GSMLog` model. Contains node coordinates, RBF means, RBF variance, weights, etc.


# Report
The fields of `report(mach)` are:
- `W`: the fitted GTM weight matrix
- `β⁻¹`: the fitted GTM variance
- `Φ`: the node RBF activation matrix
- `Ψ`: the projected node means in the data space
- `Ξ`: the node coordinates in the latent space
- `llhs`: the log-likelihood values from each iteration of the training loop
- `converged`: is `true` if the convergence critera were met before reaching `niter`
- `AIC`: the Aikike Information Criterion
- `BIC`: the Bayesian Information Criterion

# Examples
```
using MLJ
gsm = @load GSMLog pkg=GenerativeTopographicMapping
model = gsm()
X, y = make_blob(100, 10; centers=5) # synthetic data
mach = machine(model, X) |> fit!
X̃ = transform(mach, X)

rpt = report(mach)
classes = rpt.classes
```
"""
GSMLog


const DOC_GSM= ""*
    ""*
    ""



"""
$(MLJModelInterface.doc_header(GSM))

Generative Simplex Mapping based based on [Generative Topographic Mapping](https://direct.mit.edu/neco/article/10/1/215-234/6127),
  Neural Computation; Bishop, C.; (1998):
  \"GTM: The Generative Topographic Mapping\"
where the latent points are configured as a gridded n-simplex.

# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)
where
- `X`: `Table` of input features whose columns are of scitype `Continuous.`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters
- `k=10`: Number of nodes along once side of GTM latent grid. There are `k²` total nodes.
- `m=5`:  There are `m²` total RBFs.
- `s=1.0`: Scale factor for RBF variance.
- `Nv=3`: Number of vertices for the simplex. Alternatively, the number of model endmembers.
- `λ=0.1`:  Model weight regularization parameter (0.0 for no regularization)
- `tol=0.1`: Tolerance used for determining convergence during expectation-maximization fitting.
- `nepochs=100`: Maximum number of training epochs.
- `nrepeats=4`: Number of steps to repeat at/below `tol` before GTM is considered converged.

# Operations
- `transform(mach, X)`: returns the coordinates `ξᵢ` cooresponding to the mean of the latent node responsibility distribution.
- `predict(mach, X)`: returns the index of the node corresponding to the mode of the latent node responsibility distribution.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `gsm`: The `GenerativeSimplexMap` object fit by the `GSM` model. Contains node coordinates, RBF means, RBF variance, weights, etc.


# Report
The fields of `report(mach)` are:
- `W`: the fitted GTM weight matrix
- `β⁻¹`: the fitted GTM variance
- `Φ`: the node RBF activation matrix
- `Ψ`: the projected node means in the data space
- `Ξ`: the node coordinates in the latent space
- `llhs`: the log-likelihood values from each iteration of the training loop
- `converged`: is `true` if the convergence critera were met before reaching `niter`
- `AIC`: the Aikike Information Criterion
- `BIC`: the Bayesian Information Criterion

# Examples
```
using MLJ
gsm = @load GSM pkg=GenerativeTopographicMapping
model = gsm()
X, y = make_blob(100, 10; centers=5) # synthetic data
mach = machine(model, X) |> fit!
X̃ = transform(mach, X)

rpt = report(mach)
classes = rpt.classes
```
"""
GSM




end
