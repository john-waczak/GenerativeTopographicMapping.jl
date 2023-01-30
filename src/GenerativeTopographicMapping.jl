module GenerativeTopographicMapping



import MLJModelInterface
import MLJModelInterface: Continuous, Multiclass, metadata_model, metadata_pkg
import Random.GLOBAL_RNG

include("GTM.jl")


## CONSTANTS
const MMI = MLJModelInterface
const PKG = "SelfOrganizingMaps"

MMI.@mlj_model mutable struct GTM <: MMI.Unsupervised
    k::Int = 16::(_ ≥ 5)
    m::Int = 4::(_ ≥ 2)
    σ::Float64 = 0.3::(_ > 0.0)
    α::Float64 = 0.1::(_ ≥ 0.0)
    tol::Float64 = 0.0001::(_ > 0.0)
    niter::Int = 200::(_ ≥ 1)
    nrepeats::Int = 4::(_ ≥ 1)
    representation::Symbol = :means::(_ ∈ (:means, :modes))
end



function MMI.fit(g::GTM, verbosity::Int, X)
    Dataset = MMI.matrix(X)

    if verbosity > 0
        printiters = true
    else
        printiters = false
    end

    # 1. build the GTM
    gtm = GenerativeTopographicMap(g.k,
                                   g.m,
                                   g.σ,
                                   Dataset;
                                   α=g.α,
                                   tol=g.tol,
                                   niter=g.niter,
                                   nrepeats=g.nrepeats,
                                   verbose=printiters
                                   )
    # 2. Fit the GTM
    fit!(gtm, Dataset)

    # 3. collect results
    cache  = nothing
    # put class labels in the report
    classes = getModeidx(gtm, Dataset)
    report = (; :classes=>classes)

    return gtm, cache, report
end


# there's lots you can do with this so just return the whole thing anyways
MMI.fitted_params(m::GTM, fitresult) = (gtm=fitresult,)


function MMI.transform(m::GTM, fitresult, X)
    gtm = fitresult
    # return the coordinates of the bmu for each instance
    Dataset = MMI.matrix(X)

    if m.representation == :means
        res = get_means(gtm, Dataset)
    else
        res = get_modes(gtm, Dataset)
    end

    return res
end



metadata_pkg.(
    (GTM,),
    name="GenerativeTopographicMapping",
    uuid="110c1e60-17ba-4aeb-8cee-444277a6d160",
    url="https://github.com/john-waczak/GenerativeTopographicMapping.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
)

metadata_model(
    GTM,
    input_scitype = Union{AbstractMatrix{Continuous}, MMI.Table(Continuous)},
    output_scitype = AbstractMatrix{Continuous},
    load_path = "$(PKG).GenerativeTopographicMapping",
)



# ------------ documentation ------------------------


const DOC_GTM = "[Generative Topographic Mapping](https://direct.mit.edu/neco/article/10/1/215-234/6127)"*
    ", Neural Computation; Bishop, C.; (1998):"*
    "\"GTM: The Generative Topographic Mapping\""


"""
$(MMI.doc_header(GenerativeTopographicMapping))

GenerativeTopographicMapping implements $(DOC_GTM)

# Training data
In MLJ or MLJBase, bind an instance `model` to data with
    mach = machine(model, X)
where
- `X`: an `AbstractMatrix` or `Table` of input features whose columns are of scitype `Continuous.`

Train the machine with `fit!(mach, rows=...)`.

# Hyper-parameters
- `k=16`: Number of nodes along once side of GTM latent grid. There are `k²` total nodes.
- `m=4`: Square root of the number of RBF functions in latent transformation. There are `m²` total RBFs.
- `σ=0.3`: Standard deviation for RBF functions in latent transformation.
- `α=0.1`  Model weight regularization parameter (0.0 for regularization)
- `tol=0.0001` Tolerance used for determining convergence during expectation-maximization fitting.
- `niter=200` Maximum number of iterations to use.
- `nrepeats=4` Number of steps to repeat at/below `tol` before GTM is considered converged.
- `representation=:means` Method to apply to fitted responsability matrix. One of `(:means, :modes)`.

# Operations
- `transform(mach, X)`: returns the coordinates corresponding to mean latent node responsability or mode latent node responsability for each data point. This can be used as a two-dimensional representation of the original dataset `X`.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `gtm`: The `GenerativeTopographicMap` object fit by the `GTM` model. Contains node coordinates, RbF means, RBF variance, weights, etc.


# Report
The fields of `report(mach)` are:
- `classes`: the index of the mode node responsability for each datapoint in X interpreted as a class label

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







end


