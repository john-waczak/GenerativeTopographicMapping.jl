module GenerativeTopographicMapping


include("gtm-base.jl")

export DataMeans, DataModes
export responsibility, data_reproduction

using MLJModelInterface
import MLJBase
export predict_responsibility

MLJModelInterface.@mlj_model mutable struct GTM <: MLJModelInterface.Unsupervised
    k::Int = 16::(_ > 0)
    m::Int = 4::(_ > 0)
    s::Float64 = 2.0::(_ > 0)
    α::Float64 = 0.1::(_ ≥ 0)
    topology::Symbol = :square::(_ in (:square, :cylinder, :torus))
    nepochs::Int = 100::(_ ≥ 1)
    batchsize::Int = 0::(_ ≥ 0)
    tol::Float64 = 1e-3::(_ > 0)
    nconverged::Int = 4::(_ ≥ 1)
    rand_init::Bool = false::(_ in (false, true))
end



function MLJModelInterface.fit(m::GTM, verbosity, Datatable)
    # assume that X is a table
    X = MLJModelInterface.matrix(Datatable)

    batchsize = m.batchsize
    if batchsize ≥ size(X,1)
        println("Batch size is ≥ number of records. Setting batch size to n records...")
        batchsize=0
    end


    if verbosity > 0
        verbose = true
    else
        verbose = false
    end


    # 1. build the GTM
    gtm = GTMBase(m.k, m.m, m.s, X; topology=m.topology, rand_init=m.rand_init)

    # 2. Fit the GTM
    converged, llhs, AIC, BIC = fit!(
        gtm,
        X,
        α = m.α,
        nepochs=m.nepochs,
        tol=m.tol,
        nconverged=m.nconverged,
        verbose=verbose,
    )

    # if batchsize == 0
    #     converged, llhs, AIC, BIC  = fit!(
    #         gtm,
    #         X,
    #         α = m.α,
    #         nepochs=m.nepochs,
    #         tol=m.tol,
    #         nconverged=m.nconverged,
    #         verbose=verbose,
    #     )
    # else
    #     converged, llhs, AIC, BIC = fit_incremental!(
    #         gtm,
    #         X,
    #         α = m.α,
    #         nepochs=m.nepochs,
    #         batchsize=batchsize,
    #         tol=m.tol,
    #         nconverged=m.nconverged,
    #         verbose=verbose,
    #     )
    # end

    # 3. Collect results
    cache = nothing
    report = (;
              :W => gtm.W,
              :β⁻¹ => gtm.β⁻¹,
              :Φ => gtm.Φ,
              :Ξ => gtm.Ξ,
              :llhs => llhs,
              :converged => converged,
              :AIC => AIC,
              :BIC => BIC,
              )

    return (gtm, cache, report)
end



MLJModelInterface.fitted_params(m::GTM, fitresult) = (gtm=fitresult,)


function MLJModelInterface.predict(m::GTM, fitresult, Data_new)
    # Return the mode index as a class label
    Xnew = MLJModelInterface.matrix(Data_new)
    labels = MLJModelInterface.categorical(1:m.k^2) # there are k^2 many SOM nodes

    return labels[class_labels(fitresult, Xnew)]
end


function MLJModelInterface.transform(m::GTM, fitresult, Data_new)
    # return a table with the mean ξ₁ and ξ₂ for each record
    Xnew = MLJModelInterface.matrix(Data_new)
    μs = DataMeans(fitresult, Xnew)

    return MLJModelInterface.table(μs; names=(:ξ₁, :ξ₂))
end



function predict_responsibility(m::MLJBase.Machine{GTM, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gtm = fitted_params(m)[:gtm]
    return responsibility(gtm, Xnew)
end


MLJModelInterface.metadata_pkg.(
    GTM,
    name = "GenerativeTopographicMapping",
    uuid = "110c1e60-17ba-4aeb-8cee-444277a6d160", # see your Project.toml
    url  = "https://github.com/john-waczak/GenerativeTopographicMapping.jl",  # URL to your package repo
    julia = true,          # is it written entirely in Julia?
    license = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)

# Then for each model,
MLJModelInterface.metadata_model(
    GTM,
    input_scitype = MLJModelInterface.Table(MLJModelInterface.Continuous),
    output_scitype  = MLJModelInterface.Table(MLJModelInterface.Continuous),
    supports_weights = false,
	  load_path    = "GenerativeTopographicMapping.GTM"
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
- `niter=200`: Maximum number of training epochs.
- `nrepeats=4`: Number of steps to repeat at/below `tol` before GTM is considered converged.
- `rand_init=false`: Whether or not to randomly initialize weight matrix `W`. If false, `W` is initialized using first two principal components of the dataset.

# Operations
- `transform(mach, X)`: returns the coordinates `ξ₁` and `ξ₂` cooresponding to the mean of the latent node responsibility distribution.
- `predict(mach, X)`: returns the index of the node corresponding to the mode of the latent node responsibility distribution.

# Fitted parameters
The fields of `fitted_params(mach)` are:

- `gtm`: The `GenerativeTopographicMap` object fit by the `GTM` model. Contains node coordinates, RbF means, RBF variance, weights, etc.


# Report
The fields of `report(mach)` are:
- `W`: the fitted GTM weight matrix
- `β⁻¹`: the fitted GTM variance
- `Φ`: the node RBF activation matrix
- `Ξ`: the node coordinates in the latent space
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


end
