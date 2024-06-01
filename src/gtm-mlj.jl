include("gtm-base.jl")

using MLJModelInterface
import MLJBase


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
    ξmeans = DataMeans(fitresult, Xnew)

    return MLJModelInterface.table(ξmeans; names=(:ξ₁, :ξ₂))
end



# Note that the interface changed a bit...
function predict_responsibility(m::MLJBase.Machine{GTM, GTM, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gtm = fitted_params(m)[:gtm]
    return responsibility(gtm, Xnew)
end


