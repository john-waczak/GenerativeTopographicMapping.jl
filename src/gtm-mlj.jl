# include("gtm-base.jl")

# using MLJModelInterface
# import MLJBase


mutable struct GTM<: MLJModelInterface.Unsupervised
    k::Int
    m::Int
    s::Float64
    λ::Float64
    topology::Symbol
    nepochs::Int
    tol::Float64
    nconverged::Int
    rand_init::Bool
    rng::Any
end



function GTM(; k=16, m=4, s=1.0, λ=0.1, topology=:square, nepochs=100, tol=1e-3, nconverged=4, rand_init=false, rng=123)
    model = GTM(k, m, s, λ, topology, nepochs, tol, nconverged, rand_init, mk_rng(rng))
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end



function MLJModelInterface.clean!(m::GTM)
    warning =""

    if m.k ≤ 0
        warning *= "Parameter `k` expected to be positive, resetting to 16\n"
        m.k = 16
    end

    if m.m ≤ 0
        warning *= "Parameter `m` expected to be positive, resetting to 4\n"
        m.m = 4
    end

    if m.s ≤ 0
        warning *= "Parameter `s` expected to be positive, resetting to 1.0\n"
        m.s = 1.0
    end

    if m.λ < 0
        warning *= "Parameter `λ` expected to be non-negative, resetting to 0.1\n"
        m.λ = 0.1
    end

    if !(m.topology ∈ [:square, :cylinder, :torus])
        warning *= "Parameter `topology` expected to be one of `[:square, :cylinder, :torus]`, resetting to `:square`\n"
        m.topology = :square
    end

    if m.nepochs ≤ 0
        warning *= "Parameter `nepochs` expected to be positive, resetting to 100\n"
        m.nepochs = 100
    end

    if m.tol ≤ 0
        warning *= "Parameter `tol` expected to be positive, resetting to 1e-3\n"
        m.tol = 1e-3
    end

    if m.nconverged < 0
        warning *= "Parameter `nconverged` expected to be non-negative, resetting to 4\n"
        m.nconverged = 4
    end

    return warning
end



function MLJModelInterface.fit(m::GTM, verbosity, Datatable)
    # assume that X is a table
    X = MLJModelInterface.matrix(Datatable)

    if verbosity > 0
        verbose = true
    else
        verbose = false
    end


    # 1. build the GTM
    gtm = GTMBase(m.k, m.m, m.s, X; topology=m.topology, rand_init=m.rand_init, )

    # 2. Fit the GTM
    converged, llhs, AIC, BIC = fit!(
        gtm,
        X,
        λ = m.λ,
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
              :Ψ => gsm.W*gsm.Φ',
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


