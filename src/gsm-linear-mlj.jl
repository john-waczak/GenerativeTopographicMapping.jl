mutable struct GSMLinear<: MLJModelInterface.Unsupervised
    k::Int
    Nv::Int
    λ::Float64
    nepochs::Int
    niters::Int
    tol::Float64
    nconverged::Int
    rng::Any
end



function GSMLinear(; k=10, Nv=3, λ=0.1, nepochs=100, niters=100, tol=1e-3, nconverged=4, rng=123)
    model = GSMLinear(k, Nv, λ, nepochs, niters, tol, nconverged, mk_rng(rng))
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end


function MLJModelInterface.clean!(m::GSMLinear)
    warning =""

    if m.k ≤ 0
        warning *= "Parameter `k` expected to be positive, resetting to 10\n"
        m.k = 10
    end

    if m.Nv ≤ 0
        warning *= "Parameter `Nv` expected to be positive, resetting to 3\n"
        m.Nv = 3
    end

    if m.λ < 0
        warning *= "Parameter `λ` expected to be non-negative, resetting to 0.1\n"
        m.λ = 0.1
    end

    if m.nepochs ≤ 0
        warning *= "Parameter `nepochs` expected to be positive, resetting to 100\n"
        m.nepochs = 100
    end

    if m.niters ≤ 0
        warning *= "Parameter `niters` expected to be positive, resetting to 100\n"
        m.niters = 100
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




function MLJModelInterface.fit(m::GSMLinear, verbosity, Datatable)
    # assume that X is a table
    X = MLJModelInterface.matrix(Datatable)

    if verbosity > 0
        verbose = true
    else
        verbose = false
    end

    # 1. build the GTM
    gsm = GSMLinearBase(m.k, m.Nv, X; rng=m.rng)

    # 2. Fit the GSM
    converged, Qs, llhs, AIC, BIC = fit!(
        gsm,
        X,
        λ = m.λ,
        nepochs=m.nepochs,
        tol=m.tol,
        nconverged=m.nconverged,
        verbose=verbose,
        n_steps = m.niters
    )

    # 3. Collect results
    cache = nothing

    # get indices of vertices
    idx_vertices = vcat([findall(isapprox.(gsm.Z[:,j], 1, atol=1e-8)) for j ∈ axes(gsm.Z,2)]...)

    report = (;
              :W => gsm.W,
              :β⁻¹ => gsm.β⁻¹,
              :Φ => gsm.Φ,
              :node_data_means => gsm.W*gsm.Φ',
              :Z => gsm.Z,
              :Q => Qs,
              :llhs => llhs,
              :converged => converged,
              :AIC => AIC,
              :BIC => BIC,
              :idx_vertices => idx_vertices
              )

    return (gsm, cache, report)
end



MLJModelInterface.fitted_params(m::GSMLinear, fitresult) = (gsm=fitresult,)


function MLJModelInterface.predict(m::GSMLinear, fitresult, Data_new)
    # Return the mode index as a class label
    Xnew = MLJModelInterface.matrix(Data_new)
    n_nodes = binomial(m.k + m.Nv - 2, m.Nv - 1)
    labels = MLJModelInterface.categorical(1:n_nodes) # there are k^2 many SOM nodes

    return labels[class_labels(fitresult, Xnew)]
end



function MLJModelInterface.transform(m::GSMLinear, fitresult, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    zmeans = DataMeans(fitresult, Xnew)
    names = [Symbol("z_$(i)") for i ∈ 1:m.Nv]

    return MLJModelInterface.table(zmeans; names=names)
end




function predict_responsibility(m::MLJBase.Machine{GSMLinear, GSMLinear, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gsm = fitted_params(m)[:gsm]
    return responsibility(gsm, Xnew)
end


function data_reconstruction(m::MLJBase.Machine{GSMLinear, GSMLinear, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gsm = fitted_params(m)[:gsm]

    # compute means
    zmeans = DataMeans(gsm, Xnew)

    # compute new Φ
    Φ = zmeans

    return Φ * gsm.W'
end









