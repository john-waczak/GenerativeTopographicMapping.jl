mutable struct GSMBigCombo<: MLJModelInterface.Unsupervised
    n_nodes::Int
    n_rbfs::Int
    s::Float64
    Nv::Int
    λe::Float64
    λw::Float64
    nepochs::Int
    niters::Int
    tol::Float64
    nconverged::Int
    rng::Any
    zero_init::Bool
end



function GSMBigCombo(; n_nodes=1000, n_rbfs=500, s=0.05, Nv=3, λe=0.01, λw=0.1, nepochs=100, niters=100, tol=1e-3, nconverged=4, rng=123, zero_init=true)
    model = GSMBigCombo(n_nodes, n_rbfs, s, Nv, λe, λw, nepochs, niters, tol, nconverged, mk_rng(rng), zero_init)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end


function MLJModelInterface.clean!(m::GSMBigCombo)
    warning =""

    if m.n_nodes ≤ 0
        warning *= "Parameter `n_nodes` expected to be positive, resetting to 1000\n"
        m.nodes = 1000
    end

    if m.n_rbfs ≤ 0
        warning *= "Parameter `n_rbfs` expected to be positive, resetting to 500\n"
        m.n_rbfs = 500
    end

    if m.s ≤ 0.0 || m.s ≥ 1.0
        warning *= "Parameter `s` expected to be in [0,1], resetting to 0.05\n"
        m.s = 0.05
    end

    if m.Nv ≤ 0
        warning *= "Parameter `Nv` expected to be positive, resetting to 3\n"
        m.Nv = 3
    end

    if m.λe < 0
        warning *= "Parameter `λe` expected to be non-negative, resetting to 0.01\n"
        m.λe = 0.01
    end

    if m.λw < 0
        warning *= "Parameter `λw` expected to be non-negative, resetting to 0.1\n"
        m.λw = 0.1
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




function MLJModelInterface.fit(m::GSMBigCombo, verbosity, Datatable)
    # assume that X is a table
    X = MLJModelInterface.matrix(Datatable)

    if verbosity > 0
        verbose = true
    else
        verbose = false
    end

    # 1. build the GTM
    gsm = GSMBigComboBase(m.n_nodes, m.n_rbfs, m.s, m.Nv, X; rng=m.rng, zero_init=m.zero_init)

    @assert all(gsm.W .≥ 0.0)

    # 2. Fit the GSM
    converged, Qs, llhs, AIC, BIC = fit!(
        gsm,
        m.Nv,
        X,
        λe = m.λe,
        λw = m.λw,
        nepochs = m.nepochs,
        tol = m.tol,
        nconverged = m.nconverged,
        verbose = verbose,
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



MLJModelInterface.fitted_params(m::GSMBigCombo, fitresult) = (gsm=fitresult,)


function MLJModelInterface.predict(m::GSMBigCombo, fitresult, Data_new)
    # Return the mode index as a class label
    Xnew = MLJModelInterface.matrix(Data_new)
    labels = MLJModelInterface.categorical(1:m.n_nodes)

    return labels[class_labels(fitresult, Xnew)]
end



function MLJModelInterface.transform(m::GSMBigCombo, fitresult, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    zmeans = DataMeans(fitresult, Xnew)
    names = [Symbol("Z$(i)") for i ∈ 1:m.Nv]

    return MLJModelInterface.table(zmeans; names=names)
end




function predict_responsibility(m::MLJBase.Machine{GSMBigCombo, GSMBigCombo, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gsm = fitted_params(m)[:gsm]
    return responsibility(gsm, Xnew)
end




function data_reconstruction(m::MLJBase.Machine{GSMBigCombo, GSMBigCombo, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gsm = fitted_params(m)[:gsm]

    # compute means
    zmeans = DataMeans(gsm, Xnew)

    # compute new Φ
    Φ = []

    # add in linear terms
    push!(Φ, zmeans)

    let
        Δ = pairwise(euclidean, zmeans, gsm.M, dims=1)
        push!(Φ, linear_elem_big.(Δ, m.model.s))
    end

    Φ = hcat(Φ...)

    return Φ * gsm.W'
end


