mutable struct GSMCombo<: MLJModelInterface.Unsupervised
    n_nodes::Int
    n_rbfs::Int
    Nv::Int
    s::Float64
    λe::Float64
    λw::Float64
    make_positive::Bool
    nepochs::Int
    tol::Float64
    nconverged::Int
    rand_init::Bool
    rng::Any
end

function GSMCombo(; n_nodes=1000, n_rbfs=500, Nv=3, s=0.05, λe=0.01, λw=0.1, make_positive=false, nepochs=100, tol=1e-3, nconverged=4, rand_init=false, rng=123)
    model = GSMCombo(n_nodes, n_rbfs, Nv, s, λe, λw, make_positive, nepochs, tol, nconverged, rand_init, mk_rng(rng))
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end


function MLJModelInterface.clean!(m::GSMCombo)
    warning =""

    if m.n_nodes ≤ 0
        warning *= "Parameter `n_nodes` expected to be positive, resetting to 1000\n"
        m.n_nodes = 1000
    end

    if m.n_rbfs ≤ 0
        warning *= "Parameter `n_rbfs` expected to be positive, resetting to 500\n"
        m.n_rbfs = 500
    end

    if m.Nv ≤ 0
        warning *= "Parameter `Nv` expected to be positive, resetting to 3\n"
        m.Nv = 3
    end

    if m.s ≤ 0.0 || m.s ≥ 1.0
        warning *= "Parameter `s` expected to be in [0,1], resetting to 0.05\n"
        m.s = 0.05
    end

    if m.λe < 0
        warning *= "Parameter `λ` expected to be non-negative, resetting to 0.1\n"
        m.λe = 0.1
    end

    if m.λw < 0
        warning *= "Parameter `λ` expected to be non-negative, resetting to 0.1\n"
        m.λw = 0.1
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




function MLJModelInterface.fit(m::GSMCombo, verbosity, Datatable)
    # assume that X is a table
    X = MLJModelInterface.matrix(Datatable)

    if verbosity > 0
        verbose = true
    else
        verbose = false
    end

    # 1. build the GTM
    gsm = GSM_big(m.n_nodes, m.n_rbfs, m.Nv, m.s, X; rand_init=m.rand_init, rng=m.rng, nonlinear=true, linear=true, bias=false)

    # 2. Fit the GSM
    converged, Qs, llhs, AIC, BIC = fit_combo!(
        gsm,
        m.Nv,
        X,
        λe = m.λe,
        λw = m.λw,
        nepochs=m.nepochs,
        tol=m.tol,
        nconverged=m.nconverged,
        verbose=verbose,
        make_positive=m.make_positive
    )

    # 3. Collect results
    cache = nothing

    # get indices of vertices
    idx_vertices = 1:m.Nv

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



MLJModelInterface.fitted_params(m::GSMCombo, fitresult) = (gsm=fitresult,)


function MLJModelInterface.predict(m::GSMCombo, fitresult, Data_new)
    # Return the mode index as a class label
    Xnew = MLJModelInterface.matrix(Data_new)
    labels = MLJModelInterface.categorical(1:m.n_nodes)

    return labels[class_labels(fitresult, Xnew)]
end



function MLJModelInterface.transform(m::GSMCombo, fitresult, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    zmeans = DataMeans(fitresult, Xnew)
    names = [Symbol("z_$(i)") for i ∈ 1:m.Nv]

    return MLJModelInterface.table(zmeans; names=names)
end




function predict_responsibility(m::MLJBase.Machine{GSMCombo, GSMCombo, true}, Data_new)
    Xnew = MLJModelInterface.matrix(Data_new)
    gsm = fitted_params(m)[:gsm]
    return responsibility(gsm, Xnew)
end









