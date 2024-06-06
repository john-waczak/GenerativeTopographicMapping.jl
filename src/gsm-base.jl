mutable struct GSMBase{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractArray, T6 <: AbstractArray, T7 <: AbstractArray}
    Ξ::T1                 # Latent coordinates
    M::T2                 # RBF coordinates
    Φ::T3                 # RBF activations
    W::T4                 # RBF weights
    Ψ::T5                 # projected node means
    Δ²::T6                # Node-data distance matrix
    R::T7                 # Responsibilities
    β⁻¹::Float64           # precision
    πk::Vector{Float64}   # prior distribution on nodes
end




ELU(x) = (x ≥ 0) ? (x + 1)  : exp(x)
dELU_dx(x) = (x ≥ 0) ? (1) : exp(x)
InvELU(y) = (y ≥ 1) ? (y - 1)  : log(y)


function GSMBase(k, m, s, Nᵥ, α, X; rng=mk_rng(123))
    # 1. define grid parameters
    n_records, n_features = size(X)
    n_nodes = binomial(k + Nᵥ - 2, Nᵥ - 1)
    n_rbf_centers = binomial(m + Nᵥ - 2, Nᵥ - 1)

    # 2. create grid of K latent nodes
    Ξ = get_barycentric_grid_coords(k, Nᵥ)'

    # 3. create grid of M rbf centers (means)
    M = get_barycentric_grid_coords(m, Nᵥ)'


    # 4. initialize rbf width
    σ = s * (1/k)  # all side lengths are 1.0

    # 5. create rbf activation matrix Φ
    Φ = ones(n_nodes, n_rbf_centers + Nᵥ + 1)
    let
        Δ² = zeros(size(Ξ,1), size(M,1))
        pairwise!(sqeuclidean, Δ², Ξ, M, dims=1)
        Φ[:, 1:end-Nᵥ-1] .= exp.(-Δ² ./ (2*σ^2))
        Φ[:, end-Nᵥ:end-1] .= Ξ
    end

    # 6. perform PCA on data to get principle component variances
    pca_vars = zeros(Nᵥ+1)
    let
        Data = Array(X)
        data_means = vec(mean(Data, dims=1))
        D = Data' .- data_means
        Svd = svd(D)

        # U = Svd.U[:, 1:Nᵥ]
        pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]
    end

    # 7. Randomly initialize the weights
    W = rand(rng, n_features, n_rbf_centers + Nᵥ + 1)

    # 8. Initialize data manifold Ψ using W and Φ
    Ψ = ELU.(W) * Φ'

    # 9. Set the variance parameter
    β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)


    # 10. Set up prior distribution
    πk = zeros(n_nodes)
    f_dirichlet = Dirichlet(α)
    e = 0.5 * (1/k)  # offset to deal with Inf value on boundary

    for k ∈ axes(Ξ,1)
        p = Ξ[k,:]
        for l ∈ axes(Ξ, 2)
            p[l] = e
        end
        p = p ./ sum(p)
        πk[k] = pdf(f_dirichlet, p)
    end

    πk = πk ./ sum(πk)  # normalize

    R = ones(n_nodes , n_records)
    for n ∈ axes(R,2)
        R[:,n] .= πk
    end


    # 11. return final GSM object
    return GSMBase(Ξ, M, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, πk)
end



function fit!(gsm::GSMBase, X; λ = 0.1, η=0.001, nepochs=100, tol=1e-3, nconverged=5, verbose=false)
    # get the needed dimensions
    N,D = size(X)
    K = size(gsm.Ξ,1)

    # M = size(gsm.M,1) + 1  # don't forget the bias term!
    M = size(gsm.Φ,2)

    # log of node weights ax K×N matrix for computing responsibilities
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    prefac = 0.0

    # RX = zeros(K,D)
    # GΦ = zeros(K,M)
    # LHS = zeros(M,M)
    # RHS = zeros(M,D)
    F = zeros(size(gsm.W))
    Fprime = zeros(size(gsm.W))

    l = 0.0
    llh_prev = 0.0
    llhs = Float64[]
    nclose = 0
    converged = false

    for i in 1:nepochs
        # EXPECTATION

        # is this allocating???
        F .= ELU.(gsm.W)
        Fprime .= dELU_dx.(gsm.W)

        mul!(gsm.Ψ, F, gsm.Φ')                                # update latent node means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)     # update distance matrix
        gsm.Δ² .*= -(1/(2*gsm.β⁻¹))
        softmax!(gsm.R, gsm.Δ² .+ LnΠ, dims=1)

        # UPDATE LOG-LIKELIHOOD
        prefac = (N*D/2)*log(1/(2* gsm.β⁻¹* π))

        if i == 1
            l = max(prefac + sum(logsumexp(gsm.Δ² .* LnΠ, dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)
        else
            llh_prev = l
            l = max(prefac + sum(logsumexp(gsm.Δ² .* LnΠ, dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)

            # check for convergence
            rel_diff = abs(l - llh_prev)/min(abs(l), abs(llh_prev))

            if rel_diff <= tol
                # increment the number of "close" updates
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end

        # MAXIMIZATION Step

        # 1. Update W
        β = 1/(gsm.β⁻¹)
        G = Diagonal(sum(gsm.R, dims=2)[:])

        gsm.W .+= η .* (β .* (X'*gsm.R'*gsm.Φ) .* Fprime - β .*(F*gsm.Φ'*G*gsm.Φ) .* Fprime - λ .* F .* Fprime)
        # gsm.W .= gsm.W .+ η .* (β .* (X'*gsm.R'*gsm.Φ) .* Fprime .- β .*(F*gsm.Φ'*G*gsm.Φ) .* Fprime)


        # 2. Update β
        mul!(gsm.Ψ, ELU.(gsm.W), gsm.Φ')                         # update means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)  # update distance matrix
        gsm.β⁻¹ = sum(gsm.R .* gsm.Δ²)/(N*D)                    # update variance

        if verbose
            println("iter: $(i), log-likelihood = $(l)")
        end
    end

    AIC = 2*length(gsm.W) - 2*llhs[end]
    BIC = log(size(X,1))*length(gsm.W) - 2*llhs[end]

    return converged, llhs, AIC, BIC
end





function DataMeans(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    mul!(gsm.Ψ, ELU.(gsm.W), gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    return R'*gsm.Ξ
end


function DataModes(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    mul!(gsm.Ψ, ELU.(gsm.W), gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gsm.Ξ[idx,:]
end


function class_labels(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    mul!(gsm.Ψ, ELU.(gsm.W), gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return idx
end


function responsibility(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    mul!(gsm.Ψ, ELU.(gsm.W), gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    return R'
end


