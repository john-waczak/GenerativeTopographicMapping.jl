mutable struct GSMMultUpNonlinearBase{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractArray, T6 <: AbstractArray, T7 <: AbstractArray, T8 <: AbstractArray}
    Z::T1                 # Latent coordinates
    M::T2                 # RBF coordinates
    σ::Float64            # Scale Factor
    Φ::T3                 # RBF activations
    W::T4                 # RBF weights
    Ψ::T5                 # projected node means
    Δ²::T6                # Node-data distance matrix
    R::T7                 # Responsibilities
    β⁻¹::Float64           # precision
    πk::T8
end




function GSMMultUpNonlinearBase(k, m, s, Nᵥ, X; rand_init=true, rng=mk_rng(123))
    # 1. define grid parameters
    n_records, n_features = size(X)
    n_nodes = binomial(k + Nᵥ - 2, Nᵥ - 1)
    n_rbf_centers = binomial(m + Nᵥ - 2, Nᵥ - 1)

    # 2. create grid of K latent nodes
    Z = get_barycentric_grid_coords(k, Nᵥ)'

    # 3. create grid of M rbf centers (means)
    M = get_barycentric_grid_coords(m, Nᵥ)'
    σ = s * (1/k)

    # 4. create rbf activation matrix Φ
    Δ² = zeros(size(Z,1), size(M,1))
    pairwise!(sqeuclidean, Δ², Z, M, dims=1)
    Φ =  exp.(-Δ² ./ (2*σ^2))  # using gaussian RBF kernel

    # 5. perform PCA on data to get principle component variances
    data_means = vec(mean(Array(X), dims=1))
    D = Array(X)' .- data_means
    Svd = svd(D)

    U = Svd.U[:, 1:Nᵥ]
    pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]

    # 6. Initialize weights
    W = rand(rng, n_features, size(Φ, 2))

    if !(rand_init)
        # convert to loadings
        for i ∈ 1:Nᵥ
            U[:,i] .= U[:,i] .* sqrt(pca_vars[i])
        end

        Znorm = copy(Z)
        for i ∈ 1:Nᵥ
            Znorm[:,i] = (Znorm[:,i] .-  mean(Znorm[:,i])) ./ std(Znorm[:,i])
        end

        W = U*Znorm' * pinv(Φ')
    end

    # 7. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'

    if !(rand_init)
        for i ∈ axes(Ψ,1)
            Ψ[i,:] .= Ψ[i,:] .+ data_means[i]
        end
    end

    # 8. Set the variance parameter
    β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)

    # 9. Set up prior distribution for mixing coefficients
    πk = ones(n_nodes)/n_nodes  # initialize to 1/K

    # 10. Initialize responsibility matrix
    R = ones(n_nodes , n_records)
    for n ∈ axes(R,2)
        R[:,n] .= πk
    end

    # 11. return final GSM object
    return GSMMultUpNonlinearBase(Z, M, σ, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, πk)
end





function fit!(gsm::GSMMultUpNonlinearBase, X; λ = 0.1, nepochs=100, tol=1e-3, nconverged=5, verbose=false, n_steps=25)
    # get the needed dimensions
    N,D = size(X)

    K = size(gsm.Z,1)
    M = size(gsm.Φ,2)

    prefac = 0.0
    G = Diagonal(sum(gsm.R, dims=2)[:])

    # compute log of mixing coefficients
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end


    Q = 0.0
    Q_prev = 0.0

    llhs = Float64[]
    Qs = Float64[]
    nclose = 0
    converged = false

    for i in 1:nepochs
        # EXPECTATION
        mul!(gsm.Ψ, gsm.W, gsm.Φ')                             # update latent node means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)    # update distance matrix
        gsm.Δ² .*= -(1/(2*gsm.β⁻¹))
        softmax!(gsm.R, gsm.Δ² .+ LnΠ, dims=1)

        G .= Diagonal(sum(gsm.R, dims=2)[:])

        # MAXIMIZATION

        # 1. Update the πk values
        # gsm.πk .= (1/N) .* sum(gsm.R, dims=2)
        gsm.πk .= max.((1/N) .* sum(gsm.R, dims=2), eps(eltype(gsm.πk)))
        for n ∈ axes(LnΠ,2)
            LnΠ[:,n] .= log.(gsm.πk)
        end

        # 2. update weight matrix
        # gsm.W = ((gsm.Φ'*G*gsm.Φ + λ*gsm.β⁻¹*I)\(gsm.Φ'*gsm.R*X))'
        for step ∈ 1:n_steps
            gsm.W .*= (X' * gsm.R' * gsm.Φ ./ gsm.β⁻¹) ./ (gsm.W * gsm.Φ' * G * gsm.Φ ./ gsm.β⁻¹ + λ .* gsm.W)
        end


        # 3. update precision β
        mul!(gsm.Ψ, gsm.W, gsm.Φ')                         # update means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)  # update distance matrix
        gsm.β⁻¹ = sum(gsm.R .* gsm.Δ²)/(N*D)                    # update variance


        # UPDATE LOG-LIKELIHOOD
        prefac = (N*D/2)*log(1/(2* gsm.β⁻¹* π))

        if i == 1
            l = max(prefac + sum(logsumexp(gsm.Δ² .* LnΠ, dims=1)), nextfloat(typemin(1.0)))

            Q = (N*D/2)*log(1/(2* gsm.β⁻¹* π)) + sum(gsm.R .* LnΠ) + (length(gsm.W)/2)*log(λ/(2π)) - sum(gsm.R .* gsm.Δ²) - (λ/2)*sum(gsm.W)

            push!(llhs, l)
            push!(Qs, Q)
        else
            Q_prev = Q

            l = max(prefac + sum(logsumexp(gsm.Δ² .* LnΠ, dims=1)), nextfloat(typemin(1.0)))
            Q = (N*D/2)*log(1/(2* gsm.β⁻¹* π)) + sum(gsm.R .* LnΠ) + (length(gsm.W)/2)*log(λ/(2π)) - sum(gsm.R .* gsm.Δ²) - (λ/2)*sum(gsm.W)

            push!(llhs, l)
            push!(Qs, Q)

            # check for convergence
            rel_diff = abs(Q - Q_prev)/min(abs(Q), abs(Q_prev))

            if rel_diff <= tol
                # increment the number of "close" difference
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end


        if verbose
            println("iter: $(i), Q=$(Qs[end]), log-likelihood = $(llhs[end])")
        end
    end

    AIC = 2*length(gsm.W) - 2*llhs[end]
    BIC = log(size(X,1))*length(gsm.W) - 2*llhs[end]

    return converged, Qs, llhs, AIC, BIC
end




function DataMeans(gsm::GSMMultUpNonlinearBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))

    LnΠ = ones(size(Δ²))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    R = softmax(Δ² .+ LnΠ, dims=1)

    return R'*gsm.Z
end


function DataModes(gsm::GSMMultUpNonlinearBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))

    LnΠ = ones(size(Δ²))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    R = softmax(Δ² .+ LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gsm.Z[idx,:]
end


function class_labels(gsm::GSMMultUpNonlinearBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))

    LnΠ = ones(size(Δ²))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    R = softmax(Δ² .+ LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return idx
end


function responsibility(gsm::GSMMultUpNonlinearBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))

    LnΠ = ones(size(Δ²))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end


    R = softmax(Δ² .+ LnΠ, dims=1)

    return R'
end


