mutable struct GSMBase{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractArray, T6 <: AbstractArray, T7 <: AbstractArray}
    Z::T1                 # Latent coordinates
    Φ::T2                 # RBF activations
    W::T3                 # RBF weights
    Ψ::T4                 # projected node means
    Δ²::T5                # Node-data distance matrix
    R::T6                 # Responsibilities
    β⁻¹::Float64           # precision
    πk::T7
end




function GSMBase(k, m, s, Nᵥ, X; rand_init=false, rng=mk_rng(123), nonlinear=true, linear=false, bias=false)
    # 1. define grid parameters
    n_records, n_features = size(X)
    n_nodes = binomial(k + Nᵥ - 2, Nᵥ - 1)
    n_rbf_centers = binomial(m + Nᵥ - 2, Nᵥ - 1)

    # 2. create grid of K latent nodes
    Z = get_barycentric_grid_coords(k, Nᵥ)'

    # 3. create grid of M rbf centers (means)
    M = get_barycentric_grid_coords(m, Nᵥ)'
    σ = s * (1/k)

    # 5. create rbf activation matrix Φ
    Φ = []
    if nonlinear
        # Ξ has size K×Nᵥ
        Δ² = zeros(size(Z,1), size(M,1))
        pairwise!(sqeuclidean, Δ², Z, M, dims=1)
        push!(Φ, exp.(-Δ² ./ (2*σ^2)))  # using gaussian RBF kernel
    end

    if linear
        push!(Φ, Z)
    end

    if bias
        push!(Φ, ones(size(Z,1)))
    end

    # join them together
    Φ = hcat(Φ...)


    # 6. perform PCA on data to get principle component variances
    data_means = vec(mean(Array(X), dims=1))
    D = Array(X)' .- data_means
    Svd = svd(D)

    U = Svd.U[:, 1:Nᵥ]
    pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]


    # convert to loadings
    for i ∈ 1:Nᵥ
        U[:,i] .= U[:,i] .* sqrt(pca_vars[i])
    end


    # 7. Randomly initialize the weights
    Znorm = copy(Z)
    for i ∈ 1:Nᵥ
        Znorm[:,i] = (Znorm[:,i] .-  mean(Znorm[:,i])) ./ std(Znorm[:,i])
    end

    W = U*Znorm' * pinv(Φ')

    if rand_init
        W = rand(rng, n_features, size(Φ, 2))
    end

    # 8. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'

    # add the means back to each row since PCA uses
    # a mean-subtracted data matrix

    if !(rand_init)
        for i ∈ axes(Ψ,1)
            Ψ[i,:] .= Ψ[i,:] .+ data_means[i]
        end
    end

    # 9. Set the variance parameter
    β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)


    # 10. Set up prior distribution

    # Initialize to value of 1/K
    πk = ones(n_nodes)/n_nodes

    R = ones(n_nodes , n_records)
    for n ∈ axes(R,2)
        R[:,n] .= πk
    end


    # 11. return final GSM object
    return GSMBase(Z, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, πk)
end





function fit!(gsm::GSMBase, X; λ = 0.1, nepochs=100, tol=1e-3, nconverged=5, verbose=false, make_positive=false)
    # get the needed dimensions
    N,D = size(X)

    K = size(gsm.Z,1)

    M = size(gsm.Φ,2)

    prefac = 0.0
    # RX = zeros(K,D)
    # GΦ = zeros(K,M)
    # LHS = zeros(M,M)
    # RHS = zeros(M,D)
    G = Diagonal(sum(gsm.R, dims=2)[:])

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

        # if desired, force weights to be positive.
        if make_positive
            gsm.W = max.(gsm.W, 0.0)
        end



        # EXPECTATION
        mul!(gsm.Ψ, gsm.W, gsm.Φ')                             # update latent node means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)    # update distance matrix
        gsm.Δ² .*= -(1/(2*gsm.β⁻¹))
        softmax!(gsm.R, gsm.Δ² .+ LnΠ, dims=1)

        G .= Diagonal(sum(gsm.R, dims=2)[:])

        # mul!(GΦ, Diagonal(sum(gsm.R, dims=2)[:]), gsm.Φ)          # update the G matrix diagonal
        # mul!(RX, gsm.R, X)                                 # update intermediate for R.H.S

        # MAXIMIZATION

        # 1. Update the πk values
        # gsm.πk .= (1/N) .* sum(gsm.R, dims=2)
        gsm.πk .= max.((1/N) .* sum(gsm.R, dims=2), eps(eltype(gsm.πk)))
        for n ∈ axes(LnΠ,2)
            LnΠ[:,n] .= log.(gsm.πk)
        end

        # 2. update weight matrix
        gsm.W = ((gsm.Φ'*G*gsm.Φ + λ*gsm.β⁻¹*I)\(gsm.Φ'*gsm.R*X))'

        # if desired, force weights to be positive.
        if make_positive
            gsm.W = max.(gsm.W, 0.0)
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




function DataMeans(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end


    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    return R'*gsm.Z
end


function DataModes(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end


    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gsm.Z[idx,:]
end


function class_labels(gsm::GSMBase, X)
    LnΠ = ones(size(gsm.R))
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(gsm.πk)
    end

    mul!(gsm.Ψ, gsm.W, gsm.Φ')
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

    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ LnΠ, dims=1)

    return R'
end


