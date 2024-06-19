function GSM_big(n_nodes, n_rbfs, Nᵥ, s, X; rand_init=false, rng=mk_rng(123), nonlinear=true, linear=false, bias=false)

    # 1. define grid parameters
    n_records, n_features = size(X)

    # 2. create grid of K latent nodes
    f_d = Dirichlet(ones(Nᵥ))

    Z = zeros(n_nodes, Nᵥ)
    Z[1:Nᵥ, :] .= Diagonal(ones(Nᵥ))
    Z[Nᵥ+1:end,:] .= rand(rng, f_d, n_nodes-Nᵥ)'


    # 3. create grid of M rbf centers (means)
    M = zeros(n_rbfs, Nᵥ)
    M[1:Nᵥ, :] .= Diagonal(ones(Nᵥ))
    M[Nᵥ+1:end,:] .= rand(rng, f_d, n_rbfs - Nᵥ)'

    # 5. create rbf activation matrix Φ
    Φ = []
    if nonlinear
        # Ξ has size K×Nᵥ
        Δ² = zeros(size(Z,1), size(M,1))
        pairwise!(sqeuclidean, Δ², Z, M, dims=1)
        push!(Φ, exp.(-Δ² ./ (2*s^2)))  # using gaussian RBF kernel
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


