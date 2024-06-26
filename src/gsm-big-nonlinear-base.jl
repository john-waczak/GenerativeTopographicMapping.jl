function GSMBigNonlinearBase(n_nodes, n_rbfs, Nᵥ, s, X; rng=mk_rng(123))

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

    # 4. create rbf activation matrix Φ
    Δ² = zeros(size(Z,1), size(M,1))
    pairwise!(sqeuclidean, Δ², Z, M, dims=1)
    Φ =  exp.(-Δ² ./ (2*s^2))  # using gaussian RBF kernel


    # 5. perform PCA on data to get principle component variances
    data_means = vec(mean(Array(X), dims=1))
    D = Array(X)' .- data_means
    Svd = svd(D)
    pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]

    # 6. Initialize weights
    W = rand(rng, n_features, size(Φ, 2))

    # 7. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'

    # 8. Set the variance parameter
    β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)

    # 9. Set up prior distribution
    πk = ones(n_nodes)/n_nodes

    R = ones(n_nodes , n_records)
    for n ∈ axes(R,2)
        R[:,n] .= πk
    end


    # 11. return final GSM object
    return GSMNonlinearBase(Z, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, πk)
end

