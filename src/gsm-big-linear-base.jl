function GSMBigLinearBase(n_nodes, Nᵥ, X; rng=mk_rng(123))
    # 1. define grid parameters
    n_records, n_features = size(X)

    # 2. create grid of K latent nodes
    f_d = Dirichlet(ones(Nᵥ))

    Z = zeros(n_nodes, Nᵥ)
    Z[1:Nᵥ, :] .= Diagonal(ones(Nᵥ))               # Make sure to include vertices
    Z[Nᵥ+1:end,:] .= rand(rng, f_d, n_nodes-Nᵥ)'   # Add in uniformly sampled points

    Φ = Z

    # 3. perform PCA on data to get principle component variances
    data_means = vec(mean(Array(X), dims=1))
    D = Array(X)' .- data_means
    Svd = svd(D)
    pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]


    # 4. Initialize weights
    W = rand(rng, n_features, size(Φ, 2))

    # 5. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'

    # 6. Set the variance parameter
    β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)

    # 7. Set up prior distribution for mixing coefficients
    πk = ones(n_nodes)/n_nodes  # initialize to 1/K

    # 8. Initialize responsibility matrix
    R = ones(n_nodes , n_records)
    for n ∈ axes(R,2)
        R[:,n] .= πk
    end

    # 9. return final GSM object
    return GSMLinearBase(Z, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, πk)
end

