using Combinatorics
using LinearAlgebra


function get_barycentric_grid_coords(Nₑ, Nᵥ)
    D = Nᵥ - 1
    Npts = binomial(Nₑ+D-1, D)

    rᵥ = collect(with_replacement_combinations([1,Nₑ], D))
    rᵢ = collect(with_replacement_combinations(1:Nₑ, D))

    Tinv = inv(hcat(rᵥ[1:end-1]...) .- rᵥ[end])
    R = hcat(rᵢ...) .- rᵥ[end]

    Λ  = zeros(Nᵥ, Npts)
    Λ[1:end-1, :] .= Tinv*R
    Λ[end:end,:] .= 1 .- sum(Λ[1:end-1, :], dims=1)

    return Λ
end



function GenerativeSimplexMap(k, m, s, Nᵥ, X; rand_init=false,)
    # 1. define grid parameters
    n_records, n_features = size(X)
    n_nodes = binomial(k + Nᵥ - 2, Nᵥ - 1)
    n_rbf_centers = binomial(m + Nᵥ - 2, Nᵥ - 1)

    # 2. create grid of K latent nodes
    Ξ = get_barycentric_grid_coords(k, Nᵥ)'

    # 3. create grid of M rbf centers (means)
    M = get_barycentric_grid_coords(m, Nᵥ)'


    # 4. initialize rbf width
    σ = s * (1/k)

    # 5. create rbf activation matrix Φ
    Φ = ones(n_nodes, n_rbf_centers+1)
    let
        Δ² = zeros(size(Ξ,1), size(M,1))
        pairwise!(sqeuclidean, Δ², Ξ, M, dims=1)
        Φ[:, 1:end-1] .= exp.(-Δ² ./ (2*σ^2))
    end

    # 6. perform PCA on data
    data_means = vec(mean(Array(X), dims=1))
    D = Array(X)' .- data_means
    Svd = svd(D)

    U = Svd.U[:, 1:Nᵥ]
    pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]


    # convert to loadings
    for i ∈ 1:Nᵥ
        U[:,i] .= U[:,i] .* sqrt(pca_vars[i])
    end

    # 7. Initialize parameter matrix W using U and Φ

    Ξnorm = copy(Ξ)
    for i ∈ 1:Nᵥ
        Ξnorm[:,i] = (Ξnorm[:,i] .-  mean(Ξnorm[:,i])) ./ std(Ξnorm[:,i])
    end
    W = U*Ξnorm' * pinv(Φ')


    if rand_init
        W = rand(n_features, n_rbf_centers+1)
    end


    # 8. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'


    # add the means back to each row since PCA uses
    # a mean-subtracted data matrix
    if !(rand_init)
        for i ∈ axes(Ψ,1)
            Ψ[i,:] .= Ψ[i,:] .+ mean(X[:,i])
        end
    end

    # 9. Set the variance parameter
    β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)


    # 10. return final GTM object
    return GTMBase(Ξ, M, Φ, W, Ψ, zeros(n_nodes, n_records), (1/n_nodes) .* ones(n_nodes, n_records), β⁻¹)
end








