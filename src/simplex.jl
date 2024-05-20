using Combinatorics


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

    # # # 6. perform PCA on data
    # pca = MultivariateStats.fit(PCA, X'; pratio=1, maxoutdim=Nᵥ+1)
    # pca_vecs = projection(pca)
    # pca_vars = principalvars(pca)

    # println(length(pca_vars))
    # println(pca)

    # # 7. create matrix U from first Nᵥ principal axes of data cov. matrix
    # U = pca_vecs[:, 1:Nᵥ]

    # # scale by square root of variance for some reason
    # for i ∈ 1:Nᵥ
    #     U[:,i] .= U[:,i] .* sqrt(pca_vars[i])
    # end

    # # 8. Initialize parameter matrix W using U and Φ

    # Ξnorm = copy(Ξ)
    # for i ∈ 1:Nᵥ
    #     Ξnorm[:,i] = (Ξnorm[:,i] .-  mean(Ξnorm[:,i])) ./ std(Ξnorm[:,i])
    # end
    # W = U*Ξnorm' * pinv(Φ')


    # if rand_init
    #     W = rand(n_features, n_rbf_centers+1)
    # end


    W = rand(n_features, n_rbf_centers+1)

    # 9. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'


    β⁻¹ = mean(pairwise(sqeuclidean, Ψ, dims=2))/2

    # # add the means back to each row since PCA uses
    # # a mean-subtracted covariance matrix
    # if !(rand_init)
    #     for i ∈ axes(Ψ,1)
    #         Ψ[i,:] .= Ψ[i,:] .+ mean(X[:,i])
    #     end
    # end

    # # 10. See the variance parameter
    # if length(pca_vars) > Nᵥ
    #     β⁻¹ = max(pca_vars[Nᵥ+1], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)
    # else
    #     β⁻¹ = mean(pairwise(sqeuclidean, Ψ, dims=2))/2
    # end


    # 11. return final GTM object
    return GTMBase(Ξ, M, Φ, W, Ψ, zeros(n_nodes, n_records), (1/n_nodes) .* ones(n_nodes, n_records), β⁻¹)
end








