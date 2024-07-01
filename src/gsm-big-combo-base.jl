function GSMBigComboBase(n_nodes, n_rbfs, s, Nᵥ, X; rand_init=true, rng=mk_rng(123), zero_init=true)
    # 1. define grid parameters
    n_records, n_features = size(X)

    # 2. create grid of K latent nodes
    f_d = Dirichlet(ones(Nᵥ))

    Z = zeros(n_nodes, Nᵥ)
    Z[1:Nᵥ, :] .= Diagonal(ones(Nᵥ))
    Z[Nᵥ+1:end,:] .= rand(rng, f_d, n_nodes-Nᵥ)'


    # 3. create grid of M rbf centers (means)
    #    sample and reject any that are within
    #    s of the vertices
    M = zeros(n_rbfs, Nᵥ)
    for i ∈ axes(M,1)
        is_valid = false
        while !is_valid
            rbf_center = rand(rng, f_d)
            if all(colwise(euclidean, rbf_center, Diagonal(ones(Nᵥ))) .> s)
                M[i, :] = rbf_center
                is_valid=true
            end
        end
    end


    # 4. create rbf activation matrix Φ
    Φ = []

    # add in linear terms
    push!(Φ, Z)

    let
        Δ = zeros(size(Z,1), size(M,1))
        pairwise!(euclidean, Δ, Z, M, dims=1)
        push!(Φ, linear_elem_big.(Δ, s))
    end

    Φ = hcat(Φ...)

    # 5. perform PCA on data to get principle component variances
    data_means = vec(mean(Array(X), dims=1))
    D = Array(X)' .- data_means
    Svd = svd(D)

    U = Svd.U[:, 1:Nᵥ]
    pca_vars = (abs2.(Svd.S) ./ (n_records-1))[1:Nᵥ+1]

    # 6. Initialize weights
    W = rand(rng, n_features, size(Φ, 2))
    if zero_init
        # optionally zero out the nonlinear terms
        W[:, Nᵥ+1:end] .= 0.0
    end

    if !(rand_init)
        # convert to loadings
        for i ∈ 1:Nᵥ
            U[:,i] .= U[:,i] .* sqrt(pca_vars[i])
        end

        Znorm = copy(Z)
        for i ∈ 1:Nᵥ
            Znorm[:,i] = (Znorm[:,i] .-  mean(Znorm[:,i])) ./ std(Znorm[:,i])
        end

        if zero_init
            W[:, 1:Nᵥ] = U*Znorm' * pinv(Φ[:,1:Nᵥ]')
        else
            W = U*Znorm' * pinv(Φ')
        end
    end

    # 7. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'

    # add back the means if using PCA
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
    return GSMComboBase(Z, M, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, πk)
end


