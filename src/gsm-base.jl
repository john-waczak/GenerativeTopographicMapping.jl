mutable struct GSMBase{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractArray, T6 <: AbstractArray, T7 <: AbstractArray, T8 <: AbstractArray}
    Ξ::T1                 # Latent coordinates
    M::T2                 # RBF coordinates
    Φ::T3                 # RBF activations
    W::T4                 # RBF weights
    Ψ::T5                 # projected node means
    Δ²::T6                # Node-data distance matrix
    R::T7                 # Responsibilities
    β⁻¹::Float64           # precision
    LnΠ::T8   # prior distribution on nodes
end



function GSMBase(k, m, s, Nᵥ, α, X; rand_init=false, rng=mk_rng(123), nonlinear=true, linear=false, bias=false)
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
    Φ = []
    if nonlinear
        Δ² = zeros(size(Ξ,1), size(M,1))
        pairwise!(sqeuclidean, Δ², Ξ, M, dims=1)
        push!(Φ, exp.(-Δ² ./ (2*σ^2)))
    end

    if linear
        push!(Φ, Ξ)
    end

    if bias
        push!(Φ, ones(size(Ξ,1)))
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
    Ξnorm = copy(Ξ)
    for i ∈ 1:Nᵥ
        Ξnorm[:,i] = (Ξnorm[:,i] .-  mean(Ξnorm[:,i])) ./ std(Ξnorm[:,i])
    end

    W = U*Ξnorm' * pinv(Φ')

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
    πk = zeros(n_nodes)
    f_dirichlet = Dirichlet(α)
    e = 0.5 * (1/k)  # offset to deal with Inf value on boundary

    for k ∈ axes(Ξ,1)
        p = Ξ[k,:]
        for l ∈ axes(Ξ, 2)
            if isapprox(Ξ[k,l], 0.0, atol=1e-6)
                p[l] = e
            end
        end
        p = p ./ sum(p)
        πk[k] = pdf(f_dirichlet, p)
    end

    πk = πk ./ sum(πk)  # normalize


    LnΠ = ones(n_nodes , n_records)
    for n ∈ axes(LnΠ,2)
        LnΠ[:,n] .= log.(πk)
    end


    R = ones(n_nodes , n_records)
    for n ∈ axes(R,2)
        R[:,n] .= πk
    end


    # 11. return final GSM object
    return GSMBase(Ξ, M, Φ, W, Ψ, zeros(n_nodes, n_records), R, β⁻¹, LnΠ)
end





function fit!(gsm::GSMBase, X; λ = 0.1, nepochs=100, tol=1e-3, nconverged=5, verbose=false, make_positive=false)
    # get the needed dimensions
    N,D = size(X)

    K = size(gsm.Ξ,1)

    # M = size(gsm.M,1) + 1  # don't forget the bias term!
    M = size(gsm.Φ,2)

    prefac = 0.0
    RX = zeros(K,D)
    GΦ = zeros(K,M)
    LHS = zeros(M,M)
    RHS = zeros(M,D)

    l = 0.0
    llh_prev = 0.0
    llhs = Float64[]
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
        softmax!(gsm.R, gsm.Δ² .+ gsm.LnΠ, dims=1)

        mul!(GΦ, diagm(sum(gsm.R, dims=2)[:]), gsm.Φ)          # update the G matrix diagonal
        mul!(RX, gsm.R, X)                                 # update intermediate for R.H.S

        # UPDATE LOG-LIKELIHOOD
        prefac = (N*D/2)*log(1/(2* gsm.β⁻¹* π))

        if i == 1
            l = max(prefac + sum(logsumexp(gsm.Δ² .* gsm.LnΠ, dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)
        else
            llh_prev = l
            l = max(prefac + sum(logsumexp(gsm.Δ² .* gsm.LnΠ, dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)

            # check for convergence
            rel_diff = abs(l - llh_prev)/min(abs(l), abs(llh_prev))

            if rel_diff <= tol
                # increment the number of "close" difference
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end

        # MAXIMIZATION
        mul!(LHS, gsm.Φ', GΦ)                              # update left-hand-side
        if λ > 0
            LHS[diagind(LHS)] .+= λ * gsm.β⁻¹               # add regularization
        end
        mul!(RHS, gsm.Φ', RX)                              # update right-hand-side

        gsm.W = (LHS\RHS)'                                 # update weights

        # enforce positivity of weights
        # if desired, force weights to be positive.
        if make_positive
            gsm.W = max.(gsm.W, 0.0)
            for j ∈ axes(gsm.W, 2)
                gsm.W[:,j] = gsm.W[:,j] ./ maximum(gsm.W[:,j])
            end
        end

        mul!(gsm.Ψ, gsm.W, gsm.Φ')                         # update means
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
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ gsm.LnΠ, dims=1)

    return R'*gsm.Ξ
end


function DataModes(gsm::GSMBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ gsm.LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gsm.Ξ[idx,:]
end


function class_labels(gsm::GSMBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ gsm.LnΠ, dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return idx
end


function responsibility(gsm::GSMBase, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ² .+ gsm.LnΠ, dims=1)

    return R'
end


