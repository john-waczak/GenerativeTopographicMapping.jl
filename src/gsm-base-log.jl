using Combinatorics
using LinearAlgebra
using Statistics, MultivariateStats
using Distances
using LogExpFunctions
using ProgressMeter


mutable struct GSMBaseLog{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractArray, T6 <: AbstractArray, T7 <: AbstractArray}
    Ξ::T1
    M::T2
    Φ::T3
    W::T4
    Ψ::T5
    Δ²::T6
    R::T7
    β⁻¹::Float64
end





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



function GSMBaseLog(k, m, s, Nᵥ, X; rand_init=false,)
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

    # 6. perform PCA on data

    # use nat-log of data since we are fitting log-normal
    Data = Array(X)
    Data = log.(Data .+ eps(eltype(Data)))

    data_means = vec(mean(Data, dims=1))
    D = Data' .- data_means
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
        W = rand(n_features, n_rbf_centers + Nᵥ + 1)
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


    # 10. return final GSM object
    return GSMBaseLog(Ξ, M, Φ, W, Ψ, zeros(n_nodes, n_records), (1/n_nodes) .* ones(n_nodes, n_records), β⁻¹)
end














function fit!(gsm::GSMBaseLog, X; α = 0.1, nepochs=100, tol=1e-3, nconverged=5, verbose=false)
    # get the needed dimensions
    N,D = size(X)

    K = size(gsm.Ξ,1)

    # M = size(gsm.M,1) + 1  # don't forget the bias term!
    M = size(gsm.Φ,2)

    # set up the prefactor
    LnX = log.(X .+ eps(eltype(X)))
    ΣLnX = sum(LnX)

    prefac = 0.0
    RLnX = zeros(K,D)
    GΦ = zeros(K,M)
    LHS = zeros(M,M)
    RHS = zeros(M,D)

    l = 0.0
    llh_prev = 0.0
    llhs = Float64[]
    nclose = 0
    converged = false

    for i in 1:nepochs
        # EXPECTATION
        mul!(gsm.Ψ, gsm.W, gsm.Φ')                             # update latent node means
        #pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, X', dims=2)     # update distance matrix
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, LnX', dims=2)    # update distance matrix
        gsm.Δ² .*= -(1/(2*gsm.β⁻¹))
        softmax!(gsm.R, gsm.Δ², dims=1)

        mul!(GΦ, diagm(sum(gsm.R, dims=2)[:]), gsm.Φ)          # update the G matrix diagonal
        mul!(RLnX, gsm.R, LnX)                                 # update intermediate for R.H.S

        # UPDATE LOG-LIKELIHOOD
        prefac = (N*D/2)*log(1/(2* gsm.β⁻¹* π)) - N*log(K) - ΣLnX



        if i == 1
            l = max(prefac + sum(logsumexp(gsm.Δ², dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)
        else
            llh_prev = l
            l = max(prefac + sum(logsumexp(gsm.Δ², dims=1)), nextfloat(typemin(1.0)))
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
        if α > 0
            LHS[diagind(LHS)] .+= α * gsm.β⁻¹               # add regularization
        end
        mul!(RHS, gsm.Φ', RLnX)                              # update right-hand-side

        gsm.W = (LHS\RHS)'                                 # update weights

        mul!(gsm.Ψ, gsm.W, gsm.Φ')                         # update means
        pairwise!(sqeuclidean, gsm.Δ², gsm.Ψ, LnX', dims=2)  # update distance matrix
        gsm.β⁻¹ = sum(gsm.R .* gsm.Δ²)/(N*D)                    # update variance

        if verbose
            println("iter: $(i), log-likelihood = $(l)")
        end
    end

    AIC = 2*length(gsm.W) - 2*llhs[end]
    BIC = log(size(X,1))*length(gsm.W) - 2*llhs[end]

    return converged, llhs, AIC, BIC
end





function DataMeans(gsm::GSMBaseLog, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, log.(X .+ eps(eltype(X)))', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ², dims=1)

    return R'*gsm.Ξ
end


function DataModes(gsm::GSMBaseLog, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, log.(X .+ eps(eltype(X)))', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ², dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gsm.Ξ[idx,:]
end


function class_labels(gsm::GSMBaseLog, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, log.(X .+ eps(eltype(X)))', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ², dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return idx
end


function responsibility(gsm::GSMBaseLog, X)
    mul!(gsm.Ψ, gsm.W, gsm.Φ')
    Δ² = pairwise(sqeuclidean, gsm.Ψ, log.(X .+ eps(eltype(X)))', dims=2)
    Δ² .*= -(1/(2*gsm.β⁻¹))
    R = softmax(Δ², dims=1)

    return R'
end


