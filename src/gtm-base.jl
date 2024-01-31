using LinearAlgebra
using Statistics, MultivariateStats
using Distances

mutable struct GTMBase{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray}
    Ξ::T1
    M::T2
    Φ::T3
    W::T4
    β⁻¹::Float64
    σ²::Float64
end


function GTMBase(k, m, s, X; rand_init=false)
    # 1. define grid parameters
    n_records, n_features = size(X)
    n_nodes = k*k
    n_rbf_centers = m*m

    # 2. create grid of K latent nodes
    ξ = range(-1.0, stop=1.0, length=k)
    Ξ = hcat([ξ[i] for i in axes(ξ,1) for j in axes(ξ,1)],[ξ[j] for i in axes(ξ,1) for j in axes(ξ,1)])

    # 3. create grid of M rbf centers (means)
    μ = range(-1.0, stop=1.0, length=m)
    M = hcat([μ[i] for i in axes(μ,1) for j in axes(μ,1)],[μ[j] for i in axes(μ,1) for j in axes(μ,1)])

    # 4. initialize rbf width
    σ² = s * abs(μ[2]-μ[1])

    # 5. create rbf activation matrix Φ
    Φ = ones(n_nodes, n_rbf_centers+1)
    let
        Δ² = pairwise(sqeuclidean, Ξ, M, dims=1)
        Φ[:, 1:end-1] .= exp.(-Δ² ./ (2*σ²) )
    end

    # 6. perform PCA on data
    pca = MultivariateStats.fit(PCA, X', maxoutdim=3, pratio=0.99999)
    pca_vecs = projection(pca)
    pca_vars = principalvars(pca)

    # 7. create matrix U from first two principal axes of data cov. matrix
    U = pca_vecs[:, 1:2]
    # scale by square root of variance for some reason
    U[:,1] .= U[:,1] .* sqrt(pca_vars[1])
    U[:,2] .= U[:,2] .* sqrt(pca_vars[2])

    # 8. Initialize parameter matrix W using U and Φ

    Ξnorm = copy(Ξ)
    Ξnorm[:,1] = (Ξnorm[:,1] .-  mean(Ξnorm[:,1])) ./ std(Ξnorm[:,1])
    Ξnorm[:,2] = (Ξnorm[:,2] .-  mean(Ξnorm[:,2])) ./ std(Ξnorm[:,2])

    W = U*Ξnorm' * pinv(Φ')
    if rand_init
        W = rand(n_features, n_rbf_centers+1)
    end

    # 9. Initialize data manifold Ψ using W and Φ
    Ψ = W * Φ'
    # offset by column means
    Xmeans = mean(X, dims=1)
    D = zeros(size(W,1), size(Φ,1))
    for i ∈ axes(D,1)
        D[i,:] .= Xmeans[i]
    end

    Ψ .= Ψ .+ D

    # 10. Set noise variance parameter to largest between
    #     - 3rd principal component variance
    #     - 1/2 the average distance between data manifold centers (from Y)

    β⁻¹ = max(pca_vars[3], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)
    pca_vars[3]

    # 11. return final GTM object

    return GTMBase(Ξ, M, Φ, W, β⁻¹, σ²)
end




function exp_normalize(Λ)
    maxes = maximum(Λ, dims=1)
    res = zeros(size(Λ))
    for j in axes(Λ, 2)
        res[:,j] .= exp.(Λ[:,j] .- maxes[j])
    end
    return res
end


function getPMatrix(Δ², β⁻¹)
    # use exp-normalize trick
    return exp_normalize(-(1/(2*β⁻¹)) .* Δ²)
end


function Responsabilities(P)
    R = zeros(size(P))
    for j in axes(R,2)
        R[:,j] .= P[:,j] ./ sum(P[:,j])
    end
    return R
end



function getUpdateW(R, G, Φ, X, β⁻¹, α)
    # W is the solution of
    # (Φ'GΦ + (αβ⁻¹)I)W' = Φ'RX
    if α > 0
        return ((Φ'*G*Φ + (α*β⁻¹)*I)\(Φ'*R*X))'
    else
        return ((Φ'*G*Φ)\(Φ'*R*X))'
    end
end



function getUpdateβ⁻¹(R, Δ², X)
    n_records, n_features = size(X)

    return sum(R .* Δ²)/(n_records*n_features)
end


function loglikelihood(P, β⁻¹, X, Ξ)
    N, D = size(X)
    K = size(Ξ,1)

    prexp = (1/(2* β⁻¹* π))^(D/2)

    return sum(log.((prexp/K) .* sum(P, dims=1)))
end




function fit!(gtm, X; α = 0.1, niter=100, tol=0.001, nconverged=5, printiters=false)
    # 1. create distance matrix Δ² between manifold points and data matrix
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    # 2. Until convergence, i.e. log-likelihood < tol

    l = 0.0
    llh_prev = 0.0
    llhs = Float64[]

    nclose = 0
    converged = false  # a flag to tell us if we converged successfully
    for i in 1:niter
        # expectation
        P = getPMatrix(Δ², gtm.β⁻¹)
        R = Responsabilities(P)
        G = diagm(sum(R, dims=2)[:])

        # (maximization)
        gtm.W = getUpdateW(R, G, gtm.Φ, X, gtm.β⁻¹, α)
        Ψ = gtm.W * gtm.Φ'
        #Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
        pairwise!(sqeuclidean, Δ², Ψ, X', dims=2)
        gtm.β⁻¹ = getUpdateβ⁻¹(R, Δ², X)

        # compute log-likelihood
        if i == 1
            l = loglikelihood(P, gtm.β⁻¹, X, gtm.Ξ)
            push!(llhs, l)
        else
            llh_prev = l
            l = loglikelihood(P, gtm.β⁻¹, X, gtm.Ξ)
            push!(llhs, l)

            # check for convergence
            llh_diff = abs(l - llh_prev)

            if llh_diff <= tol
                # increment the number of "close" difference
                nclose += 1
            end

            if nclose == nconverged
                converged = true
                break
            end
        end

        if printiters
            println("iter: $(i), log-likelihood = $(l)")
        end
    end

    # update responsabilities after final pass
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)

    return converged,llhs, R
end



function DataMeans(gtm, X)
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)

    return R'*gtm.Ξ
end


function DataModes(gtm, X)
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gtm.Ξ[idx,:]
end

function class_labels(gtm, X)
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return idx
end



function responsability(gtm, X)
    # return N × K table of resonsabilities

    # 0. get position of latend nodes in data space
    Ψ = gtm.W * gtm.Φ'

    # 1. create distance matrix
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)

    # 2. create P matrix
    P = getPMatrix(Δ², gtm.β⁻¹)

    # 3. create R matrix
    R = Responsabilities(P)

    return R'
end



function data_reconstruction(gtm, X)
    R = transform_responsability(gtm, X)
    Ψ = gtm.W * gtm.Φ'

    # compute rmse reproduction error...
    return R * Ψ'
end




function BIC(gtm, X)
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)
    l = loglikelihood(P, gtm.β⁻¹, X, gtm.Ξ)

    return log(size(X,1))*length(gtm.W) - 2*l
end


function AIC(gtm, X)
    Ψ = gtm.W * gtm.Φ'
    Δ² = pairwise(sqeuclidean, Ψ, X', dims=2)
    P = getPMatrix(Δ², gtm.β⁻¹)
    R = Responsabilities(P)
    l = loglikelihood(P, gtm.β⁻¹, X, gtm.Ξ)

    return 2*length(gtm.W) - 2*l
end
