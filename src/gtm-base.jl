using LinearAlgebra
using Statistics, MultivariateStats
using Distances
using LogExpFunctions
using ProgressMeter

mutable struct GTMBase{T1 <: AbstractArray, T2 <: AbstractArray, T3 <: AbstractArray, T4 <: AbstractArray, T5 <: AbstractArray, T6 <: AbstractArray, T7 <: AbstractArray}
    Ξ::T1
    M::T2
    Φ::T3
    W::T4
    Ψ::T5
    Δ²::T6
    R::T7
    β⁻¹::Float64
end


function GTMBase(k, m, s, X; rand_init=false, topology=:square)
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
    σ = s * abs(μ[2]-μ[1])

    # 5. create rbf activation matrix Φ
    Φ = ones(n_nodes, n_rbf_centers+1)
    let
        Δ² = zeros(size(Ξ,1), size(M,1))
        ℓ = 2  # height and width of latent space grid
        if topology == :cylinder
            Threads.@threads for j ∈ axes(Δ², 2)
                for i ∈ axes(Δ², 1)
                    Δx = abs(Ξ[i,1] - M[j,1])
                    Δy = abs(Ξ[i,2] - M[j,2])
                    Δ²[i,j] = min(Δx, 2-Δx)^2 + Δy^2  # only wrap in x axis
                end
            end
        elseif topology == :torus
            Threads.@threads for j ∈ axes(Δ², 2)
                for i ∈ axes(Δ², 1)
                    Δx = abs(Ξ[i,1] - M[j,1])
                    Δy = abs(Ξ[i,2] - M[j,2])
                    Δ²[i,j] = min(Δx, 2-Δx)^2 + min(Δy, 2-Δy)^2  # wrap both axes
                end
            end
        else
            pairwise!(sqeuclidean, Δ², Ξ, M, dims=1)
        end


        Φ[:, 1:end-1] .= exp.(-Δ² ./ (2*σ^2))
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

    # add the means back to each row since PCA uses
    # a mean-subtracted covariance matrix
    for i ∈ axes(Ψ,1)
        Ψ[i,:] .= Ψ[i,:] .+ mean(X[:,i])
    end

    β⁻¹ = max(pca_vars[3], mean(pairwise(sqeuclidean, Ψ, dims=2))/2)

    # 11. return final GTM object
    return GTMBase(Ξ, M, Φ, W, Ψ, zeros(n_nodes, n_records), (1/n_nodes) .* ones(n_nodes, n_records), β⁻¹)
end


function fit!(gtm, X; α = 0.1, nepochs=100, tol=1e-3, nconverged=5, verbose=false)

    # get the needed dimensions
    N,D = size(X)
    K = size(gtm.Ξ,1)
    M = size(gtm.M,1) + 1  # don't forget the bias term!

    # set up the prefactor
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
        # EXPECTATION
        mul!(gtm.Ψ, gtm.W, gtm.Φ')                     # update latent node means
        pairwise!(sqeuclidean, gtm.Δ², gtm.Ψ, X', dims=2)  # update distance matrix
        gtm.Δ² .*= -(1/(2*gtm.β⁻¹))

        softmax!(gtm.R, gtm.Δ², dims=1)

        mul!(GΦ, diagm(sum(gtm.R, dims=2)[:]), gtm.Φ)      # update the G matrix diagonal
        mul!(RX, gtm.R, X)                                 # update intermediate for R.H.S

        # UPDATE LOG-LIKELIHOOD
        log_prefac = (D/2)*log(1/(2* gtm.β⁻¹* π)) + log(1/K)

        if i == 1
            l = max(sum(log_prefac .+ logsumexp(gtm.Δ², dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)
        else
            llh_prev = l
            l = max(sum(log_prefac .+ logsumexp(gtm.Δ², dims=1)), nextfloat(typemin(1.0)))
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
        mul!(LHS, gtm.Φ', GΦ)                              # update left-hand-side
        if α > 0
            LHS[diagind(LHS)] .+= α * gtm.β⁻¹               # add regularization
        end
        mul!(RHS, gtm.Φ', RX)                              # update right-hand-side

        gtm.W = (LHS\RHS)'                                 # update weights

        mul!(gtm.Ψ, gtm.W, gtm.Φ')                         # update means
        pairwise!(sqeuclidean, gtm.Δ², gtm.Ψ, X', dims=2)  # update distance matrix
        gtm.β⁻¹ = sum(gtm.R .* gtm.Δ²)/(N*D)                    # update variance

        if verbose
            println("iter: $(i), log-likelihood = $(l)")
        end
    end

    AIC = 2*length(gtm.W) - 2*llhs[end]
    BIC = log(size(X,1))*length(gtm.W) - 2*llhs[end]

    return converged, llhs, AIC, BIC
end


function get_batches(batchsize, N)
    idxs = [i for i in range(1, step=batchsize, stop=N)]
    if idxs[end] < N
        push!(idxs, N)
    end
    idx_batches = [idxs[i]:idxs[i+1] for i ∈ 1:length(idxs)-1]
end




# TODO: Fix me!
function fit_incremental!(gtm, X; α = 0.1, nepochs=100, batchsize=32, tol=1e-3, nconverged=5, verbose=false)

    # get the needed dimensions
    N,D = size(X)
    K = size(gtm.Ξ,1)
    M = size(gtm.M,1) + 1  # don't forget the bias term!

    # set up the prefactor
    prefac = 0.0

    #GΦ = diagm(sum(gtm.R, dims=2)[:]) * gtm.Φ
    G = diagm(sum(gtm.R, dims=2)[:])
    RX = gtm.R*X
    LHS = zeros(M,M)
    RHS = zeros(M,D)

    l = 0.0
    llh_prev = 0.0
    llhs = Float64[]
    nclose = 0
    converged = false

    idx_batches = get_batches(batchsize, N)

    for i in 1:nepochs
        @showprogress for idx_batch in idx_batches
            Nbatch = length(idx_batch)

            # set up array views
            Xb = @view X[idx_batch, :]
            Δ²_b = @view gtm.Δ²[:, idx_batch]
            Rb_old = gtm.R[:, idx_batch]
            Rb = @view gtm.R[:, idx_batch]

            # expectation step
            mul!(gtm.Ψ, gtm.W, gtm.Φ')

            pairwise!(sqeuclidean, Δ²_b, gtm.Ψ, Xb', dims=2)
            softmax!(Rb, -(1/(2*gtm.β⁻¹)) .* Δ²_b, dims=1)

            #mul!(GΦ, diagm(sum(gtm.R, dims=2)[:]), gtm.Φ)      # update the G matrix diagonal
            #mul!(RX, gtm.R, X)                                 # update intermediate for R.H.S

            G[diagind(G)] += sum(Rb - Rb_old, dims=2)
            RX +=  (Rb .- Rb_old)*Xb                            # update intermediate for R.H.S


            # MAXIMIZATION
            #mul!(LHS, gtm.Φ', GΦ)                              # update left-hand-side
            mul!(LHS, gtm.Φ', G*gtm.Φ)                              # update left-hand-side
            if α > 0
                LHS[diagind(LHS)] .+= α * gtm.β⁻¹               # add regularization
            end

            mul!(RHS, gtm.Φ', RX)                              # update right-hand-side

            gtm.W = (LHS\RHS)'

            Δ²_b_old = copy(Δ²_b)
            mul!(gtm.Ψ, gtm.W, gtm.Φ')
            pairwise!(sqeuclidean, Δ²_b, gtm.Ψ, Xb', dims=2)

            gtm.β⁻¹ = sum(gtm.R .* gtm.Δ²)/(N*D)                    # update variance

            # gtm.β⁻¹ = gtm.β⁻¹ + sum(Rb .* Δ²_b)/(N*D) - sum(Rb_old .* Δ²_b_old)/(N*D)  # update variance

            # gtm.β⁻¹ += sum((Rb .- Rb_old) .* Δ²_b)/(Nbatch*D)
        end

        # UPDATE LOG-LIKELIHOOD
        log_prefac = log((1/(2* gtm.β⁻¹* π))^(D/2) * (1/K))
        if i == 1
            l = max(sum(log_prefac .+ logsumexp(-(1/(2*gtm.β⁻¹)) .* gtm.Δ², dims=1)), nextfloat(typemin(1.0)))
            push!(llhs, l)
        else
            llh_prev = l
            l = max(sum(log_prefac .+ logsumexp(-(1/(2*gtm.β⁻¹)) .* gtm.Δ², dims=1)), nextfloat(typemin(1.0)))
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

        if verbose
            println("iter: $(i), log-likelihood = $(l)")
        end
    end

    AIC = 2*length(gtm.W) - 2*llhs[end]
    BIC = log(size(X,1))*length(gtm.W) - 2*llhs[end]

    return converged, llhs, AIC, BIC
end



function DataMeans(gtm, X)
    mul!(gtm.Ψ, gtm.W, gtm.Φ')
    Δ² = pairwise(sqeuclidean, gtm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gtm.β⁻¹))
    R = softmax(Δ², dims=1)

    return R'*gtm.Ξ
end


function DataModes(gtm, X)
    mul!(gtm.Ψ, gtm.W, gtm.Φ')
    Δ² = pairwise(sqeuclidean, gtm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gtm.β⁻¹))
    R = softmax(Δ², dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return gtm.Ξ[idx,:]
end


function class_labels(gtm, X)
    mul!(gtm.Ψ, gtm.W, gtm.Φ')
    Δ² = pairwise(sqeuclidean, gtm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gtm.β⁻¹))
    R = softmax(Δ², dims=1)

    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return idx
end


function responsibility(gtm, X)
    mul!(gtm.Ψ, gtm.W, gtm.Φ')
    Δ² = pairwise(sqeuclidean, gtm.Ψ, X', dims=2)
    Δ² .*= -(1/(2*gtm.β⁻¹))
    R = softmax(Δ², dims=1)

    return R'
end


