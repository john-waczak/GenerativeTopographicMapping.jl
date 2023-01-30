using LinearAlgebra
using Statistics
using Distances
using MultivariateStats



# ----------------------------------------------
#     Initialization functions
# ----------------------------------------------



"""
    getCoordsMatrix(k::Int)

Generate a matrix of `k²` node coordinates on a regular grid with ``x∈ [-1,1] `` and ``y∈[-1, 1]``. 
"""
function getCoordsMatrix(k::Int)
    # return size: (k^2, 2)

    x = range(-1.0, 1.0, length=k)
    xs = vcat([x[i] for i ∈ 1:length(x), j ∈ 1:length(x)]...)
    ys = vcat([x[j] for i ∈ 1:length(x), j∈ 1:length(x)]...)
    X = hcat(xs, ys)  # X[:,1] are the x positions, X[:,2] are the y positions
end



"""
    initializeVariance(σ::Float64, M::)

Initilize RBF variance by combining supplied standard deviation with minimum RBF mean distance.
"""
function initializeVariance(σ, M)
    mdist = pairwise(euclidean, M, M, dims=1)
    minimum(mdist[mdist .> 0.0])
    σ² = σ * minimum(mdist[mdist .> 0.0])
    return σ²
end



"""
    getΦMatrix(X, M, σ²)

Given a matrix of latent node coordinates `X`, RBF mean coordinates `M`, and variance `σ²`, return a matrix `Φ` of dimension `(n_nodes, n_rbf_centers+1)`. The final column is set to `1.0` to include a bias offset in addition to the RBFs.
"""
function getΦMatrix(X, M, σ²)
    n_nodes = size(X, 1)
    n_rbf_centers = size(M,1)
    Φ = zeros(n_nodes, n_rbf_centers + 1)
    Φdistances = pairwise(sqeuclidean, X, M, dims=1)

    Φ[:, 1:end-1] .= exp.(-1.0 .* Φdistances ./ (2σ²))
    Φ[:, end] .= 1.0

    # set the last column to ones to allow for a bias term
    Φ[:,end] .= 1
    return Φ
end



"""
    getUMatrix(Dataset)

Perform PCA on the `Dataset` and return a matrix `U` containing the first two principal components (first two columns of data covariance matrix) and the variance of the third principal component. Size of returned matrix `U` is `(n_features, 2)`
"""
function getUMatrix(Dataset)
    pca = fit(PCA, Dataset'; maxoutdim=3)

    pca_vecs = projection(pca)  # this results the princiap component vectors (columns) sorted in order of explained variance
    pca_var = principalvars(pca)

    U = pca_vecs[:,1:2]
    for i ∈ axes(U,2)
        U[:,i] .= sqrt(pca_var[i]).*U[:,i]
    end
    return U, pca_var[3]
end


"""
    initWMatrix(X,Φ,U)

Initialize parameter matrix `W`. Initial weights are chosen so that `WΦ'` reproduces PCA projections such that `WΦ' ≈ UX'`
"""
function initWMatrix(X,Φ,U)
    #return size: (n_features, n_rbf_centers+1)
    # We want to find W such that WΦ' = UX'
    # therefore, W' is the solution to Φ'⋅Φ⋅W' = Φ'UX'
    return ((Φ'*Φ)\(Φ'*X*U'))'
end


"""
    initβ⁻¹(β⁻¹, Y)

Initialized β⁻¹ using our first guess for β⁻¹ (from 3rd principal component variance) and the mean distance between projected rbf centers in data space.
"""
function initβ⁻¹(β⁻¹, Y)
    return maximum([mean(pairwise(sqeuclidean, Y, dims=2))/2, β⁻¹])
end




mutable struct GenerativeTopographicMap{T<:AbstractArray, T2<:AbstractArray,  T3<:AbstractArray, T4<:AbstractArray}
    X::T
    M::T2
    σ²::Float64
    Φ::T3
    W::T4
    β⁻¹::Float64
    α::Float64
    tol::Float64
    niter::Int
    nrepeats::Int
    verbose::Bool
end



"""
    GTM(k, m, σ, Dataset; α=0.0, tol=0.0001, verbose=false)

Initialize hyperparameters for a GTM model.

- `k`: square root of the number of latent nodes
- `m`: square root of the number of RBF centers in latent space
- `σ`: standard deviation for latent space RBF functions
- `Dataset`: dataset to fit GTM model to. Assumed shape is `(n_datapoints, n_features)`
- `α`: Weight regularization parameter (`0.0` means no regularization)
- `tol`: absolute tolerance used during fitting.
- `verbose`: Set to true for extra print statements.
"""
function GenerativeTopographicMap(k, m, σ, Dataset; α=0.1, tol=0.0001, niter=200, nrepeats=4, verbose=false)
    n_features =  size(Dataset, 2)
    n_datapoints = size(Dataset, 1)
    n_nodes = k*k
    n_rbf_centers = m*m

    X = getCoordsMatrix(k)
    M = getCoordsMatrix(m)
    σ² = initializeVariance(σ, M)
    Φ = getΦMatrix(X, M, σ²)
    U,β⁻¹ = getUMatrix(Dataset)
    W = initWMatrix(X, Φ, U)

    Y = W*Φ'

    β⁻¹ = initβ⁻¹(β⁻¹, Y)

    return GenerativeTopographicMap(X, M, σ², Φ, W, β⁻¹, α, tol, niter, nrepeats, verbose)
end



# ----------------------------------------------
#     Fitting functions
# ----------------------------------------------

"""
    getYMatrix(gtm::GenerativeTopographicMap)

Compute Gaussian centers in data space via `Y=W*Φ'`. Return size is `(n_features, n_nodes)`.
"""
function getYMatrix(gtm::GenerativeTopographicMap)
    return gtm.W * gtm.Φ'
end

"""
    getDMatrix(gtm::GenerativeTopographicMap, Dataset)

Compute pairwise distances between projected gaussian centers `Y` and data points in `Dataset`. Resulting size is `(n_nodes, n_datapoints)`.
"""
function getDMatrix(gtm::GenerativeTopographicMap, Dataset)
    # (n_features, n_nodes)  (n_datapoints, n_features)
    return pairwise(sqeuclidean, getYMatrix(gtm), Dataset', dims=2)
end



"""
    Posterior(gtm::GenerativeTopographicMap)

Compute a matrix of contributions to posterior probabilities. This is an intermediate result to facilitate computation of true posterior probabilities given by the responsability matrix `R`. The returned size is (n_nodes, n_datapoints). The [exp-normalize trick](https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/) is used for numerical stability.
"""
function Posterior(gtm::GenerativeTopographicMap, Dataset)
    D = getDMatrix(gtm, Dataset)
    # inner = BigFloat.(- D ./ (2*β⁻¹))
    inner = - D ./ (2*gtm.β⁻¹)
    maxes = maximum(inner, dims=1)

    res = inner
    for j ∈ axes(inner, 2)
        res[:, j] .= exp.(inner[:, j] .- maxes[j])
    end

    return res
end


function Posterior(β⁻¹, D)
    # inner = BigFloat.(- D ./ (2*β⁻¹))
    inner = - D ./ (2*β⁻¹)
    maxes = maximum(inner, dims=1)

    res = inner
    for j ∈ axes(inner, 2)
        res[:, j] .= exp.(inner[:, j] .- maxes[j])
    end

    return res
end


"""
    Responsabilities(gtm::GenerativeTopographicMapping, Dataset)

Compute matrix of responsabilities of each node in `X` to datapoints in `Dataset`. Return matrix is of size `(n_nodes, n_datapoints)`.
"""
function Responsabilities(gtm::GenerativeTopographicMap, Dataset)
    # sum along rows since each column is a new data point
    # here we should use the exp-normalize trick
    P = Posterior(gtm, Dataset)
    Σs = sum(P, dims=1)
    R = P
    for j ∈ axes(R,2)
        R[:,j] .= (R[:,j] ./ Σs[j])
    end
    return R
end


function Responsabilities(P)
    # sum along rows since each column is a new data point
    # here we should use the exp-normalize trick
    Σs = sum(P, dims=1)
    R = P
    for j ∈ axes(R,2)
        R[:,j] .= (R[:,j] ./ Σs[j])
    end
    return R
end






"""
    getGMatrix(R)

Create diagonal matrix `G` from responsability matrix `R`. Return size is `(n_nodes, n_nodes)`.
"""
function getGMatrix(R)
    # return size: (n_nodes, n_nodes)
    # G is determined by the sum over data points at each node
    Σs = vec(sum(R, dims=2))
    return diagm(Σs)
end


"""
    updateW!(gtm::GenerativeTopographicMap, Dataset)

Update model weights using responsability matrix.
"""
function updateW!(gtm::GenerativeTopographicMap, R, Dataset)
    G = getGMatrix(R)
    LHS = gtm.Φ'*G*gtm.Φ
    if gtm.α > 0
        LHS .= LHS +  gtm.α*gtm.β⁻¹*I
    end

    # now we have (LHS)W =  Φ'R𝒟
    gtm.W .= (LHS\(gtm.Φ'*R*Dataset))'
end


"""
    updateBeta(R, D)
"""
function updateBeta!(gtm::GenerativeTopographicMap, R, D)
    ND = size(R,1)*size(R,2)
    #NOTE: this is element wise multiplication
    gtm.β⁻¹ = sum(R .* D)/ND
end


"""
    estimateLogLikelihood(gtm::GenerativeTopographicMap, P, Dataset)

Compute the log-likelihood of obtaining our data provided parameters `W` and `β⁻¹`.
"""
function estimateLogLikelihood(gtm::GenerativeTopographicMap, P, Dataset)
    n_nodes = size(P,1)
    n_datapoints = size(P, 2)
    n_features = size(Dataset, 2)  # number of columns for each data record

    prior = 1.0/n_nodes
    # now we need the exponential prefactor that we skipped before when calculating P matrix
    prefactor = (1/(2*π*gtm.β⁻¹))^(n_features/2)

    loglikelihood = sum(log.(prior * prefactor * sum(P, dims=1)))./n_datapoints  # we want to maximize log-likelihood or minimize -log-likelihood
    return loglikelihood
end


"""
    get_means(R, X)

Compute responsability weighted mean node position: ``⟨x|tⱼ, W, β⟩=Σⱼ Rᵢⱼxᵢ ``
"""
function get_means(R, X)
    return R' * X
end

function get_means(gtm::GenerativeTopographicMap, Dataset)
    R = Responsabilities(gtm, Dataset)
    return R' * gtm.X
end


"""
    get_modes(R, X)

Compute the node corresponding to the mode responsability for each data point.
"""
function get_modes(R, X)
    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return X[idx, :]
end


function get_modes(gtm::GenerativeTopographicMap, Dataset)
    R = Responsabilities(gtm, Dataset)
    X = gtm.X
    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return X[idx, :]
end


function getModeidx(gtm::GenerativeTopographicMap, Dataset)
    R = Responsabilities(gtm, Dataset)
    idx = argmax(R, dims=1)
    return [idx[i][1] for i ∈ 1:length(idx)]
end



"""
    fit!(gtm::GenerativeTopographicMap, Dataset)

Fit an initialized generative topographic map `gtm` to a dataset `Dataset`.
"""
function fit!(gtm::GenerativeTopographicMap, Dataset)
    Y = gtm.W * gtm.Φ'
    D = pairwise(sqeuclidean, Y, Dataset', dims=2)

    i = 1
    diff = 1000
    converged = 0
    minusℓ = 1000
    minusℓ_prev = 1000
    while i < (gtm.niter) && converged < gtm.nrepeats

        # -------------------------------
        # Expectation step
        # -------------------------------

        Pmat = Posterior(gtm.β⁻¹, D)
        R = Responsabilities(Pmat)


        # -------------------------------
        # Maximization step
        # -------------------------------
        updateW!(gtm, R, Dataset)
        Y = gtm.W*gtm.Φ'
        D = pairwise(sqeuclidean, Y, Dataset', dims=2)
        updateBeta!(gtm, R, D)

        if i == 1
            minusℓ = -estimateLogLikelihood(gtm, Pmat, Dataset)
        else
            minusℓ_prev = minusℓ
            minusℓ = -estimateLogLikelihood(gtm, Pmat, Dataset)
            diff = abs(minusℓ_prev - minusℓ)
        end

        # we need to have 4 consecutaive updates with diff at or below the tolerance to exit
        if diff <= gtm.tol
            converged += 1
        else
            converged = 0
        end

        if gtm.verbose
            println("Iter: ", i, "  ℓ: ", -minusℓ)
        end

        i += 1
    end
end
