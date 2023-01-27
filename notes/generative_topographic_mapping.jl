using Pkg
Pkg.activate(".")

using Plots, LinearAlgebra, Statistics, MLJ
using Distances, MultivariateStats
using RDatasets
using Distributions


function getCoordsMatrix(k)
    # return size: (k^2, 2)

    x = range(-1.0, 1.0, length=k)
    xs = vcat([x[i] for i ∈ 1:length(x), j ∈ 1:length(x)]...)
    ys = vcat([x[j] for i ∈ 1:length(x), j∈ 1:length(x)]...)
    X = hcat(xs, ys)  # X[:,1] are the x positions, X[:,2] are the y positions
end

function getΦMatrix(X, M, σ²)
    Φ = zeros(n_nodes, n_rbf_centers + 1)
    Φdistances = pairwise(sqeuclidean, X, M, dims=1)

    Φ[:, 1:end-1] .= exp.(-1.0 .* Φdistances ./ (2σ²))
    Φ[:, end] .= 1.0

    # set the last column to ones to allow for a bias term
    Φ[:,end] .= 1
    return Φ
end

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

function initWMatrix(X,Φ,U)
    #return size: (n_features, n_rbf_centers+1)

    # We want to find W such that WΦ' = UX'
    # therefore, W' is the solution to Φ'⋅Φ⋅W' = Φ'UX'
    return ((Φ'*Φ)\(Φ'*X*U'))'
end


# NOTE: need to figure out why we are adding means here...
function initYMatrix(Dataset, W, Φ)
    # return size: (n_features, n_nodes)
    data_means = mean(Dataset, dims=1)
    means_matrix = zeros(size(W,1), size(Φ,1))
    for j ∈axes(means_matrix,2), i∈axes(means_matrix,1)
        means_matrix[i,j] = data_means[i]
    end

    Y = W*Φ'
    Y = Y + means_matrix  # <-- is this necessary
    return Y
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



function getGMatrix(R)
    # return size: (n_nodes, n_nodes)
    # G is determined by the sum over data points at each node
    Σs = vec(sum(R, dims=2))
    return diagm(Σs)
end

function updateW(R, Φ, G, 𝒟, β⁻¹; α=0.0)
    LHS = Φ'*G*Φ
    if α > 0
        LHS .= LHS +  α*β⁻¹*I
    end

    # now we have (LHS)W =  Φ'R𝒟
    W = (LHS\(Φ'*R*𝒟))'
    return W
    # let's just do it their way:
    #return (inv(LHS)*Φ'*R*𝒟)'
end

function updateBeta(R, D)
    ND = size(R,1)*size(R,2)
    #NOTE: this is element wise multiplication
    return sum(R .* D)/ND
end


function estimateLogLikelihood(P, β⁻¹, 𝒟)
    n_nodes = size(P,1)
    n_datapoints = size(P, 2)
    n_dimensions = size(𝒟, 2)  # number of columns for each data record

    prior = 1.0/n_nodes
    # now we need the exponential prefactor that we skipped before when calculating P matrix
    prefactor = (1/(2*π*β⁻¹))^(n_dimensions/2)

    loglikelihood = sum(log.(prior * prefactor * sum(P, dims=1)))./n_datapoints  # we want to maximize log-likelihood or minimize -log-likelihood
    return loglikelihood
end


function get_means(R, X)
    return R' * X
end

function get_modes(R, X)
    idx = argmax(R, dims=1)
    idx = [idx[i][1] for i ∈ 1:length(idx)]
    return X[idx, :]
end



# 0. load dataset
iris = dataset("datasets","iris")
Dataset =Matrix(iris[:, 1:4])
classes = iris[:,5]

# 1. Initialize GTM Parameters
k = 16  # there are K=k² total latent nodes
m = 4  # there are M=m² basis functions (centers of our RBFs)
 σ = 0.3 # the σ² is the variance in our RBFs
α=0.1 # reglarization parameter
tol = 0.0001 #  stop fitting if difference in log-likelihood is less than this amount
verbose = true

n_features = size(Dataset, 2)
n_datapoints = size(Dataset, 1)
n_nodes = k*k
n_rbf_centers = m*m


# 2. create node matrix `X`
X = getCoordsMatrix(k)
@assert size(X) == (n_nodes, 2)

# 3. create RBF centers matrix `M`
M = getCoordsMatrix(m)
@assert size(M) == (n_rbf_centers, 2)

# 4. Initialize RBF variance `σ`
mdist = pairwise(sqeuclidean, M, M, dims=1)
minimum(mdist[mdist .> 0.0])
σ² = σ * minimum(mdist[mdist .> 0.0])
# --------------------------------------------------------------------------------
# visualize the latent space
plot1 = plot(
    xlabel="x₁",
    ylabel="x₂",
    title="Latent Space",
)

xs = range(-1.0, 1.0, length=100)
ys = xs
zs = zeros(size(xs,1), size(xs,1))

for k ∈ 1:n_rbf_centers
    mypdf = MvNormal(M[k,:], σ²)
    for j ∈ 1:size(xs,1), i ∈ 1:size(xs,1)
        zs[i,j] += pdf(mypdf, [xs[i], ys[j]])
    end
end

heatmap!(plot1, xs, ys, zs, color=:thermal, colorbar=false)

scatter!(plot1,
         X[:,1], X[:,2],
         color=:white,
         ms=2,
         msw=0,
         label="Node Points"
         )

display(plot1)

# --------------------------------------------------------------------------------


# 5. Create RBF matrix `Φ`
Φ = getΦMatrix(X,M,σ²)  # shape (n_nodes, n_rbf_centers+1)
@assert size(Φ) == (n_nodes, n_rbf_centers+1)


# 6. Perform PCA on Data
# 7. Set U to first two columns of data covariance matrix
U,β⁻¹ = getUMatrix(Dataset)
@assert size(U) == (n_features, 2)

# 8. Initialize parameter matrix `W`
W = initWMatrix(X, Φ, U)
@assert size(W) == (n_features, n_rbf_centers+1)


# 9. Initialize projection manifold `Y`
Y = initYMatrix(Dataset, W, Φ)
@assert size(Y) == (n_features, n_nodes)

# 10. Initialize covariance `β⁻¹`
β⁻¹ = maximum([mean(pairwise(sqeuclidean, Y, dims=2))/2, β⁻¹])

# let's see how reasonable this is
1.0/(β⁻¹)  # variance of mapped distribution in data space

# 11. Create distance matrix `D`
D = pairwise(sqeuclidean, Y, Dataset')
@assert size(D) == (n_nodes, n_datapoints)


# --------------------------------------------------------------------------
# visualize results of initializations
# 1. update data distribution
Pmat = Posterior(β⁻¹, D)
@assert size(Pmat) == (n_nodes, n_datapoints)

# 2. compute responsabilities
R = Responsabilities(Pmat)
@assert size(R) == (n_nodes, n_datapoints)
@assert all(colsum ≈ 1 for colsum ∈ sum(R, dims=1))

Rmeans = get_means(R, X);
Rmodes = get_modes(R, X);



p1 = scatter(Dataset[:,1],Dataset[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Original Data")
p2 = scatter(Rmeans[:,1], Rmeans[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Mean node responsability")
p3 = scatter(Rmodes[:,1], Rmodes[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Mode node responsability")

p_prefit = plot(p1, p2, p3, layout=(1,3), size=(1200, 450), plot_title="PCA Initialization") 
savefig(p_prefit, "gtm_pre_fit.png")
savefig(p_prefit, "gtm_pre_fit.svg")
savefig(p_prefit, "gtm_pre_fit.pdf")

# --------------------------------------------------------------------------




# 12. Repeat unitl convergence
niter = 200  # default maximum number of iterations
i = 1
diff = 1000
converged =0
minusℓ = 1000
minusℓ_prev = 1000
while i < (niter) && converged < 4

    # 1. update data distribution
    Pmat = Posterior(β⁻¹, D)
    @assert size(Pmat) == (n_nodes, n_datapoints)

    # 2. compute responsabilities
    R = Responsabilities(Pmat)
    @assert size(R) == (n_nodes, n_datapoints)
    @assert all(colsum ≈ 1 for colsum ∈ sum(R, dims=1))


    # 3. Update diagonal matrix `G`
    G = getGMatrix(R)
    @assert size(G) == (n_nodes, n_nodes)


    # 4. Update parameter matrix `W`
    W = updateW(R, Φ, G, Dataset, β⁻¹; α=α)
    @assert size(W) == (n_features, n_rbf_centers+1)

    # 5. Update manifold matrix `Y`
    Y = W*Φ'
    @assert size(Y) == (n_features, n_nodes)

    # 6. Update distance matrix `D`
    D = pairwise(sqeuclidean, Y, Dataset')
    @assert size(D) == (n_nodes, n_datapoints)

    # 7. Update `β⁻¹`
    β⁻¹ = updateBeta(R, D)

    # 8. Estimate log-likelihood and check for convergence
    if i == 1
        minusℓ = -estimateLogLikelihood(Pmat, β⁻¹, Dataset)
    else
        minusℓ_prev = minusℓ
        minusℓ = -estimateLogLikelihood(Pmat, β⁻¹, Dataset)

        diff = abs(minusℓ_prev - minusℓ)
    end

    # we need to have 4 consecutaive updates with diff at or below the tolerance to exit
    if diff <= tol
        converged += 1
    else
        converged = 0
    end

    if verbose
        println("Iter: ", i, "  ℓ: ", -minusℓ)
    end


    i += 1
end


Pmat = Posterior(β⁻¹, D);
R = Responsabilities(Pmat);

# 13. Compute mean node responsability
Rmeans = get_means(R, X);
# 14. compute mode node responsability
Rmodes = get_modes(R, X);

# for elem ∈ sum(R, dims=1)
#     if !(elem ≈ 1.0)
#         println(elem)
#     end
# end


p1 = scatter(Dataset[:,1],Dataset[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Original Data")
p2 = scatter(Rmeans[:,1], Rmeans[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Mean node responsability")
p3 = scatter(Rmodes[:,1], Rmodes[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Mode node responsability")

p_postfit = plot(p1, p2, p3, layout=(1,3), size=(1200, 450), plot_title="GTM Fit")
savefig(p_postfit, "gtm_post_fit.png")
savefig(p_postfit, "gtm_post_fit.svg")
savefig(p_postfit, "gtm_post_fit.pdf")

