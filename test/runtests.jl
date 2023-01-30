using GenerativeTopographicMapping
using Test
using MLJBase
using StableRNGs  # should add this for later
using MLJTestIntegration
using Tables
using Distances

stable_rng() = StableRNGs.StableRNG(1234)

@testset "Initialization methods" begin

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    Data_X, Data_y = make_blobs(100, 10;centers=5, rng=stable_rng());
    Dataset = Tables.matrix(Data_X)

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

    X = GenerativeTopographicMapping.getCoordsMatrix(k)
    @test size(X,1) == k*k
    @test X[1,1] == -1.0
    @test X[1, end] == -1.0
    @test X[end, 1] == 1.0
    @test X[end,end] == 1.0

    M = GenerativeTopographicMapping.getCoordsMatrix(m)

    σ² = GenerativeTopographicMapping.initializeVariance(σ, M)
    @test σ² > 0

    Φ = GenerativeTopographicMapping.getΦMatrix(X, M, σ²)
    @test size(Φ) == (n_nodes, n_rbf_centers + 1)
    @test all(Φ[:,end] .== 1.0)

    U,β⁻¹ = GenerativeTopographicMapping.getUMatrix(Dataset)
    @test size(U) == (n_features, 2)

    W = GenerativeTopographicMapping.initWMatrix(X, Φ, U)
    @test size(W) == (n_features, n_rbf_centers+1)

    # Y = GenerativeTopographicMapping.initYMatrix(Dataset, W, Φ)
    # @test size(Y) == (n_features, n_nodes)
    Y = W*Φ'

    β⁻¹ = GenerativeTopographicMapping.initβ⁻¹(β⁻¹, Y)

    # compute distances
    D = pairwise(sqeuclidean, Y, Dataset', dims=2)
    @assert size(D) == (n_nodes, n_datapoints)


    gtm = GenerativeTopographicMapping.GenerativeTopographicMap(k, m, σ, Dataset)
    # make sure parameters are writable
    gtm.W[1,1] = 3.14
    @test gtm.W[1,1] == 3.14
    gtm.β⁻¹ = 0.42
    @test gtm.β⁻¹ == 0.42
end

@testset "Fitting Methods" begin
    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    Data_X, Data_y = make_blobs(100, 10;centers=5, rng=stable_rng());
    Dataset = Tables.matrix(Data_X)


    k = 16  # there are K=k² total latent nodes
    m = 4  # there are M=m² basis functions (centers of our RBFs)
    σ = 0.3 # the σ² is the variance in our RBFs

    n_features = size(Dataset, 2)
    n_datapoints = size(Dataset, 1)
    n_nodes = k*k
    n_rbf_centers = m*m

    gtm = GenerativeTopographicMapping.GenerativeTopographicMap(k, m, σ, Dataset; verbose=true)

    Y = GenerativeTopographicMapping.getYMatrix(gtm)
    @test size(Y) == (n_features, n_nodes)

    D = GenerativeTopographicMapping.getDMatrix(gtm, Dataset)
    @test size(D) == (n_nodes, n_datapoints)

    P = GenerativeTopographicMapping.Posterior(gtm, Dataset)
    @test size(P) == (n_nodes, n_datapoints)

    R = GenerativeTopographicMapping.Responsabilities(gtm, Dataset)
    @test size(P) == (n_nodes, n_datapoints)
    @test all(colsum ≈ 1 for colsum ∈ sum(R, dims=1))
    R2 = GenerativeTopographicMapping.Responsabilities(P)
    @test all(R .== R2)

    G = GenerativeTopographicMapping.getGMatrix(R)
    @test size(G) == (n_nodes, n_nodes)


    Wpre = copy(gtm.W)
    GenerativeTopographicMapping.updateW!(gtm, R, Dataset)
    @test any(Wpre .!= gtm.W)  # at least 1 weight needs to change

    β_pre = copy(gtm.β⁻¹)
    GenerativeTopographicMapping.updateBeta!(gtm, R, D)
    @test β_pre != gtm.β⁻¹

    ℓ = GenerativeTopographicMapping.estimateLogLikelihood(gtm, P, Dataset)
    @test !isnan(ℓ)

    R = GenerativeTopographicMapping.Responsabilities(gtm, Dataset)  # get them again in case they've updated
    Xmean = GenerativeTopographicMapping.get_means(R, gtm.X)
    Xmean2 = GenerativeTopographicMapping.get_means(gtm, Dataset)
    @test all(Xmean .== Xmean2)

    Xmode = GenerativeTopographicMapping.get_modes(R, gtm.X)
    Xmode2 = GenerativeTopographicMapping.get_modes(gtm, Dataset)
    @test all(Xmode .== Xmode2)

    classes = GenerativeTopographicMapping.getModeidx(gtm, Dataset)
    @test length(classes) == size(Dataset,1)


    GenerativeTopographicMapping.fit!(gtm, Dataset)
end


@testset "GTM with MLJ" begin
    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X, y = make_blobs(100, 10;centers=5, rng=stable_rng());

    model = GTM()
    m = machine(model, X)
    res = fit!(m, verbosity=0)

    X̃ = MLJBase.transform(m, X)
    @test size(X̃) == (100, 2)

    fp = fitted_params(m)
    @test Set([:gtm]) == Set(keys(fp))

    rpt = report(m)
    @test Set([:classes]) == Set(keys(rpt))
    @test size(rpt.classes, 1) == 100


end



# @testset "GTM.jl" begin
#     # Write your tests here.
# end
