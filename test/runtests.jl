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

    X = Tables.matrix(Data_X)

    k = 16
    m = 4
    s = 0.3
    α=0.1
    tol = 0.0001

    n_features = size(X, 2)
    n_datapoints = size(X, 1)
    n_nodes = k*k
    n_rbf_centers = m*m


    gtm = GenerativeTopographicMapping.GTMBase(k,m,s,X)

    Ξ = gtm.Ξ
    @test size(Ξ,1) == k*k
    @test Ξ[1,1] == -1.0
    @test Ξ[1, end] == -1.0
    @test Ξ[end, 1] == 1.0
    @test Ξ[end,end] == 1.0

    M = gtm.M

    β⁻¹ = gtm.β⁻¹
    @test β⁻¹ > 0

    Φ = gtm.Φ
    @test size(Φ) == (n_nodes, n_rbf_centers + 1)
    @test all(Φ[:,end] .== 1.0)

    # make sure parameters are still mutable
    gtm.W[1,1] = 3.14
    @test gtm.W[1,1] == 3.14
    gtm.β⁻¹ = 0.42
    @test gtm.β⁻¹ == 0.42
end




@testset "Fitting Methods" begin
    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    Data_X, Data_y = make_blobs(100, 10;centers=5, rng=stable_rng());
    X = Tables.matrix(Data_X)

    k = 16  # there are K=k² total latent nodes
    m = 4  # there are M=m² basis functions (centers of our RBFs)
    s = 0.3 # the σ² is the variance in our RBFs

    n_features = size(X, 2)
    n_datapoints = size(X, 1)
    n_nodes = k*k
    n_rbf_centers = m*m


    gtm = GenerativeTopographicMapping.GTMBase(k,m,s,X)
    converged, R, llhs, AIC, BIC, latent_means = GenerativeTopographicMapping.fit!(gtm, X; tol=0.1, nconverged=4)
    @assert converged == true
    @test size(R) == (n_nodes, n_datapoints)
    @test all(colsum ≈ 1 for colsum ∈ sum(R, dims=1))
end


@testset "MLJ Interface" begin
    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X, y = make_blobs(100, 10;centers=5, rng=stable_rng());

    model = GTM()
    m = machine(model, X)
    fit!(m, verbosity=0)

    X̃ = MLJBase.transform(m, X)
    @assert size(matrix(X̃)) == (100,2)

    classes = MLJBase.predict(m, X)
    @assert length(y) == 100


    fp = fitted_params(m)
    @test Set([:gtm]) == Set(keys(fp))

    rpt = report(m)
    @test Set([:W, :β⁻¹, :Φ, :Ξ, :R, :llhs, :converged, :AIC, :BIC, :latent_means]) == Set(keys(rpt))
end

