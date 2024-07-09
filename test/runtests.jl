using GenerativeTopographicMapping
using Test
using MLJBase
using StableRNGs  # should add this for later
using MLJTestIntegration
using Tables
using Distances
using LinearAlgebra

rng = StableRNG(1234)



@testset "Initialization methods" begin

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    Data_X, Data_y = make_blobs(100, 10;centers=5, rng=rng);

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
    Data_X, Data_y = make_blobs(100, 10;centers=5, rng=rng);
    X = Tables.matrix(Data_X)

    k = 16  # there are K=k² total latent nodes
    m = 4  # there are M=m² basis functions (centers of our RBFs)
    s = 0.3 # the σ² is the variance in our RBFs

    n_features = size(X, 2)
    n_datapoints = size(X, 1)
    n_nodes = k*k
    n_rbf_centers = m*m


    gtm = GenerativeTopographicMapping.GTMBase(k,m,s,X)
    Φ_square = gtm.Φ

    converged, llhs, AIC, BIC = GenerativeTopographicMapping.fit!(gtm, X; tol=0.1, nconverged=4)
    @test converged == true
    @test size(gtm.R) == (n_nodes, n_datapoints)
    @test all(colsum ≈ 1 for colsum ∈ sum(gtm.R, dims=1))


    # test out other topologies
    gtm = GenerativeTopographicMapping.GTMBase(k,m,s,X; topology=:cylinder)
    Φ_cylinder = gtm.Φ
    @test Φ_square != Φ_cylinder

    gtm = GenerativeTopographicMapping.GTMBase(k,m,s,X; topology=:torus)
    Φ_torus = gtm.Φ
    @test Φ_square != Φ_torus
    @test Φ_cylinder!= Φ_torus
end


@testset "GTM MLJ Interface" begin
    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X, y = make_blobs(100, 10;centers=5, rng=rng);

    model = GTM(rng=rng)
    m = machine(model, X)

    fit!(m, verbosity=0)

    X̃ = MLJBase.transform(m, X)
    @test size(matrix(X̃)) == (100,2)

    classes = MLJBase.predict(m, X)
    @test length(classes) == 100

    Resp = predict_responsibility(m, X)

    fp = fitted_params(m)
    @test Set([:gtm]) == Set(keys(fp))

    rpt = report(m)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Ξ, :llhs, :converged, :AIC, :BIC]) == Set(keys(rpt))
end


@testset "gsm-linear-base.jl" begin
    Nₑ = 5  # nodes per edge
    Nᵥ = 3  # number of vertices
    D = Nᵥ - 1
    Λ = gtm = GenerativeTopographicMapping.get_barycentric_grid_coords(Nₑ, Nᵥ)
    @test size(Λ, 2) == binomial(Nₑ + D - 1, D)

    # generate synthetic data
    X = rand(rng, 100, 10)  # test data must be non-negative

    k = 10
    Nᵥ = 3
    gsm = GenerativeTopographicMapping.GSMLinearBase(k, Nᵥ, X)

    Z = gsm.Z
    @test size(Z,1) == binomial(k + Nᵥ - 2, Nᵥ -1)
end


@testset "gsm-linear-mlj.jl" begin
    Nₑ = 5  # nodes per edge
    Nᵥ = 3  # number of vertices
    D = Nᵥ - 1


    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X = Tables.table(rand(rng, 100,10))
    N2 = 15
    X2 = Tables.table(rand(rng, N2, 10))

    model = GSMLinear(k=Nₑ, Nv=Nᵥ, nepochs=100, rng=rng)
    m = machine(model, X)
    fit!(m, verbosity=0)

    X̃ = MLJBase.transform(m, X2)
    @test size(matrix(X̃)) == (N2, Nᵥ)

    classes = MLJBase.predict(m, X2)
    @test length(classes) == N2

    Resp = predict_responsibility(m, X2)
    @test all(isapprox.(sum(Resp, dims=2), 1.0))

    fp = fitted_params(m)
    @test Set([:gsm]) == Set(keys(fp))

    rpt = report(m)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Z, :Q, :llhs, :converged, :AIC, :BIC, :idx_vertices]) == Set(keys(rpt))

    @test size(rpt[:Φ], 2) == Nᵥ

    # attempt the data reconstruction
    X_gsm = data_reconstruction(m, X2)
    @test size(X_gsm) == (N2, 10)

    @test all(rpt[:W] .≥ 0.0)
end


@testset "gsm-nonlinear-base.jl" begin
    # generate synthetic data
    X = rand(rng, 100, 10)  # test data must be non-negative

    k = 10
    m = 5
    Nᵥ = 3
    s = 1

    gsm = GenerativeTopographicMapping.GSMNonlinearBase(k, m, s, Nᵥ, X)

    Z = gsm.Z
    @test size(Z,1) == binomial(k + Nᵥ - 2, Nᵥ -1)
end


@testset "gsm-nonlinear-mlj.jl" begin
    Nₑ = 5  # nodes per edge
    Nᵥ = 3  # number of vertices
    D = Nᵥ - 1
    m = 5

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X = Tables.table(rand(rng, 100,10))
    N2 = 15
    X2 = Tables.table(rand(rng, N2, 10))

    model = GSMNonlinear(k=Nₑ, m=m, Nv=Nᵥ, nepochs=100, rng=rng)
    mach = machine(model, X)
    fit!(mach, verbosity=0)

    X̃ = MLJBase.transform(mach, X2)
    @test size(matrix(X̃)) == (N2, Nᵥ)

    classes = MLJBase.predict(mach, X2)
    @test length(classes) == N2

    Resp = predict_responsibility(mach, X2)
    @test all(isapprox.(sum(Resp, dims=2), 1.0))

    fp = fitted_params(mach)
    @test Set([:gsm]) == Set(keys(fp))

    rpt = report(mach)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Z, :Q, :llhs, :converged, :AIC, :BIC, :idx_vertices]) == Set(keys(rpt))

    # attempt the data reconstruction
    X_gsm = data_reconstruction(mach, X2)
    @test size(X_gsm) == (N2, 10)


    @test all(rpt[:W] .≥ 0.0)
end


@testset "gsm-combo-mlj.jl" begin
    k = 10
    m = 5
    Nv = 3

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X = Tables.table(rand(rng, 100,10))
    N2 = 15
    X2 = Tables.table(rand(rng, N2, 10))

    model = GSMCombo(k=k, m=m, Nv=Nv, nepochs=100, rng=rng, zero_init=false)
    mach = machine(model, X)
    fit!(mach, verbosity=0)

    X̃ = MLJBase.transform(mach, X2)
    @test size(matrix(X̃)) == (N2, Nv)

    classes = MLJBase.predict(mach, X2)
    @test length(classes) == N2

    Resp = predict_responsibility(mach, X2)
    @test all(isapprox.(sum(Resp, dims=2), 1.0))

    fp = fitted_params(mach)
    @test Set([:gsm]) == Set(keys(fp))

    rpt = report(mach)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Z, :Q, :llhs, :converged, :AIC, :BIC, :idx_vertices]) == Set(keys(rpt))

    # attempt the data reconstruction
    X_gsm = data_reconstruction(mach, X2)
    @test size(X_gsm) == (N2, 10)


    @test all(rpt[:W] .≥ 0.0)

end



@testset "gsm-big-linear-mlj.jl" begin
    n_nodes = 100
    Nv = 3

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X = Tables.table(rand(rng, 100,10))
    N2 = 15
    X2 = Tables.table(rand(rng, N2, 10))


    model = GSMBigLinear(n_nodes=n_nodes, Nv=Nv, nepochs=100, rng=rng)
    m = machine(model, X)
    fit!(m, verbosity=0)

    X̃ = MLJBase.transform(m, X2)
    @test size(matrix(X̃)) == (N2, Nv)

    classes = MLJBase.predict(m, X2)
    @test length(classes) == N2

    Resp = predict_responsibility(m, X2)
    @test all(isapprox.(sum(Resp, dims=2), 1.0))

    fp = fitted_params(m)
    @test Set([:gsm]) == Set(keys(fp))

    rpt = report(m)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Z, :Q, :llhs, :converged, :AIC, :BIC, :idx_vertices]) == Set(keys(rpt))

    # attempt the data reconstruction
    X_gsm = data_reconstruction(m, X2)
    @test size(X_gsm) == (N2, 10)

    @test all(rpt[:W] .≥ 0.0)
end




@testset "gsm-big-nonlinear-mlj.jl" begin
    n_nodes = 100
    n_rbfs = 20
    Nv = 3

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X = Tables.table(rand(rng, 100,10))
    N2 = 15
    X2 = Tables.table(rand(rng, N2, 10))


    model = GSMBigNonlinear(n_nodes=n_nodes, n_rbfs=n_rbfs, Nv=Nv, nepochs=100, rng=rng)
    m = machine(model, X)
    fit!(m, verbosity=0)

    X̃ = MLJBase.transform(m, X2)
    @test size(matrix(X̃)) == (N2, Nv)

    classes = MLJBase.predict(m, X2)
    @test length(classes) == N2

    Resp = predict_responsibility(m, X2)
    @test all(isapprox.(sum(Resp, dims=2), 1.0))

    fp = fitted_params(m)
    @test Set([:gsm]) == Set(keys(fp))

    rpt = report(m)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Z, :Q, :llhs, :converged, :AIC, :BIC, :idx_vertices]) == Set(keys(rpt))

    # attempt the data reconstruction
    X_gsm = data_reconstruction(m, X2)
    @test size(X_gsm) == (N2, 10)

    @test all(rpt[:W] .≥ 0.0)
end




@testset "gsm-big-combo-mlj.jl" begin
    k = 10
    m = 5
    Nv = 3

    # generate synthetic dataset for testing with 100 data points, 10 features, and 5 classes
    X = Tables.table(rand(rng, 100,10))
    N2 = 15
    X2 = Tables.table(rand(rng, N2, 10))

    model = GSMBigCombo(n_nodes=k, n_rbfs=m, Nv=Nv, nepochs=100, rng=rng, zero_init=false)

    mach = machine(model, X)
    fit!(mach, verbosity=0)

    X̃ = MLJBase.transform(mach, X2)
    @test size(matrix(X̃)) == (N2, Nv)

    classes = MLJBase.predict(mach, X2)
    @test length(classes) == N2

    Resp = predict_responsibility(mach, X2)
    @test all(isapprox.(sum(Resp, dims=2), 1.0))

    fp = fitted_params(mach)
    @test Set([:gsm]) == Set(keys(fp))

    rpt = report(mach)
    @test Set([:W, :β⁻¹, :Φ, :node_data_means, :Z, :Q, :llhs, :converged, :AIC, :BIC, :idx_vertices]) == Set(keys(rpt))

    # attempt the data reconstruction
    X_gsm = data_reconstruction(mach, X2)
    @test size(X_gsm) == (N2, 10)


    @test all(rpt[:W] .≥ 0.0)
end



