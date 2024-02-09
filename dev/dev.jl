using LogExpFunctions
using LinearAlgebra
using Statistics, MultivariateStats
using Distances
using ProgressMeter
using CSV, DataFrames
using CairoMakie, MintsMakieRecipes
using MLJ
using BenchmarkTools


set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)


include("../src/GenerativeTopographicMapping.jl")
using .GenerativeTopographicMapping


# Demo 1: iris dataset

# load iris datset that we can use to validate the code
df = CSV.read(download("https://ncsa.osn.xsede.org/ees230012-bucket01/mintsML/toy-datasets/iris/iris.csv"), DataFrame)


X = df[:, 1:4]
y = df[:,5]
target_labels = unique(y)
column_labels = uppercasefirst.(replace.(names(df)[1:4], "."=>" "))
y = [findfirst(y[i] .== target_labels) for i in axes(y,1)]

# visualize the dataset
gtm = GTM(k=6, m=2, tol=1e-5, nepochs=100)
mach = machine(gtm, X)
fit!(mach)


res = fitted_params(mach)[:gtm]
rpt = report(mach)
llhs = rpt[:llhs]
Ξ = rpt[:Ξ]

println(rpt[:β⁻¹])


fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 1:length(llhs), llhs, linewidth=5)
fig

save("../figures/iris-llh.png", fig)



means = MLJ.transform(mach, X)
mode_idxs = MLJ.predict(mach, X)
modes = DataModes(res, Matrix(X))

typeof(mach)

Rs = predict_responsibility(mach, X)


pca = fit(PCA, Matrix(X)', maxoutdim=3, pratio=0.99999)
Xpca = MultivariateStats.predict(pca, Matrix(X)')'

fig = Figure();
gl = fig[1,1] = GridLayout();

ax = Axis(
    gl[1,1],
    xlabel=column_labels[1],
    ylabel=column_labels[2],
    title="Irist Dataset"
)

ax2 = Axis(
    gl[1,2],
    xlabel="v₁",
    ylabel="v₂",
    title="PCA",
)


ax3 = Axis(
    gl[2,1],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Means"
)

ax4 = Axis(
    gl[2,2],
    xlabel="ξ₁",
    ylabel="ξ₂",
    title="GTM Modes"
)


idx1 = y .== 1
idx2 = y .== 2
idx3 = y .== 3

sc1 = scatter!(ax, X[idx1,1], X[idx1,2], color=mints_colors[1])
sc2 = scatter!(ax, X[idx2,1], X[idx2,2], color=mints_colors[2])
sc3 = scatter!(ax, X[idx3,1], X[idx3,2], color=mints_colors[3])


scatter!(ax2, Xpca[idx1,1], Xpca[idx1,2], color=mints_colors[1])
scatter!(ax2, Xpca[idx2,1], Xpca[idx2,2], color=mints_colors[2])
scatter!(ax2, Xpca[idx3,1], Xpca[idx3,2], color=mints_colors[3])


scatter!(ax3, means[:ξ₁][idx1], means[:ξ₂][idx1], color=mints_colors[1])
scatter!(ax3, means[:ξ₁][idx2], means[:ξ₂][idx2], color=mints_colors[2])
scatter!(ax3, means[:ξ₁][idx3], means[:ξ₂][idx3], color=mints_colors[3])

scatter!(ax4, modes[idx1,1], modes[idx1,2], color=mints_colors[1])
scatter!(ax4, modes[idx2,1], modes[idx2,2], color=mints_colors[2])
scatter!(ax4, modes[idx3,1], modes[idx3,2], color=mints_colors[3])

leg = Legend(fig[1,2], [sc1, sc2, sc3], target_labels)

fig

save("../figures/iris-gtm-compare.png", fig)




# set up parameter grid
ks = 2:1:20
ms = 2:1:15

bics = zeros(length(ks), length(ms))
aics = zeros(length(ks), length(ms))
Δrec = zeros(length(ks), length(ms))


for k in ks
    println(k)
    for m in ms
        gtm = GTM(k=k, m=m, s=1.0, α=0.1, nepochs=200)
        mach = machine(gtm, X)
        fit!(mach, verbosity=0)
        res = fitted_params(mach)[:gtm]
        rpt = report(mach)

        bics[k-1,m-1] =  rpt[:BIC]
        aics[k-1,m-1] =  rpt[:AIC]

        # compute reconstruction error
        Δrec[k-1, m-1] = sqrt(mean((rpt[:latent_means] .- Matrix(X)).^2))
    end
end

idx_bic = argmin(bics)
k_b = ks[idx_bic[1]]
m_b = ks[idx_bic[2]]

idx_aic = argmin(aics)
k_a = ks[idx_aic[1]]
m_a = ks[idx_aic[2]]

idx_Δ = argmin(Δrec)
k_Δ = ks[idx_Δ[1]]
m_Δ = ks[idx_Δ[2]]


fig = Figure();
ax = Axis(fig[1,1], xlabel="k", ylabel="m", title="Bayesian Information Criterion")
# axr = Axis(fig[1,2], xlabel="k", ylabel="m")
h = heatmap!(ax, ks, ms, bics)
s = scatter!(ax, Point2f(k_b, m_b), marker = :star5, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
axislegend(ax, [s], ["minimum value (k = $(k_b), m = $(m_b))"])
fig

save("../figures/iris-bic.png", fig)



fig = Figure();
ax = Axis(fig[1,1], xlabel="k", ylabel="m", title="Akaike Information Criterion")
h = heatmap!(ax, ks, ms, aics)
s = scatter!(ax, Point2f(k_a, m_a), marker = :star5, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
axislegend(ax, [s], ["minimum value (k = $(k_a), m = $(m_a))"])
fig

save("../figures/iris-aic.png", fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="k", ylabel="m", title="Reconstruction RMSE")
h = heatmap!(ax, ks, ms, Δrec, colorscale=log10)
s = scatter!(ax, Point2f(k_Δ, m_Δ), marker = :star5, markersize=25, color=:white, strokecolor=:gray, strokewidth=1)
axislegend(ax, [s], ["minimum value (k = $(k_Δ), m = $(m_Δ))"])
fig

save("../figures/iris-reconstruction.png", fig)



# let's try it for the mnist dataset
using MLDatasets

trainset = MNIST(:train)

targets = trainset.targets
size(trainset.features)

img_shape = size(trainset.features)[1:2]
X = collect(reshape(trainset.features, 28*28, size(trainset.features,3))')

headers = [Symbol("x_$(i)") for i in 1:28*28]
df = table(X, names=headers)




n_digits = 10

# ---- visualize a number ------------
fig = Figure();
gl = fig[1,1] = GridLayout();

idxs = [(i,j) for i in 1:5 for j in 1:5]

cmap_wb = cgrad([:white, :black])
for idx in 1:length(idxs)
    i,j = idxs[idx]

    ax = Axis(
        gl[i,j],
        aspect=DataAspect(),
        yreversed=true,
        #title="$(targets[idx])",
        #titlesize=50,
    );
    hidedecorations!(ax)
    hidespines!(ax)

    image!(ax, reshape(X[idx,:], img_shape), colormap=cmap_wb)
end

fig

save("../figures/mnist-samples.png", fig)

# ------------------------------------

k = 25
m = 10

gtm = GTM(k=k, m=m, s=0.75, α=0.1, nepochs=30)
mach = machine(gtm, df)
fit!(mach, verbosity=1)




rpt = report(mach)
res = fitted_params(mach)[:gtm]
llhs = rpt[:llhs]

fig = Figure();
ax = Axis(fig[1,1], xlabel="iteration", ylabel="log-likelihood")
lines!(ax, 3:length(llhs), llhs[3:end], linewidth=5)
fig
save("../figures/mnist-llhs.png", fig)


digit_means = MLJ.transform(mach, df)
digit_modes = DataModes(res, X)


pca = MultivariateStats.fit(PCA, X', maxoutdim=3, pratio=0.99999);
U = MultivariateStats.predict(pca, X')[1:2,:]'

cm = cgrad(:roma10, n_digits, categorical=true)

fig = Figure(resolution=(1000,350));
ax = Axis(fig[1,1], title="PCA", xlabel="u₁", ylabel="u₂"); #, aspect=DataAspect());
ax2 = Axis(fig[1,2], title="GTM Means", xlabel="ξ₁", ylabel="ξ₂"); #, aspect=DataAspect());
ax3 = Axis(fig[1,3], title="GTM Modes", xlabel="ξ₁", ylabel="ξ₂"); #, aspect=DataAspect());

sc = scatter!(ax, U[:,1], U[:,2], color=targets, colormap=cm)
sc2 = scatter!(ax2, digit_means[:ξ₁], digit_means[:ξ₂], color=targets, colormap=cm, alpha=0.75)
sc3 = scatter!(ax3, digit_modes[:,1], digit_modes[:,2], color=targets, colormap=cm, alpha=0.75)

cb = Colorbar(fig[1,4], colormap=cm, limits=(0,n_digits-1), ticks=(range(0.5, stop=n_digits-1.5, length=n_digits), string.(0:n_digits-1)))

fig

save("../figures/mnist-gtm.png", fig)



R = responsibility(res, X)
pred_1 = R[1,:]


Ψ = res.W * res.Φ'

idxs = [(i,j) for i in 1:k for j in 1:k]

idx_array = zeros(Int,k,k)
for idx in 1:length(idxs)
    i,j = idxs[idx]
    idx_array[i,j] = idx
end


kx = range(-1.0, stop=1.0, length=k)
ky = range(-1.0, stop=1.0, length=k)

δ = (kx[2]-kx[1])/2

fig = Figure(resolution = (1200,1200))
ax = Axis(fig[1,1], xlabel="ξ₁", ylabel="ξ₂", title="GTM Latent Features", xlabelsize=35, ylabelsize=35, titlesize=50)
#s = scatter!(ax, gtm.Ξ[:,1], gtm.Ξ[:,2], markersize=50, color=:white, strokecolor=:black, strokewidth=3)

for i ∈ 1:k, j ∈ 1:k
    image!(ax, [kx[i]-δ, kx[i]+δ], [ky[j]-δ, ky[j]+δ], reshape(Ψ[:, idx_array[i,j]], 28, 28)[:,end:-1:1], colormap=cmap_wb)
end

fig

save("../figures/gtm-latent-features-s_0.75.png", fig)





