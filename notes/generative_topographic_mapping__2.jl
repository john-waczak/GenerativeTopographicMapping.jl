using Pkg
Pkg.activate(".")

using GenerativeTopographicMapping
using Plots, LinearAlgebra, Statistics, MLJ
using Distances, MultivariateStats
using RDatasets
using Distributions


# load the dataset
iris = dataset("datasets","iris")
Dataset =iris[:, 1:4]
classes = iris[:,5]

model = GTM(k=10,
            m=5,
            σ=0.5,
            )

m = machine(model, Dataset)
res = fit!(m)

X̃ = MLJ.transform(m, Dataset)

gtm_fitted = fitted_params(m)[1]

rpt = report(m)
my_classes = rpt.classes


X_means = GenerativeTopographicMapping.get_means(gtm_fitted, Matrix(Dataset))
X_modes = GenerativeTopographicMapping.get_modes(gtm_fitted, Matrix(Dataset))

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

n_rbf_centers = size(gtm_fitted.M,1)

for k ∈ 1:n_rbf_centers
    mypdf = MvNormal(gtm_fitted.M[k,:], gtm_fitted.σ²)
    for j ∈ 1:size(xs,1), i ∈ 1:size(xs,1)
        zs[i,j] += pdf(mypdf, [xs[i], ys[j]])
    end
end

heatmap!(plot1, xs, ys, zs, color=:thermal, colorbar=false)

scatter!(plot1,
         gtm_fitted.X[:,1], gtm_fitted.X[:,2],
         color=:white,
         ms=2,
         msw=0,
         label="Node Points"
         )

display(plot1)

savefig("latent_space_visualization.png")
savefig("latent_space_visualization.svg")
savefig("latent_space_visualization.pdf")




# --------------------------------------------------------------------------------
p1 = scatter(Dataset[:,1],Dataset[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Original Data")
p2 = scatter(X_means[:,1], X_means[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Mean node responsability")
p3 = scatter(X_modes[:,1], X_modes[:,2], marker_z=int(classes), color=:Accent_3, label="", colorbar=false, title="Mode node responsability")

p_postfit = plot(p1, p2, p3, layout=(1,3), size=(1200, 450), plot_title="GTM Fit")


savefig(p_postfit, "gtm_post_fit.png")
savefig(p_postfit, "gtm_post_fit.svg")
savefig(p_postfit, "gtm_post_fit.pdf")

p_postfit
