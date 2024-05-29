include("../src/GenerativeTopographicMapping.jl")
using CairoMakie
using MLJ
using DataFrames
using TernaryDiagrams
using Distributions


include("./utils/makie-defaults.jl")
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



# Test 1: Linear Mixing
λs = range(400, stop=700, length=250)

Rb = exp.(-(λs .- 450.0).^2 ./(2*(5)^2))
Rg = exp.(-(λs .- 530.0).^2 ./(2*(10)^2))
Rr = exp.(-(λs .- 620.0).^2 ./(2*(15)^2))

Rb = exp.(-(abs.(λs .- 450.0)./ 10).^1)
Rg = exp.(-(abs.(λs .- 530.0)./ 30).^2)
Rr = exp.(-(abs.(λs .- 620.0)./ 40).^10)



Rb .= Rb./maximum(Rb)
Rg .= Rg./maximum(Rg)
Rr .= Rr./maximum(Rr)

fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2])
lg = lines!(ax, λs, Rg, color=mints_colors[1])
lb = lines!(ax, λs, Rb, color=mints_colors[3])
fig[1,1] = Legend(fig, [lr, lg, lb], ["Red Band", "Green Band", "Blue Band"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
fig





# generate dataframe of reflectance values from combined dataset

Npoints = 10_000
f_dir = Dirichlet([0.1, 0.1, 0.1])

abund = rand(f_dir, Npoints)

# We should normalize to account for irregular lighting effects
X = zeros(Npoints, length(λs))
for i ∈ axes(X, 1)
    X[i,:] .= abund[1,i] .* Rr + abund[2,i] .* Rg + abund[3,i] .* Rb
    X[i,:] .= X[i,:] ./ maximum(X[i,:])
end



names = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
X = DataFrame(X, names)

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10
    Rᵢ = Array(X[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBf(abund[:,i]...))
end
fig


k = 20
m = 3
s = 2
α = 0.01
Nᵥ = 3

gsm = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, α=α, tol=1e-9, rand_init=true)
mach = machine(gsm, X)
fit!(mach, verbosity=1)


ξmean_pred = DataFrame(MLJ.transform(mach, X))[1:3,:]
abund'[1:3,:]

# need a function to get the vertex coordinates...






gsm_mdl = fitted_params(mach)[:gsm]
rpt = report(mach)
M = gsm_mdl.M                          # RBF centers
Ξ = rpt[:Ξ]                            # Latent Points
Ψ = rpt[:W] * rpt[:Φ]'                 # Projected Node Means
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]

Ψ[:,idx_vertices]


fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
#lr = lines!(ax, λs, Rr, color=mints_colors[2])
#lg = lines!(ax, λs, Rg, color=mints_colors[1])
#lb = lines!(ax, λs, Rb, color=mints_colors[3])
#for idx ∈ idx_vertices[1]:idx_vertices[2]
for idx ∈ idx_vertices
    lines!(ax, λs, Ψ[:, idx],)
end

#lines!(ax, λs, Ψ[:, idx_vertices[3]])

fig[1,1] = Legend(fig, [lr, lg, lb], ["Red Band", "Green Band", "Blue Band"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
fig


# is there a prior we can add to the Weights to force them to be positive
# additionally, we may want to explore a prior which encourages sparsity in the data space...


# Need to add function to GTM to provide vertex indices in fit report, we can get them using with_replacement_combinations([1,N], k)
# Fix PCA to guarantee number of vectors



