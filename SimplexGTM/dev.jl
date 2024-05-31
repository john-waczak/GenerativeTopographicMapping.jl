include("../src/GenerativeTopographicMapping.jl")
using CairoMakie
using MLJ
using DataFrames
using TernaryDiagrams
using Distributions
using StableRNGs


rng = StableRNG(42)

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
# λs = range(400, stop=700, length=1000)

Rb = exp.(-(λs .- 480.0).^2 ./(2*(15)^2))
Rg = exp.(-(λs .- 540.0).^2 ./(2*(25)^2))
Rr = exp.(-(λs .- 600.0).^2 ./(2*(17.5)^2))

# Rb = exp.(-(abs.(λs .- 450.0)./ 10).^1)
# Rg = exp.(-(abs.(λs .- 530.0)./ 30).^2)
# Rr = exp.(-(abs.(λs .- 620.0)./ 40).^10)

# Rb = exp.(-(abs.(λs .- 450.0)./ 15).^1)
# Rg = exp.(-(abs.(λs .- 530.0)./ 35).^2)
# Rr = exp.(-(abs.(λs .- 620.0)./ 45).^10)



Rb .= Rb./maximum(Rb)
Rg .= Rg./maximum(Rg)
Rr .= Rr./maximum(Rr)

fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)
fig[1,1] = Legend(fig, [lr, lg, lb], ["Red Band", "Green Band", "Blue Band"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)
fig

save("./notes/figures/rbg-orig.png", fig)
save("./notes/figures/rbg-orig.pdf", fig)



# generate dataframe of reflectance values from combined dataset

# Npoints = 10_000
Npoints = 25_000
f_dir = Dirichlet([0.05, 0.05, 0.05])
# f_dir = Dirichlet([0.1, 0.1, 0.1])
# f_dir = Dirichlet([0.15, 0.15, 0.15])

abund = rand(rng, f_dir, Npoints)



# visualize the distribution of abundances
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund[1,:],
    abund[2,:],
    abund[3,:],
    color=[CairoMakie.RGBf(abund[:,i]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations
fig

save("./notes/figures/abundance-orig.png", fig)
save("./notes/figures/abundance-orig.pdf", fig)



X = zeros(Npoints, length(λs));    # linear mixing
X2 = zeros(Npoints, length(λs));   # non-linear mixing
for i ∈ axes(X, 1)
    X[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb)
    # X2[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb) .+ (10*abund[1,i] * abund[2,i] .* Rr .* Rg)  .+ (10*abund[1,i] * abund[3,i] .* Rr .* Rb)  .+ (10*abund[2,i] * abund[3,i] .* Rg .* Rb)
    X2[i,:] .= (abund[1,i] .* Rr) .+ (abund[2,i] .* Rg) .+ (abund[3,i] .* Rb) .+ (abund[1,i] * abund[2,i] .* Rr .* Rg)

    X[i,:] .= X[i,:] ./ maximum(X[i,:])
    X2[i,:] .= X2[i,:] ./ maximum(X2[i,:])
end


names = Symbol.(["λ_" * lpad(i, 3, "0") for i ∈ 1:length(λs)])
X = DataFrame(X, names);
X2 = DataFrame(X2, names);

fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X)
    Rᵢ = Array(X[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
fig

save("./notes/figures/sample-spectra.png", fig)
save("./notes/figures/sample-spectra.pdf", fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="λ", ylabel="Reflectance")
for i ∈ 1:10:nrow(X2)
    Rᵢ = Array(X2[i,:])
    lines!(ax, λs, Rᵢ, color=CairoMakie.RGBAf(abund[:,i]..., 0.25), linewidth=1)
end
fig

save("./notes/figures/sample-spectra-nonlinear.png", fig)
save("./notes/figures/sample-spectra-nonlinear.pdf", fig)



k = 25
m = 5
s = 1
α = 0.1
Nᵥ = 3


# k = 25
# m = 10
# s = 2
# α = 10
# Nᵥ = 3


gsm_l = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, α=α, tol=1e-9, rand_init=true, nepochs=300)
gsm_nl = GenerativeTopographicMapping.GSM(k=k, m=m, Nv=Nᵥ, s=s, α=α, tol=1e-9, rand_init=true, nepochs=300)

mach_l = machine(gsm_l, X)
mach_nl = machine(gsm_nl, X2)

fit!(mach_l, verbosity=1)
fit!(mach_nl, verbosity=1)


abund_l = DataFrame(MLJ.transform(mach_l, X));
abund_nl = DataFrame(MLJ.transform(mach_nl, X2));



# plot distribution linear data
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund_l[:,1],
    abund_l[:,2],
    abund_l[:,3],
    color=[CairoMakie.RGBf(abund_l[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations
fig

save("./notes/figures/abundance-fit.png", fig)
save("./notes/figures/abundance-fit.pdf", fig)


# plot distribution non-linear data
fig = Figure();
ax = Axis(fig[1, 1], aspect=1);

ternaryaxis!(
    ax,
    hide_triangle_labels=false,
    hide_vertex_labels=false,
    labelx_arrow = "Red",
    label_fontsize=20,
    tick_fontsize=15,
)

ternaryscatter!(
    ax,
    abund_nl[:,1],
    abund_nl[:,2],
    abund_nl[:,3],
    color=[CairoMakie.RGBf(abund_nl[i,:]...) for i ∈ 1:Npoints],
    marker=:circle,
    markersize = 10,
)

# the triangle is drawn from (0,0) to (0.5, sqrt(3)/2) to (1,0).
xlims!(ax, -0.2, 1.2) # to center the triangle and allow space for the labels
ylims!(ax, -0.3, 1.1)
hidedecorations!(ax) # to hide the axis decorations
fig

save("./notes/figures/abundance-fit-nonlinear.png", fig)
save("./notes/figures/abundance-fit-nonlinear.pdf", fig)



# plot endmembers for linear data
gsm_mdl = fitted_params(mach_l)[:gsm]
rpt = report(mach_l)
M = gsm_mdl.M                          # RBF centers
Ξ = rpt[:Ξ]                            # Latent Points
Ψ = rpt[:W] * rpt[:Φ]'                 # Projected Node Means
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]


fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    #lines!(ax, λs, exp.(Ψ[:, idx] .+ (gsm_mdl.β⁻¹/2)), color=:gray)
    Rout = exp.(Ψ[:, idx])
    Rout .= Rout ./ maximum(Rout)
    li = lines!(ax, λs, Rout, color=:gray, linewidth=3, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig


save("./notes/figures/endmembers-extracted.png", fig)
save("./notes/figures/endmembers-extracted.pdf", fig)



# plot endmembers for non-linear dataset
gsm_mdl = fitted_params(mach_nl)[:gsm]
rpt = report(mach_nl)
M = gsm_mdl.M                          # RBF centers
Ξ = rpt[:Ξ]                            # Latent Points
Ψ = rpt[:W] * rpt[:Φ]'                 # Projected Node Means
llhs = rpt[:llhs]
idx_vertices = rpt[:idx_vertices]


fig = Figure();
ax = Axis(fig[2,1], xlabel="λ", ylabel="Reflectance");
lr = lines!(ax, λs, Rr, color=mints_colors[2], linewidth=2)
lg = lines!(ax, λs, Rg, color=mints_colors[1], linewidth=2)
lb = lines!(ax, λs, Rb, color=mints_colors[3], linewidth=2)

ls_fit = []
linestyles = [:solid, :dash, :dot]
i = 1
for idx ∈ idx_vertices
    #lines!(ax, λs, exp.(Ψ[:, idx] .+ (gsm_mdl.β⁻¹/2)), color=:gray)
    Rout = exp.(Ψ[:, idx])
    Rout .= Rout ./ maximum(Rout)
    li = lines!(ax, λs, Rout, color=:gray, linewidth=3, linestyle=linestyles[i])
    push!(ls_fit, li)
    i += 1
end

fig[1,1] = Legend(fig, [lr, lg, lb, ls_fit...], ["Red Band", "Green Band", "Blue Band", "Vertex 1", "Vertex 2", "Vertex 3"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig

save("./notes/figures/endmembers-extracted-nonlinear.png", fig)
save("./notes/figures/endmembers-extracted-nonlinear.pdf", fig)





# is there a prior we can add to the Weights to force them to be positive
# additionally, we may want to explore a prior which encourages sparsity in the data space...


# Need to add function to GTM to provide vertex indices in fit report, we can get them using with_replacement_combinations([1,N], k)
# Fix PCA to guarantee number of vectors



