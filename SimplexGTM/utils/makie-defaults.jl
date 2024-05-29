montserrat_path = "./font-files/montserrat"
@assert ispath(montserrat_path)

mints_colors = [
    colorant"#3cd184", # mint green
    colorant"#f97171", # dark coral
    colorant"#1e81b0", # dark blue
    colorant"#66beb2", # dark blue-green
    colorant"#f99192", # light coral
    colorant"#8ad6cc", # middle blue-green
    colorant"#3d6647", # dark green
#    colorant"#8FDDDF", # middle blue
]


mints_font = (;
              regular = joinpath(montserrat_path, "static", "Montserrat-Regular.ttf"),
              italic = joinpath(montserrat_path, "static", "Montserrat-Italic.ttf"),
              bold = joinpath(montserrat_path, "static", "Montserrat-Bold.ttf"),
              bold_italic = joinpath(montserrat_path, "static", "Montserrat-BoldItalic.ttf"),
              )



# note: to get a specific attribute (for example, Axis attribute x), go into help mode with ?
# and type Axis.x
# for a full list, type ? then Axis.

mints_theme = Theme(
    fontsize=17,
    fonts = mints_font,
    #colormap = :haline,
    colormap = :viridis,
    palette = (
        color = mints_colors,
        patchcolor = mints_colors,
    ),
    cycle = [[:linecolor, :markercolor,] => :color,],
    Axis=(
        xlabelsize=15,                   ylabelsize=15,
        xticklabelsize=13,               yticklabelsize=13,
        xticksize=3,                     yticksize=3,
        xminorgridvisible=true,          yminorgridvisible=true,
        xgridwidth=2,                    ygridwidth=2,
        xminorgridwidth=1,               yminorgridwidth=1,
        xminorticks=IntervalsBetween(5), yminorticks=IntervalsBetween(5)
    ),
    Colorbar=(
        fontsize=13,
    )
)

