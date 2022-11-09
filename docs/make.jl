using GTM
using Documenter

DocMeta.setdocmeta!(GTM, :DocTestSetup, :(using GTM); recursive=true)

makedocs(;
    modules=[GTM],
    authors="John Waczak <john.louis.waczak@gmail.com>",
    repo="https://github.com/john-waczak/GTM.jl/blob/{commit}{path}#{line}",
    sitename="GTM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://john-waczak.github.io/GTM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/john-waczak/GTM.jl",
    devbranch="main",
)
