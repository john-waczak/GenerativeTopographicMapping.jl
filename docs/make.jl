using GenerativeTopographicMapping
using Documenter

DocMeta.setdocmeta!(GenerativeTopographicMapping, :DocTestSetup, :(using GenerativeTopographicMapping); recursive=true)

makedocs(;
    modules=[GenerativeTopographicMapping],
    authors="John Waczak <john.louis.waczak@gmail.com>",
    repo="https://github.com/john-waczak/GenerativeTopographicMapping.jl/blob/{commit}{path}#{line}",
    sitename="GenerativeTopographicMapping.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://john-waczak.github.io/GenerativeTopographicMapping.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/john-waczak/GenerativeTopographicMapping.jl",
    devbranch="main",
)
