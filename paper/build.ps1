$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Initial LaTeX compilation failed.' }
    $previousBibInputs = $env:BIBINPUTS
    try {
        $env:BIBINPUTS = "$PSScriptRoot;"
        Push-Location 'build'
        try {
            & bibtex main
            if ($LASTEXITCODE -ne 0) { throw 'BibTeX compilation failed.' }
        } finally { Pop-Location }
    } finally { $env:BIBINPUTS = $previousBibInputs }
    1..2 | ForEach-Object {
        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw 'Final LaTeX compilation failed.' }
    }
    $previewDirectory = Join-Path $PSScriptRoot '../output/pdf'
    New-Item -ItemType Directory -Force -Path $previewDirectory | Out-Null
    Copy-Item -LiteralPath 'build/main.pdf' -Destination (Join-Path $previewDirectory 'introduction_literature_review.pdf')
} finally { Pop-Location }
