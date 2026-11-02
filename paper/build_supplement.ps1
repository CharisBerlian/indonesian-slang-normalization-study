$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    1..2 | ForEach-Object {
        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build supplement.tex
        if ($LASTEXITCODE -ne 0) { throw 'Supplement LaTeX compilation failed.' }
    }
    $previewDirectory = Join-Path $PSScriptRoot '../output/pdf'
    New-Item -ItemType Directory -Force -Path $previewDirectory | Out-Null
    Copy-Item -LiteralPath 'build/supplement.pdf' -Destination (Join-Path $previewDirectory 'indonesian_slang_sentiment_supplement.pdf')
} finally { Pop-Location }
