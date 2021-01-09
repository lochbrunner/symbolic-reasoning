# White Paper

A bit more than an abstract.

## Build

```bash
cp docs/whitepaper/whitepaper.bib docs/whitepaper
pdflatex -output-directory out/whitepaper docs/whitepaper/whitepaper.tex
biber out/whitepaper/whitepaper.bcf 
```