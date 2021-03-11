all: lm-lstm.ipynb

%.ipynb: %.jl
	julia -e 'using Literate; Literate.notebook("$<", ".", execute=false)'
