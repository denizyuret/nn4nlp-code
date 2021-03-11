all: cnn-class.ipynb

%.ipynb: %.jl
	julia -e 'using Literate; Literate.notebook("$<", ".", execute=false)'
