#jl Use `Literate.notebook("lm-lstm.jl", ".", execute=false)` to convert to notebook.

# # Character Language Modeling using LSTM on Penn treebank

# ### Importing packages
using Base.Iterators, IterTools, Knet, Printf, LinearAlgebra, StatsBase, Random
#-

# ### Charset
# This will hold our set of characters that the model will be able to process.
struct Charset
    c2i::Dict{Any,Int}
    i2c::Vector{Any}
    eow::Int
    mask::Int
end

function Charset(charset::String; eow="", mask="-")
    i2c = [ eow; mask; [ c for c in charset ]  ]
    c2i = Dict( c => i for (i, c) in enumerate(i2c))
    return Charset(c2i, i2c, c2i[eow], c2i[mask])
end
#-

# ### TextReader
# Here we will read our input files and split into characters line-by-line.
struct TextReader
    file::String
    charset::Charset
end

function Base.iterate(r::TextReader, s=nothing)
    s === nothing && (s = open(r.file))
    eof(s) && return close(s)
    return [ get(r.charset.c2i, c, r.charset.eow) for c in readline(s)], s
end

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}
#-

# ### DataIterator
# In this iterator we will handle iterating over the data, and prepare our models input and output by converting lines of similar lengths into mini training batches.
struct DataIterator
    src::TextReader        
    batchsize::Int         
    maxlength::Int         
    batchmajor::Bool       
    bucketwidth::Int    
    buckets::Vector
    batchmaker::Function   
end

function DataIterator(src::TextReader; batchmaker = arraybatch, batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 2, numbuckets = min(128, maxlength ÷ bucketwidth))
    ## buckets[i] is an array of sentence pairs with similar length
    buckets = [ [] for i in 1:numbuckets ]
    DataIterator(src, batchsize, maxlength, batchmajor, bucketwidth, buckets, batchmaker)
end

Base.IteratorSize(::Type{DataIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{DataIterator}) = Base.HasEltype()
Base.eltype(::Type{DataIterator}) = NTuple{2}

function Base.iterate(d::DataIterator, state=nothing)
    ## When file is finished but buckets are partially full 
    if state == 0 
        for i in 1:length(d.buckets)
            if length(d.buckets[i]) > 0
                batch = d.batchmaker(d, d.buckets[i])
                d.buckets[i] = []
                return batch, state
            end
        end
        ## terminate iteration
        return nothing
    end

    while true
        src_next = iterate(d.src, state)
        
        if src_next === nothing
            state = 0
            return iterate(d, state)
        end
        
        (src_word, src_state) = src_next
        state = src_state
        src_length = length(src_word)
        
        (src_length > d.maxlength) && continue
        (src_length < 2) && continue

        i = Int(ceil(src_length / d.bucketwidth))
        i > length(d.buckets) && (i = length(d.buckets))

        push!(d.buckets[i], src_word)
        if length(d.buckets[i]) == d.batchsize
            batch = d.batchmaker(d, d.buckets[i])
            d.buckets[i] = []
            return batch, state
        end
    end
end

function arraybatch(d::DataIterator, bucket)
    src_eow = d.src.charset.eow
    src_lengths = map(x -> length(x), bucket)
    max_length = max(src_lengths...)
    x = zeros(Int64, length(bucket), max_length + 2) 

    for (i, v) in enumerate(bucket)
        to_be_added = fill(src_eow, max_length - length(v) + 1)
        x[i,:] = [src_eow; v; to_be_added]
    end
    
    ## default d.batchmajor is false
    d.batchmajor && (x = x')
    
    ## the output in lang. model is same as input sequence but shifted one step
    return (x[:, 1:end-1], x[:, 2:end]) 
end
#-

# ### Embedding and Linear layers 
mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))

struct Embed; w; end
Embed(charsetsize::Int, embedsize::Int) = Embed(param(embedsize, charsetsize))
(l::Embed)(x) = l.w[:, x]

struct Linear; w; b; end
Linear(inputsize::Int, outputsize::Int) = Linear(param(outputsize, inputsize), param0(outputsize))
(l::Linear)(x) = mmul(l.w,x) .+ l.b
#-

# ### NLL Loss masking function
# This function masks the training output array by zeros, this way we don't propagate loss from paddings
function mask(a, pad)
    a = copy(a)
    for i in 1:size(a, 1)
        j = size(a,2)
        while a[i, j] == pad && j > 1
            if a[i, j - 1] == pad
                a[i, j] = 0
            end
            j -= 1
        end
    end
    return a
end
#-

# ### RNN Language model
# Our model consists of four layers. Size of their outputs are as the following, where T is sequence length, B is batchsize, H is the hidden size of RNN, and V is vocabulary size, E is the embedding size:
#
# * **(T, B)** - Input 
# * **(E, T, B)** - Embedding
# * **(H, T, B)** - RNN
# * **(V, T, B)** - Projection
struct LModel
    srcembed::Embed
    rnn::RNN        
    projection::Linear  
    dropout::Real
    srccharset::Charset 
end

function LModel(hidden::Int, srcembsz::Int, srccharset::Charset; layers=1, dropout=0)
    
    srcembed = Embed(length(srccharset.i2c), srcembsz)
    rnn = RNN(srcembsz, hidden; bidirectional=false, numLayers=layers, dropout=dropout)
    projection = Linear(hidden, length(srccharset.i2c))
    
    LModel(srcembed, rnn, projection, dropout, srccharset)
end

function (s::LModel)(src, tgt; average=true)
    s.rnn.h, s.rnn.c = 0, 0
    srcembed = s.srcembed(src)
    rnn_out = s.rnn(srcembed)
    dims = size(rnn_out)
    output = s.projection(dropout(reshape(rnn_out, dims[1], dims[2] * dims[3]), s.dropout))
    scores = reshape(output, size(output, 1), dims[2], dims[3])
    nll(scores, mask(tgt, s.srccharset.eow); dims=1, average=average)
end

function generate(s::LModel; start="", maxlength=30)
    s.rnn.h, s.rnn.c = 0, 0
    chars = fill(s.srccharset.eow, 1)
    start = [ c for c in start ]
    starting_index = 1
    for i in 1:length(start)
        push!(chars, s.srccharset.c2i[start[i]])
        charembed = s.srcembed(chars[i:i])
        rnn_out = s.rnn(charembed)
        starting_index += 1
    end
    
    for i in starting_index:maxlength
        charembed = s.srcembed(chars[i:i])
        rnn_out = s.rnn(charembed)
        output = s.projection(dropout(rnn_out, s.dropout))
        push!(chars, s.srccharset.c2i[ sample(s.srccharset.i2c, Weights(Array(softmax(reshape(output, length(s.srccharset.i2c)))))) ] )
        
        if chars[end] == s.srccharset.eow
            break
        end
    end
    
    join([ s.srccharset.i2c[i] for i in chars ], "")
end
#-

# ### Evaluation metrics
#
# Here we define a `loss(model, data)` which returns a `(Σloss, Nloss)` pair if `average=false` and
# a `Σloss/Nloss` average if `average=true` for a whole dataset.
# 
# `report_lm(loss)` calculates character perplexity and bit-per-character metrics.
function loss(model, data; average=true)
    mean([model(x,y) for (x,y) in data])
end

report_lm(loss) = (loss=loss, ppl=exp.(loss), bpc=loss ./ log(2))
#-

# ### Training procedure
# Train our model using Adam optimizer and saving the best performing model on dev set.
function train!(model, steps, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps=steps) do y
        devloss = loss(model, dev)
        tstloss = map(d->loss(model,d), tst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (trn=report_lm(tstloss), dev=report_lm(devloss))
    end
    return bestmodel
end
#-

## seed random generators for reproducability
Random.seed!(123);

## Define char set, batchsize and max sequence length
char_set = " #\$&'*-./0123456789<>N\\abcdefghijklmnopqrstuvwxyz"
datadir = "../data/ptb"
BATCHSIZE, MAXLENGTH = 16, 256

@info "Reading data"
charset = Charset(char_set)
train_reader = TextReader("$datadir/train.txt", charset)
dev_reader = TextReader("$datadir/valid.txt", charset)
test_reader = TextReader("$datadir/test.txt", charset)

dtrn = DataIterator(train_reader, batchsize=BATCHSIZE, maxlength=MAXLENGTH)
ddev = DataIterator(dev_reader, batchsize=BATCHSIZE, maxlength=MAXLENGTH)
dtst = DataIterator(test_reader, batchsize=BATCHSIZE, maxlength=MAXLENGTH)
#-

@info "Initializing Language Model"
epochs = 10
ctrn = collect(dtrn)
trn = collect(flatten(shuffle!(ctrn) for i in 1:epochs))
trnmini = ctrn[1:20]
dev = collect(ddev)
model = LModel(256, 256, charset; layers=2, dropout=0.2)
#-

@info "Starting training ..."
model = train!(model, length(ctrn), trn, dev, trnmini)
#-

@info "Finished training, Starting evaluation ..."
trnloss = loss(model, dtrn);
println("Training set scores:       ", report_lm(trnloss))
devloss = loss(model, ddev);
println("Development set scores:    ", report_lm(devloss))
tstloss = loss(model, dtst);
println("Test set scores:           ", report_lm(tstloss))
#-

# ### Sampling sentences from the model
@info "Sample sentences from the model"
print("enter prompt or leave empty for no prompt, CTRL+C to exit")

while true
    print("prompt:")
    prompt = lowercase(readline()) 
    println(generate(model; start=prompt, maxlength=1024))
end
#-
