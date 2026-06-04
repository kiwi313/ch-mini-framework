from models2 import NGramCounting,MLP

ngram = NGramCounting(n=3, file="names.txt")

mlp = MLP(
    file="names.txt",
    block_size=4,
    embeding_size=20,
    neurons=300,
    training_steps=100000
)

result_ngram = ngram.run()
result_mlp = mlp.run()

ngram_samples = ngram.sample(5)
mlp_samples = mlp.sample(5)

print(result_ngram)
print(ngram_samples)
print(result_mlp)
print(mlp_samples)