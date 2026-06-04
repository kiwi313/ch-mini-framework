# Character-Level Mini Framework

This is an educational mini framework for comparing simple character-level language models.

The project is inspired by Andrej Karpathy's **makemore** series and the ideas from the 2003 paper by Bengio et al. about neural probabilistic language models and distributed representations.

The goal of this project is not to build a production-ready machine learning library, but to understand how different language modeling approaches work under the hood.

## Current models

The framework currently supports two models:

### N-gram counting model

A classical counting-based character-level n-gram model.

It builds a probability table from the training set and evaluates the model on train, development and test splits.

Example result:

```python
{
    "model": "3-gram counting",
    "train_loss": 2.1762,
    "dev_loss": 2.4173,
    "test_loss": 2.4360
}
```

### MLP model

A simple neural character-level language model using:

* character embeddings,
* a hidden layer with `tanh`,
* cross entropy loss,
* gradient descent,
* train/dev/test evaluation,
* sampling from the trained model.

Example result:

```python
{
    "model": "mlp",
    "block_size": 4,
    "embeding_size": 20,
    "neurons": 300,
    "train_loss": 1.9120,
    "dev_loss": 2.3181,
    "test_loss": 2.3675,
    "training_steps": 100000
}
```

## Example usage

```python
from models import NGramCounting, MLP

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
```

## Project purpose

This project was created to practice and understand:

* train/dev/test splits,
* negative log likelihood,
* cross entropy,
* n-gram language models,
* embeddings,
* simple neural networks,
* gradient descent,
* model evaluation,
* sampling,
* basic object-oriented design in Python.

## Inspiration

This project is based on ideas from:

* Andrej Karpathy's **makemore** series,
* Bengio et al. 2003, *A Neural Probabilistic Language Model*.

## Status

Work in progress.

Planned improvements:

* cleaner class structure,
* more model types,
* better sampling options,
* experiment comparison utilities,
* improved documentation.
