import torch
import torch.nn.functional as F

def load_data(file):
    data = open(f"{file}","r").read().splitlines()

    chars = sorted(list(set("".join(data))))

    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi["."] = 0
    itos = {i:s for s,i in stoi.items()}


    n1 = int(len(data)*0.8)
    n2 = int(len(data)*0.9)

    return {
        "all_data": data,
        "train": data[:n1],
        "dev": data[n1:n2],
        "test": data[n2:],
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(stoi)
    }

class NGramCounting:

    def __init__(self,n,file):
        self.n = n
        self.file = file
        self.N = None
        self.dataset = None
        self.P = None
        self.train_loss = None
        self.dev_loss = None
        self.test_loss = None

    def _build_n_gram_counts(self):
        words = self.dataset["train"]
        vocab_size = self.dataset["vocab_size"]
        context_size = self.n - 1

        N = torch.zeros((vocab_size ** context_size, vocab_size), dtype=torch.int32)

        for w in words:
            context = [0] * context_size
            chars = list(w) + ["."]

            for ch in chars:
                ix = self.dataset["stoi"][ch]

                row = 0
                for c in context:
                    row = row * vocab_size + c

                N[row, ix] += 1

                context = context[1:] + [ix]

        return N
    
    def _counts_to_probs(self):
        P = (self.N+1).float()
        P = P/P.sum(1,keepdim = True)   
        return P

    def _evaluate_n_gram_loss(self,split):
        vocab_size = self.dataset["vocab_size"]
        words = self.dataset[split]
        context_size = self.n-1
        loglikelihood = 0
        count = 0 


        for w in words:
            context = [0] * context_size
            chars = list(w) + ["."]

            for ch in chars:
                ix = self.dataset["stoi"][ch]

                row = 0
                for c in context:
                    row = row * vocab_size + c

                prob = self.P[row,ix]

                loglikelihood += torch.log(prob)
                count+=1
                context = context[1:] + [ix]
                
        loss = -loglikelihood/count
        return loss.item()

    def sample(self, samples):
        if self.P is None:
            raise ValueError("Run the model before sampling.")

        names = []

        for i in range(samples):
            out = []

            vocab_size = self.dataset["vocab_size"]
            context_size = self.n - 1
            context = [0] * context_size

            while True:
                row = 0
                for c in context:
                    row = row * vocab_size + c

                probs = self.P[row]

                ix = torch.multinomial(probs,num_samples=1,replacement=True).item()

                if ix == 0:
                    break

                out.append(self.dataset["itos"][ix])
                context = context[1:] + [ix]

            name = "".join(out)
            names.append(name)

        return names

           

                        
    
    def run(self):

        if self.n<2:
            raise ValueError("n must be at least 2")

        self.dataset = load_data(self.file)
        self.N = self._build_n_gram_counts() 
        self.P = self._counts_to_probs()
        
        self.train_loss = self._evaluate_n_gram_loss(split="train")
        self.dev_loss = self._evaluate_n_gram_loss(split="dev")
        self.test_loss = self._evaluate_n_gram_loss(split = "test")

        return {
            "model" : f"{self.n}-gram counting",
            "train_loss": self.train_loss,
            "dev_loss": self.dev_loss,
            "test_loss" : self.test_loss

        }
    

class MLP:
    def __init__(self,file,block_size,embeding_size,neurons,training_steps):
        self.file = file
        self.block_size = block_size
        self.embeding_size = embeding_size
        self.neurons = neurons
        self.training_steps = training_steps

        self.train_loss = None
        self.dev_loss = None
        self.test_loss = None

        self.dataset = None
        self.tensors = None

        self.C = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        


    def _build_mlp_tensors(self):
        words = self.dataset["all_data"]
        def build_sets(words):
            X,Y = [],[]
            for w in words:
                context = [0] * self.block_size
                for ch in w + ".":
                    ix = self.dataset["stoi"][ch]
                    X.append(context)
                    Y.append(ix)
                    context = context[1:] + [ix]
            X = torch.tensor(X)
            Y = torch.tensor(Y)

            return X,Y
        n1 = int(len(words)*0.8)
        n2 = int(len(words)*0.9)

        Xtrain,Ytrain = build_sets(words[:n1]) 
        Xdev, Ydev = build_sets(words[n1:n2])
        Xtest, Ytest = build_sets(words[n2:])

        return{
            "Xtrain" : Xtrain,
            "Ytrain" : Ytrain,
            "Xdev" : Xdev,
            "Ydev" : Ydev,
            "Xtest" : Xtest,
            "Ytest" : Ytest
        }

    def _train(self):
        g = torch.Generator().manual_seed(2147483647)

        vocab_size = self.dataset["vocab_size"]

        self.C = torch.randn((vocab_size, self.embeding_size), generator=g)
        self.W1 = torch.randn((self.embeding_size * self.block_size, self.neurons), generator=g) * 0.01
        self.b1 = torch.randn(self.neurons, generator=g)
        self.W2 = torch.randn((self.neurons, vocab_size), generator=g) * 0.01
        self.b2 = torch.zeros(vocab_size)

        parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

        for p in parameters:
            p.requires_grad = True

        Xtrain = self.tensors["Xtrain"]
        Ytrain = self.tensors["Ytrain"]

        for i in range(self.training_steps):
            ix = torch.randint(0, Xtrain.shape[0], (64,))

            emb = self.C[Xtrain[ix]]
            h = torch.tanh(emb.view(-1, self.block_size * self.embeding_size) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2

            loss = F.cross_entropy(logits, Ytrain[ix]) + 0.001 * sum((p ** 2).mean() for p in parameters)

            for p in parameters:
                p.grad = None

            loss.backward()

            lr = 0.1 if i < 100000 else 0.01

            for p in parameters:
                p.data += -lr * p.grad
        
    def _evaluate_loss(self, split):
        X = self.tensors[f"X{split}"]
        Y = self.tensors[f"Y{split}"]

        emb = self.C[X]
        h = torch.tanh(emb.view(-1, self.block_size * self.embeding_size) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        loss = F.cross_entropy(logits, Y)
        return loss.item()
    
    
    def sample(self,samples):
        if self.C is None:
            raise ValueError("Run the model before sampling.")
        names = []
        for i in range(samples):
            out = []
            context = [0] * self.block_size
            while True:
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1,-1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim = 1)
                ix = torch.multinomial(probs,num_samples = 1).item()
                
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
            
            #print(''.join(itos[i] for i in out))
            name = ''.join(self.dataset["itos"][i] for i in out)
            names.append(name)
        return names

    def run(self):
        self.dataset = load_data(self.file)

        self.tensors = self._build_mlp_tensors()

        self._train()
        self.train_loss = self._evaluate_loss("train")
        self.dev_loss = self._evaluate_loss("dev")
        self.test_loss = self._evaluate_loss("test")


        return {
            "model": "mlp",
            "block_size": self.block_size,
            "embeding_size": self.embeding_size,
            "neurons": self.neurons,
            "train_loss": self.train_loss,
            "dev_loss": self.dev_loss,
            "test_loss": self.test_loss,
            "training_steps": self.training_steps,
    }