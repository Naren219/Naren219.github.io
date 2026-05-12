---
layout: post
title: neural network weight prediction
asset_path: /assets/images/posts/neuralpred/
---

**TL;DR**: I tried two related tasks: predicting an MLP's outputs from its weights (easy — works with concat or residual nets), and the inverse problem of inferring weights from input-output pairs (harder — required moving to a Deep Sets architecture, where the bottleneck shifts from sample count to model capacity).

---

I wanted to build a simple neural network to do a slightly tricky task: can we predict the weights of the model from just the inputs and outputs? Part I walks through a simpler task of confirming that we can simulate a forward pass from a certain architecture, and [Part II](#part-ii) will delve into the question above. 

## Part I

We have a well-defined task: given inputs and weights of another, smaller neural net, can we predict its outputs with this network? Since the function defined by the generator net (smaller one) is well-defined and continuous, by [Universal approximation theorem (UAT)](https://en.wikipedia.org/wiki/Universal_approximation_theorem), a sufficiently large ReLU net (nonlinearities are key) can approximate it. The theorem guarantees that this is possible, but provides no details for how to do it. So let's do it.

### Architecture

To keep things concise, we'll define two layers with ReLU activations for our data generator. We can easily pass in weight tensors and evaluate any input.

```py
def get_mlp_out(x, w1, w2):
  y1 = nn.ReLU()(x @ w1)
  y2 = nn.ReLU()(y1 @ w2)
  return y2
```

The predictor net has two network variations we'll compare. 

```py
class XLinear(nn.Module):
  def __init__(self, concat_features=17):
    super().__init__()

    hidden_dims = [256, 512, 512, 256, 128]
    layers = []
    in_dim = concat_features
    for hidden_dim in hidden_dims:
      layers.append(nn.Linear(in_dim, hidden_dim))
      in_dim = hidden_dim + in_dim

    self.layers = nn.ModuleList(layers)
    self.out = nn.Linear(in_dim, 1)

  def forward(self, x):
    out = x
    for layer in self.layers:
      out = F.relu(layer(out))
      out = torch.cat((out, x), dim=1)

    return self.out(out)
```

`XLinear` has strength over a plain Linear network because the generator input and weights that we concatenate at each layer can skip information bottlenecks from previous hidden layers.

To provide even more scaffolding, we'll also look at `ResLinear` (it doesn't actually have any conv layers like a typical ResNet, but maybe the skip connections would still help).

```py
class ResBlock(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.BatchNorm1d(out_dim),
      nn.ReLU(),
      nn.Linear(out_dim, out_dim),
      nn.BatchNorm1d(out_dim),
    )
    if in_dim != out_dim:
        self.skip = nn.Linear(in_dim, out_dim)
    else:
        self.skip = nn.Identity()

  def forward(self, x):
    return F.relu(self.net(x) + self.skip(x))

class ResLinear(nn.Module):
  def __init__(self, concat_features = 17, 
                hidden_dim = 256, out_dim = 1):
    super().__init__()
    self.l1 = nn.Linear(concat_features, hidden_dim)
    self.res2 = ResBlock(hidden_dim, hidden_dim)
    self.res3 = ResBlock(hidden_dim, hidden_dim)
    self.res4 = ResBlock(hidden_dim, hidden_dim)
    self.l5 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    x = F.relu((self.l1(x)))
    x = self.res2(x)
    x = self.res3(x)
    x = self.res4(x)
    return self.l5(x)
```

The x samples are the concatenation of the generator net's input and weight tensors and the y samples are the evaluated output (using `get_mlp_out`).

### Results

After training with a batch_size of 1024 and 50000 training steps, we get the following (note: all the loss curves below use exponential moving average to smooth out the noise).

![Comparing XLinear to Linear]({{ page.asset_path }}xlin-comp.png)

The concatenation technique did work! But `ResLinear` did even better.

![Comparing XLinear to ResLinear]({{ page.asset_path }}xvsres.png)

Looking at the tail of this graph, we see that we've reached almost 0 loss (as expected)! This loss can't be memorization since we sample fresh generator inputs and weights at each batch.

---
## Part II

To reiterate: can we build a weight predictor by taking in the inputs and outputs of a generator network?

My first instinct is to try doing the same thing as above and see what results we get. Before that, a few comments -- the possible number of weight matrix combinations is massive. how do we incentivize the bigger network to learn the structure of the smaller one. With sufficient I/O samples, would the smaller model's architecture be the least lossy path?

To begin, let's first try a much simpler generator net with only one linear layer. 

```py
def get_simple_data(num_unknowns, num_pairs):
  # weights are the unknowns
  w = torch.randn(batch_size, num_unknowns, 1)
  pairs = []
  # we take multiple pairs for each input 
  # to give more info about the weight layer
  for i in range(num_pairs):
    x = torch.randn(batch_size, num_unknowns)
    y = torch.bmm(x.unsqueeze(1), w).squeeze(1)
    pairs.append(x)
    pairs.append(y)

  inp = torch.cat(tuple(pairs), dim=1)
  out = w.flatten(1)

  return inp, out
```

Using this data generator, we should expect that our model (we used `ResLinear`) should perform better when given more equations (`num_pairs`) than unknowns (`num_unknowns`) since the system is overdetermined. So we try this out:

![Comparing three pairs with five pairs with three unknowns (simple generator)]({{ page.asset_path }}simple-io-pair-comp.png)

And it worked! The model with five I/O pairs for each weight tensor performed much better. As the network is purely linear, the unknowns can be solved purely by least-squares, so this test is more of a sanity check.

Now, we move onto a more complex generation network that isn't purely linear.

```py
def get_data_v2(num_pairs):
  """
  30 pairs for 15 unknowns (2*5+5*1).
  """
  w1 = torch.randn(batch_size, 2, 5)
  w2 = torch.randn(batch_size, 5, 1)
  pairs = []
  for i in range(num_pairs):
    x = torch.randn(batch_size, 2)
    y = batch_mlp(x, w1, w2)  # MLP generator defined above
    pairs.append(x)
    pairs.append(y)

  inp = torch.cat(tuple(pairs), 1)
  return inp, (w1, w2)
```

Training sequence:
```py
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss = nn.MSELoss()

for i in t:
  x, weights = get_data_v2(num_pairs=30)
  optimizer.zero_grad()

  x_eval = torch.randn(batch_size, 2)

  # eval true weights on x_eval
  w1, w2 = weights
  y = batch_mlp(x_eval, w1, w2)

  # get weight predictions from test x
  weight_pred = model(x)

  # eval predicted weights on x_eval
  y_pred = compute_weights(x_eval, weight_pred)

  output = loss(y_pred, y)
  output.backward()
  optimizer.step()
```
We naively apply the same model to this task. However, the key modifications in the training procedure were:
1. The loss isn't computed between the predicted and actual weights. Doing such would make the training incredibly inefficient since there are many valid weight tensors that can implement the same function and we don't need the exact sequence of them.
2. The loss is computed between the true y -- which is outputted from our generator net -- and predicted y -- which is the result of running a newly-sampled eval x sample through the predicted weights.

Turns out, despite having 2x the equations as unknowns, loss doesn't decrease at all. A few reasons why:
1. Since we blindly smush the xy pairs into a 1D list in `get_data_v2`, we don't give the model any inductive biases about the groupings of values. This makes learning the I/O mappings an uphill battle.
2. The input tensors aren't permutation invariant, meaning the same layout of pairs in a different arrangement is a new unseen example for the model (which it shouldn't be).
3. With the nonlinearity in the generator net now (ReLU!), it's a lot harder for another model to learn (especially with issues 1 and 2 included).

Let's try something else: what if we directly embed x and y into some unified vector and have the model predict weights from here?

### Architecture

```py
class PairEncoder(nn.Module):
  def __init__(self, x_dim, y_dim, embed_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(x_dim + y_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, embed_dim)
    )

  def forward(self, x, y):
    pair = torch.cat([x, y], dim=-1)
    return self.net(pair)
```

Now, we have a pair encoder that transforms a concatenated x and y into an embedding vector. 

```py
class WeightPredictor(nn.Module):
  def __init__(self, x_dim, y_dim, out_dim, embed_dim=128):
    super().__init__()
    self.encoder = PairEncoder(x_dim, y_dim, embed_dim)
    self.decoder = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, out_dim)
    )

  def forward(self, x, y):
    embeddings = self.encoder(x, y)
    # avgs across input pairs in each batch. 
    # key for permutation invariance.
    embed_pooled = embeddings.mean(dim=1)
    return self.decoder(embed_pooled)
```

Finally, we decode these embeddings into the weights.

### Results
Following the same paradigm of predicting weights from the train input and using a new eval x to make a y prediction, we get this graph where we try predicting 15 unknowns from the weight layer using 20, 30, and 45 equations.

![Predicting 15 unknowns from 20, 30, and 45 eqns]({{ page.asset_path }}15-vars-test.png)

Cool results -- our new method works! Jumping from 20 -> 30 equations is helping the loss a lot, but with 45 equations, we don't see any improvement at all. The network is likely saturated by this point as the extra 15 equations are redundant.

Let's scale up the weights of the generator model and see if we get the same trend.

![Predicting 50 unknowns from 67 (🤷‍♂️), 100, and 150 eqns]({{ page.asset_path }}50-vars-test.png)

I followed the same scaling factors as before: x1.33, x2, and x3, but with 50 unknowns instead. And wow...with more weights, the number of pairs you give doesn't matter. All converges to the same loss of ~0.25. This means the bottleneck shifted from sample count to the predictor architecture: decoder may lack capacity, embedding size too small, or some other optimization difficulty. Each of these are future directions of inquiry.

### Discussion

Previously, we raised a question on whether the generator net's weight layout provides the least-cost option for predicting the right outputs. It's not clear if the training process results in a weight configuration that falls in the same equivalence class as the true set of weights or the predicted weights happen approximate the function only for this training distribution and not elsewhere. Further tests might include adding more nonlinearities to the generator network or sampling the evaluation x tensors from a different distribution and see if accuracy still holds up.

Turns out, the architecture we used is essentially a [Deep Sets](https://arxiv.org/abs/1703.06114) model where we encode the inputs independently, pool across the set, and decode. It's advantages permutation invariance and the ability to handle input sets of varying sizes. It's cool that our pretty simple weight prediction challenge independently arrived at the same methods as the paper.

There are many avenues to continue this experiment, from increasing the embedding size to see if it's the bottleneck to scaling the generator and predictor networks. Checkout the [code](https://colab.research.google.com/drive/10NUYSQniXh1DWmZ_hkGcCnLoBe_EGyhk?usp=sharing) and [email me](mailto:nmanikandan219@gmail.com) what you think. Thanks!