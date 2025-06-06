# TODO

## Experiments - ICML

### Linear Approximation of strong class in primal

1. First I want to try just directly using the embeddings and taking either heave-side or sigmoid of the features. See how this does. To really determine if it works well, we need to measure the discrepancy between the linear approximation and the actual solution in terms of KL-divergence. We can do this in two ways:
   - measure it in the L-infinity sense: For strong class $\mathcal{F}$ and primal embeddings $\{p_i: \mathcal{X} \to [0,1] \}_{i=1}^n$, we compute the discripeancy as $$\sup_{f \in \mathcal{F}}  \min_{a \in \Delta^{n-1}} D\left(\sum_{i=1}^n a_i p_i | f \right).$$ Note the sup is achieved when $\mathcal{F}$ is compact (true in the LR case: the dual space will be compact when extended to infinity - it will be homeomorphic to a quotient of a closed finite-dim ball, and the sigmoid gives a homeomorphism to the primal space.)
   - Alternatively we could place a prior on $\mathcal{F}$. One natural choice is to do it in such a way that $\mathcal{F}$ gets a "uniform" measure.

## Experiments to run - ICLR

1. Fixed strong gt model, train various levels of conv combo on image datasets
2. Imagenet reduced -> test how convex combination count depends on num_labels
   - [x] keep k most common labels. use 100 combo.
   - [ ] update alexnet to only output specified classes
