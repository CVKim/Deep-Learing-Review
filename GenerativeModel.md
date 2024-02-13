# Generative Model Comparison

Generative models can be broadly classified into six categories. The features, advantages, and disadvantages of each model are as follows:

## Auto-regressive models (ARMs)
- **Features**: Calculate the likelihood of data by multiplying the conditional probabilities of ordered variables.
- **Advantages**: Can calculate the likelihood of given data.
- **Disadvantages**: Slow sampling and cannot learn latent features inherent in the data.

## Variational Auto-encoders (VAEs)
- **Features**: A latent variable-based generative model that marginalizes over the joint distribution of data x and latent variable z.
- **Advantages**: Fast learning and sampling, can learn latent features of data.
- **Disadvantages**: Difficult to handle likelihood, and the prior distribution is limited.

## Energy Based Models (EBMs)
- **Features**: Estimate distribution using an energy function.
- **Advantages**: No constraints make it convenient.
- **Disadvantages**: Likelihood and sampling are difficult to manage.

## Generative Adversarial Networks (GANs)
- **Features**: Generate data by adversarially training a discriminator and a generator.
- **Advantages**: Good sample quality, fast training, and sampling.
- **Disadvantages**: Likelihood is undefined, and training can be unstable.

## Normalizing Flows
- **Features**: Estimate distribution using an invertible mapping function from a simple base distribution p(z) to a complex data distribution p(x).
- **Advantages**: Can calculate exact likelihood values, fast sampling.
- **Disadvantages**: Architecture imposes certain restrictions.

## Diffusion
- **Features**: Learn the process of gradually adding noise to data x to create noisy data, then reversing back to data x.
- **Advantages**: Can estimate complex distributions well without special restrictions, despite likelihood being difficult to handle.
- **Disadvantages**: Typically slower than other models due to the iterative nature of the reverse process.
