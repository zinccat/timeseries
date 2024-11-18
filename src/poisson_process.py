import jax
import jax.numpy as jnp

def poisson_process(lambda_, total_time, key):
    # approx total points is total_time * lam
    # time intervals are exp(lambda_) distributed, which is exp(1) / lambda_
    t = jax.random.exponential(key, shape=(int(total_time * lambda_ * 2),)) / lambda_
    t = jnp.cumsum(t)
    t = t[t < total_time]
    y = jnp.arange(1, t.size + 1)
    # add final time point
    t = jnp.append(t, total_time)
    y = jnp.append(y, y[-1])
    return t, y

if __name__ == "__main__":
    lam = 20.0
    total_time = 1.0
    key = jax.random.PRNGKey(0)

    t, y = poisson_process(lam, total_time, key)
    from matplotlib import pyplot as plt
    import os
    os.makedirs('figures', exist_ok=True)
    
    plt.step(t, y)
    plt.xlabel('Time')
    plt.ylabel('Number of events')
    plt.title('Poisson process with rate $\lambda = 20$')
    plt.savefig('figures/poisson_process.png')
    plt.show()