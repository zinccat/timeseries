import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0, 1))
def poisson_process_core(lambda_: float, total_time: float, key: jax.random.PRNGKey):
    # approx total points is total_time * lam
    # time intervals are exp(lambda_) distributed, which is exp(1) / lambda_
    t = jax.random.exponential(key, shape=(int(total_time * lambda_ * 1.3),)) / lambda_
    t = jnp.cumsum(t)
    y = jnp.arange(1, t.size + 1)
    return t, y

def poisson_process(lambda_: float, total_time: float, key: jax.random.PRNGKey):
    full_t, full_y = poisson_process_core(lambda_, total_time, key)
    mask = full_t < total_time
    t = full_t[mask]
    y = full_y[mask]
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