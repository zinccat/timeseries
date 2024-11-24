import jax
import jax.numpy as jnp

from poisson_process import poisson_process_core

def skellam_process(lam1: float, lam2: float, total_time: float, key: jax.random.PRNGKey):
    key1, key2 = jax.random.split(key)
    t1, y1 = poisson_process_core(lam1, total_time, key1)
    t2, y2 = poisson_process_core(lam2, total_time, key2)

    # merge t1 and t2
    total_t = jnp.concatenate([t1, t2])
    # argsort
    idx = jnp.argsort(total_t)
    total_t = total_t[idx]
    mask = total_t < total_time
    total_t = total_t[mask]
    y = jnp.concatenate([jnp.ones_like(t1), -jnp.ones_like(t2)])[idx][mask]
    y = jnp.cumsum(y)
    total_t = jnp.append(total_t, total_time)
    y = jnp.append(y, y[-1])
    return total_t, y

if __name__ == "__main__":
    lam1 = 20.0
    lam2 = 10.0
    total_time = 1.0
    key = jax.random.PRNGKey(0)

    t, y = skellam_process(lam1, lam2, total_time, key)
    from matplotlib import pyplot as plt
    import os
    os.makedirs('figures', exist_ok=True)
    
    plt.step(t, y)
    plt.xlabel('Time')
    plt.ylabel('Number of events')
    plt.title('Skellam process with rates $\lambda_1 = 20$ and $\lambda_2 = 10$')
    plt.savefig('figures/skellam_process.png')
    plt.show()