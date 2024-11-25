import jax.numpy as jnp
from jax import random
import jax
from functools import partial

# fractional brownian motion covariance
def fbm_cov(t: jnp.ndarray, H: float) -> jnp.ndarray:
    def inner(t1, t2):
        return 0.5 * (t1**(2*H) + t2**(2*H) - jnp.abs(t1-t2)**(2*H))
    f = jax.vmap(inner, in_axes=(0, None))
    f = jax.vmap(f, in_axes=(None, 0))
    return f(t, t)

# fractional brownian motion
@partial(jax.jit, static_argnums=(2))
def fbm(key: jnp.ndarray, total_time: float = 1.0, num_timestep: int = 100, H: float = 0.5) -> jnp.ndarray:
    t = jnp.arange(1, num_timestep) * total_time / (num_timestep - 1)
    cov = fbm_cov(t, H)
    mean = jnp.zeros_like(t)
    sample = jax.random.multivariate_normal(key, mean, cov)
    # start from 0
    sample = jnp.concatenate([jnp.zeros(1), sample])
    t = jnp.concatenate([jnp.zeros(1), t])
    return t, sample

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    key = random.PRNGKey(0)
    total_time = 1.0
    timestep = 100
    # t = jnp.arange(1, timestep + 1) * total_time / timestep
    for H in [0.2, 0.5, 0.8]:
        t, sample = fbm(key, total_time, timestep, H)
        plt.plot(t, sample, label=f"H = {H}")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Fractional Brownian motion')
    plt.savefig("figures/fractional_brownian_motion.png")