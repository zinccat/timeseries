import jax.numpy as jnp
from jax import random
import jax
from functools import partial

# brownian motion
@partial(jax.jit, static_argnums=(2))
def bm(key: jnp.ndarray, total_time: float = 1.0, num_timestep: int = 100) -> jnp.ndarray:
    dt = total_time / num_timestep
    sqrt_dt = jnp.sqrt(dt)
    dW = jax.random.normal(key, (num_timestep,))
    dW = dW.at[0].set(0.0)
    W = jnp.cumsum(sqrt_dt * dW)
    t = jnp.arange(num_timestep) * total_time / num_timestep
    return t, W

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    key = random.PRNGKey(0)
    total_time = 1.0
    num_timestep = 100
    t, sample = bm(key, total_time, num_timestep)
    plt.plot(t, sample, label='Brownian motion')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Brownian motion')
    plt.savefig("figures/brownian_motion.png")