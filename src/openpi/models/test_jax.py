import jax
import jax.numpy as jnp


def hello():
    a = jnp.array([1, 2, 3, 4])
    x0 = 0
    t0 = 0

    print(a)

    def cond(carry):
        _, t, _ = carry
        return t <= 3

    def step(carry):
        x, t, a = carry
        a = a.at[t].set(1)
        a[t]
        a[:t]
        t += 1
        return x, t, a

    t = jax.lax.while_loop(cond, step, (x0, t0, a))
    print(t)
    print(a)


jax.jit(hello)

hello()
