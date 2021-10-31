# cached_method

The `@cached_method` decorator is the equivalent of
[`functools.cached_property`](https://docs.python.org/3/library/functools.html#functools.cached_property)
for methods. This means that each instance has its own cache, so that the caches get
garbage collected as soon as the owning objects are. The main advantages of
`cached_method` over applying `functools.lru_cache` directly to methods are
1. the surrounding class need not be hashable,
2. and the class objects are not collected in a global cache, extending their lifetime.
This makes `cached_method` applicable to classes holding references to scarce resources
such as GPU memory that you want to be freed as soon as possible. Furthermore, the
decorator can cache the output of `__hash__` because it does not hash the object itself
for cache lookups.

Implementation-wise `cached_method` closely follows `functools.cached_property` though it
eschews the internal locking, which is now [considered a
mistake](https://bugs.python.org/issue43468). Since cached methods should be idempotent
anyway, we just accept possibly calling the method multiple times in parallel with
equivalent arguments if the object is used in multi-threaded contexts.

```python
from cached_method import cached_method

class GPUVector:
    def __init__(self, data):
        # data is some smart tensor object as found in pytorch, tensorflow, etc.
        self.data = data

    # Only cache the 2 most-recently used norms
    @cached_method(maxsize=2)
    def norm(self, p=2):
        return (self.data ** p).sum() ** (1 / p)

    @cached_method
    def __hash__(self):
        # A costly GPU-to-CPU transfer, so we want to hash the result
        return hash(tuple(self.data.to_cpu()))
```

If you are working with small, hashable objects that do not have to be gargabe collected
as soon as possible, consider the [method hashing technique described in the Python
FAQ](https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls). It gives
you an easy way to control the total cache size and allows cache hits between
equivalent-but-not-identical objects. Of course, caching on the class level means that
objects stay live until you clear the cache manually, even if the cache is the last object
referencing them.
