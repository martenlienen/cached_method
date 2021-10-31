import weakref
from functools import lru_cache, update_wrapper


class WeaklyBoundMethod:
    """A bound method that only holds a weak reference to its object.
    Using weak references is required so that `cached_method` does not create reference
    cycles and prevent garbage collection.
    Because calling this object references the bound method and immediately calls it,
    `WeaklyBoundMethod` objects can be passed directly to `lru_cache`.
    """

    def __init__(self, method):
        self.ref = weakref.WeakMethod(method)

    def __call__(self, *args, **kwargs):
        method = self.ref()
        if method is None:
            raise RuntimeError(
                "Bound object has been garbage collected and method cannot be "
                "called anymore."
            )
        return method(*args, **kwargs)


class cached_method:
    """Cache calls to the decorated method.

    Each instance gets its own cache, so this decorator can be applied to any method, even
    on classes that are not hashable or to a classes own `__hash__` implementation.

    Behind the scenes, `cached_method` builds on `lru_cache`, so you can optionally
    specify the cache parameters.
    """

    def __init__(self, func=None, /, maxsize=None, typed=False):
        self.func = None
        self.methname = None
        self.maxsize = maxsize
        self.typed = typed

        self(func)

    def __call__(self, func):
        if self.func is not None and func is not self.func:
            raise TypeError("Cannot re-assign cached_method to a different method")
        self.func = func
        if self.func is not None:
            update_wrapper(self, self.func)
        return self

    def __set_name__(self, owner, name):
        if self.methname is None:
            self.methname = name
        elif name != self.methname:
            raise TypeError(
                "Cannot assign the same cached_method to two different names "
                f"({self.methname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.methname is None:
            raise TypeError(
                "Cannot use cached_method instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.methname!r} method."
            )
            raise TypeError(msg) from None
        method = cache.get(self.methname)
        if method is None:
            methodcache = lru_cache(maxsize=self.maxsize, typed=self.typed)
            method = methodcache(WeaklyBoundMethod(self.func.__get__(instance)))
            update_wrapper(method, self.func)
            try:
                cache[self.methname] = method
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {self.methname!r} method."
                )
                raise TypeError(msg) from None
        return method
