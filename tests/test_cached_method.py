import pytest
from cached_method import cached_method


class CachedMethodWithDoc:
    @cached_method
    def do(self, arg):
        """Do the computation."""
        return arg * 5


def test_returns_cached_result_for_same_args():
    class CachedCallCounter:
        def __init__(self):
            self.called = 0

        @cached_method
        def compute(self, arg):
            self.called += 1
            return arg

    obj = CachedCallCounter()

    obj.compute(10)
    assert obj.called == 1

    # Check that the same value is returned without the method being called again
    assert obj.compute(10) == 10
    assert obj.called == 1

    # Calling with different arguments triggers another method call
    obj.compute(20)
    assert obj.called == 2


def test_lru_cache_arguments_can_be_set():
    class RestrictedCachedCallCounter:
        def __init__(self):
            self.called = 0

        @cached_method(maxsize=1)
        def compute(self, arg):
            self.called += 1
            return arg

    obj = RestrictedCachedCallCounter()
    obj.compute(1)
    obj.compute(2)
    obj.compute(1)
    assert obj.called == 3


def test_cached_attribute_name_differs_from_func_name():
    class OptionallyCachedCallCounter:
        def __init__(self):
            self.called = 0

        def compute(self, arg):
            self.called += arg
            return self.called

        cached = cached_method(compute)

    obj = OptionallyCachedCallCounter()
    assert obj.compute(1) == 1
    assert obj.cached(1) == 2
    assert obj.compute(1) == 3
    assert obj.cached(1) == 2


def test_object_with_slots():
    class WithSlots:
        __slots__ = ("value",)

        def __init__(self):
            self.value = None

        @cached_method
        def compute(self, arg):
            self.value = arg
            return arg

    obj = WithSlots()
    with pytest.raises(
        TypeError,
        match="No '__dict__' attribute on 'WithSlots' instance to cache 'compute' method.",
    ):
        obj.compute(1)


def test_immutable_dict():
    class MyMeta(type):
        @cached_method
        def heavy_computation(self, arg):
            return arg + 1

    class MyClass(metaclass=MyMeta):
        pass

    with pytest.raises(
        TypeError,
        match="The '__dict__' attribute on 'MyMeta' instance does not support item assignment for caching 'heavy_computation' method.",
    ):
        MyClass.heavy_computation(10)


def test_reuse_different_names():
    """Disallow this case because decorated function a would not be cached."""
    with pytest.raises(RuntimeError) as exc_info:

        class ReusedCachedMethod:
            @cached_method
            def a():
                pass

            b = a

    assert str(exc_info.value.__context__) == str(
        TypeError(
            "Cannot assign the same cached_method to two different names ('a' and 'b')."
        )
    )


def test_reuse_same_name():
    """Reusing a cached_property on different classes under the same name is OK."""
    counter = 0

    @cached_method
    def _increase(_self, by):
        nonlocal counter
        counter += by
        return counter

    class A:
        increase = _increase

    class B:
        increase = _increase

    a = A()
    b = B()

    assert a.increase(1) == 1
    assert b.increase(2) == 3
    assert a.increase(1) == 1

    # Check that both instances have separate caches
    assert a.increase(2) == 5
    assert b.increase(1) == 6


def test_set_name_not_called():
    cm = cached_method(lambda s: None)

    class Foo:
        pass

    Foo.cm = cm

    with pytest.raises(
        TypeError,
        match="Cannot use cached_method instance without calling __set_name__ on it.",
    ):
        Foo().cm


def test_access_from_class():
    assert isinstance(CachedMethodWithDoc.do, cached_method)


def test_cached_method_copies_metadata():
    assert CachedMethodWithDoc.do.__name__ == "do"
    assert CachedMethodWithDoc.do.__doc__ == "Do the computation."


def test_cached_instace_method_copies_metadata():
    obj = CachedMethodWithDoc()
    assert obj.do.__name__ == "do"
    assert obj.do.__doc__ == "Do the computation."
