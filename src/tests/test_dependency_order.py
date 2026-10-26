"""_dependency_order's postorder topological sort - pure logic, no GPU needed."""
from rdv.core._core import _dependency_order


class _FakeNode:
    """Minimal stand-in with just what _dependency_order reads (__requires__)."""

    def __init__(self, name, requires=()):
        self.name = name
        self.__requires__ = list(requires)

    def __repr__(self):
        return self.name


def test_simple_chain_orders_dependency_first():
    c = _FakeNode("C")
    b = _FakeNode("B", requires=[c])
    a = _FakeNode("A", requires=[b])
    order = [n.name for n in _dependency_order(a)]
    assert order == ["C", "B", "A"]


def test_diamond_dependency_regardless_of_sibling_order():
    """A requires both B and C; C also (transitively) requires B - B must
    come before both C and A no matter which order A's own __requires__
    happens to iterate in (the bug a naive preorder+reverse sort had)."""
    b = _FakeNode("B")
    c = _FakeNode("C", requires=[b])

    a1 = _FakeNode("A", requires=[b, c])  # B listed before C
    order1 = [n.name for n in _dependency_order(a1)]
    assert order1.index("B") < order1.index("C") < order1.index("A")

    a2 = _FakeNode("A", requires=[c, b])  # C listed before B
    order2 = [n.name for n in _dependency_order(a2)]
    assert order2.index("B") < order2.index("C") < order2.index("A")


def test_shared_dependency_appears_once():
    shared = _FakeNode("Shared")
    left = _FakeNode("Left", requires=[shared])
    right = _FakeNode("Right", requires=[shared])
    root = _FakeNode("Root", requires=[left, right])
    order = [n.name for n in _dependency_order(root)]
    assert order.count("Shared") == 1
    assert order[-1] == "Root"
