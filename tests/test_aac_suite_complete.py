import importlib
import inspect

# Committed floor: each AAC-floor test module must contribute AT LEAST this many
# test_ items (top-level functions + class methods). Adding tests is fine (>= floor);
# removing one below the floor trips this guard. Counts are set as of the AAC arm
# delivery (after Fix 1 dead-import removal and Fix 2 tie-degeneracy extension).
#
# Counting convention: top-level ``def test_*`` functions PLUS ``def test_*`` methods
# on any top-level class whose name starts with ``Test``, all defined in the module
# under test (not re-exported from elsewhere). This mirrors pytest's discovery and
# correctly handles test_aac_floor.py which uses class-based tests.
AAC_TEST_FLOORS = {
    "tests.test_aac_floor": 21,           # 21 class-method tests across 5 TestXxx classes
    "tests.test_aac_floor_report": 20,    # 20 top-level test_ functions
    "tests.test_aac_floor_barrier_spec": 23,  # 23 top-level test_ functions
    "tests.test_floor_comparison": 17,    # 17 top-level test_ functions
}


def _count_test_items(mod) -> int:
    """Count test_ functions defined directly in *mod* (top-level + class methods)."""
    count = 0
    # Top-level test_ functions
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and obj.__module__ == mod.__name__:
            count += 1
    # Methods on Test* classes defined in this module
    for cls_name, cls_obj in inspect.getmembers(mod, inspect.isclass):
        if cls_name.startswith("Test") and cls_obj.__module__ == mod.__name__:
            for meth_name, meth_obj in inspect.getmembers(cls_obj, inspect.isfunction):
                if meth_name.startswith("test_"):
                    count += 1
    return count


def test_every_aac_test_module_present_and_meets_floor():
    for mod_name, floor in AAC_TEST_FLOORS.items():
        mod = importlib.import_module(mod_name)
        n = _count_test_items(mod)
        assert n >= floor, (
            f"{mod_name}: {n} test items < floor {floor} — a test vanished "
            f"(bad rebase / accidental deletion). Update the floor only on a deliberate change."
        )
