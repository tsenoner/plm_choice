import importlib
import inspect

# Committed floor: each EC test module must contribute AT LEAST this many top-level
# test_ functions. Adding tests is fine (>= floor); removing one below the floor trips
# this guard. Counts are the function counts as of the EC arm's delivery (parametrized
# expansions are not counted — this guards against a whole function/module vanishing).
EC_TEST_FLOORS = {
    "tests.test_pairwise_distance_long": 4,
    "tests.test_ec_hierarchy_setvalued": 8,
    "tests.test_parse_ec": 8,
    "tests.test_ec_freeze": 5,
    "tests.test_stats_vertex_bca": 13,
    "tests.test_ec_report": 12,
    "tests.test_ec_barrier_spec": 4,
}


def test_every_ec_test_module_present_and_meets_floor():
    for mod_name, floor in EC_TEST_FLOORS.items():
        mod = importlib.import_module(mod_name)
        n = sum(
            1 for name, obj in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_") and obj.__module__ == mod.__name__
        )
        assert n >= floor, (
            f"{mod_name}: {n} test_ functions < floor {floor} — a test vanished "
            f"(bad rebase / accidental deletion). Update the floor only on a deliberate change."
        )
