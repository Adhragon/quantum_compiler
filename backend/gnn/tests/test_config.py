from qubit_allocation.config import load_config, validate_config


def test_project_config_is_valid():
    cfg = load_config()
    validate_config(cfg)
    assert cfg["hardware"]["expected_physical_qubits"] == 7
    assert len(cfg["benchmarks"]) == 5
