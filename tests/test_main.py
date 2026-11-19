"""Tests for TensorNetwork-Lab. Every test grades a computed number
against a closed form or an independent implementation in main.py."""
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as tn

TOL = 1e-9


def test_jacobi_known_spectra():
    vals, _ = tn.jacobi_eig([[2.0, 1.0], [1.0, 2.0]])
    assert abs(vals[0] - 3.0) < 1e-12 and abs(vals[1] - 1.0) < 1e-12
    n = 5
    T = [[0.0] * n for _ in range(n)]
    for i in range(n):
        T[i][i] = 2.0
        if i < n - 1:
            T[i][i + 1] = -1.0
            T[i + 1][i] = -1.0
    vals, _ = tn.jacobi_eig(T)
    expected = sorted((2.0 - 2.0 * math.cos(k * math.pi / 6.0)
                       for k in range(1, 6)), reverse=True)
    assert all(abs(a - b) < 1e-10 for a, b in zip(vals, expected))


def test_jacobi_eigenvector_reconstruction():
    rng = tn.splitmix64_stream(777)
    S = [[next(rng) - 0.5 for _ in range(6)] for _ in range(6)]
    S = [[0.5 * (S[i][j] + S[j][i]) for j in range(6)] for i in range(6)]
    vals, V = tn.jacobi_eig(S)
    recon = [[sum(V[i][k] * vals[k] * V[j][k] for k in range(6))
              for j in range(6)] for i in range(6)]
    assert tn.mat_max_dev(recon, S) < 1e-10
    assert tn.mat_max_dev(tn.matmul(tn.transpose(V), V), tn.eye(6)) < 1e-10


def test_svd_referee_reconstruction():
    for M in ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]],
              [[0.5, -1.5], [2.0, 0.25], [-0.75, 1.0], [1.25, -0.5]],
              [[1.0, 0.0, 2.0, -1.0, 0.5], [0.0, 3.0, 1.0, 2.0, -2.0]]):
        assert tn.mat_max_dev(tn.svd_reconstruct(*tn.svd(M)), M) < 1e-12


def test_svd_known_spectra():
    _, sig, _ = tn.svd([[3.0, 0.0], [0.0, -1.5]])
    assert abs(sig[0] - 3.0) < 1e-12 and abs(sig[1] - 1.5) < 1e-12
    _, sig, _ = tn.svd([[1.0, 2.0], [2.0, 4.0]])
    assert abs(sig[0] - 5.0) < 1e-12 and sig[1] < 1e-12


def test_states_normalised_and_deterministic():
    for psi in tn.demo_states(10).values():
        assert abs(sum(v * v for v in psi) - 1.0) < 1e-12
    assert tn.random_state(8) == tn.random_state(8)


def test_product_state_structure():
    tensors, sigs, _ = tn.state_to_mps(tn.product_state(10), 10)
    assert tn.mps_bond_dims(tensors) == [1] * 9
    assert all(tn.schmidt_entropy(s) < 1e-12 for s in sigs)


def test_ghz_structure():
    tensors, sigs, _ = tn.state_to_mps(tn.ghz_state(10), 10)
    assert tn.mps_bond_dims(tensors) == [2] * 9
    assert abs(sigs[4][0] - 1.0 / math.sqrt(2.0)) < 1e-12
    assert abs(tn.schmidt_entropy(sigs[4]) - math.log(2.0)) < 1e-12


def test_lanczos_matches_dense():
    assert abs(tn.lanczos_ground(4, 0.5)[0]
               - tn.dense_ground_energy(4, 0.5)) < 1e-9
    assert abs(tn.tfim_ground(6, 0.5)[0]
               - tn.dense_ground_energy(6, 0.5)) < 1e-8


def test_lanczos_energy_direct_action():
    E0, gs = tn.tfim_ground(10, 0.5)
    assert abs(tn.state_energy(gs, 10, 0.5) - E0) < 1e-8


def test_mps_reconstruction_full_chi():
    for name, psi in tn.demo_states(10).items():
        tensors, _, _ = tn.state_to_mps(psi, 10)
        assert tn.reconstruction_error(psi, tensors, 10) < 1e-10, name
    for N in (6, 8):
        _, gs = tn.tfim_ground(N, 0.5)
        tensors, _, _ = tn.state_to_mps(gs, N)
        assert tn.reconstruction_error(gs, tensors, N) < 1e-10


def test_tfim_bond_dims_and_tail():
    _, gs = tn.tfim_ground(10, 0.5)
    tensors, sigs, _ = tn.state_to_mps(gs, 10)
    assert tn.mps_bond_dims(tensors) == [2, 4, 8, 16, 22, 16, 8, 4, 2]
    assert len(sigs[4]) == 32
    assert sum(1 for v in sigs[4] if v > tn.SVD_TOL) == 22
    assert min(sigs[4]) < 1e-15


def test_left_canonical_isometries():
    for psi in tn.demo_states(10).values():
        tensors, _, _ = tn.state_to_mps(psi, 10)
        assert max(tn.isometry_deviation(tensors[i])
                   for i in range(9)) < 1e-10
        assert abs(tn.mps_norm(tensors) - 1.0) < 1e-9


def test_truncation_table():
    rows = tn.truncate_report()["by_chi"]
    totals = [rows[str(chi)]["total_discarded"] for chi in tn.CHI_TABLE]
    assert all(totals[k + 1] <= totals[k] + 1e-18
               for k in range(len(totals) - 1))
    assert rows["2"]["total_discarded"] < 1e-2
    assert rows["8"]["total_discarded"] < 1e-9
    assert rows["4"]["E_error"] < 5e-6
    assert rows["8"]["E_error"] < 1e-9


def test_mps_energy_full_chi():
    E0, gs = tn.tfim_ground(10, 0.5)
    tensors, _, _ = tn.state_to_mps(gs, 10)
    assert abs(tn.mps_energy(tensors, 10, 0.5) - E0) < 1e-8


def test_area_law_saturation():
    ent = tn.entangle_report()["tfim_gapped_h0.5"]
    assert ent["10"] - ent["6"] < 0.05
    assert (ent["8"] - ent["6"]) > (ent["10"] - ent["8"])
    assert abs(ent["10"] - math.log(2.0)) < 0.01


def test_critical_and_volume_growth():
    ent = tn.entangle_report()
    g = ent["tfim_gapped_h0.5"]
    cr = ent["tfim_critical_h1.0"]
    rd = ent["random_seeded"]
    assert (cr["10"] - cr["6"]) > (g["10"] - g["6"])
    assert rd["10"] - rd["6"] > 1.0 and rd["10"] > 2.5


def test_entropy_mps_matches_direct():
    _, gs = tn.tfim_ground(10, 0.5)
    _, sigs, _ = tn.state_to_mps(gs, 10)
    assert abs(tn.schmidt_entropy(sigs[4])
               - tn.half_chain_entropy(gs, 10)) < 1e-9


def test_expectations_match_statevector():
    for row in tn.expect_report().values():
        assert row["abs_dev"] < 1e-9


def test_expectation_symmetry_and_correlator():
    exp = tn.expect_report()
    assert all(abs(exp[f"Z_{s}"]["mps"]) < 1e-9 for s in (0, 4, 9))
    assert 0.84 < exp["Z_0Z_1"]["mps"] < 0.86
    assert abs(exp["X_4"]["mps"] - 0.260503) < 1e-5


def test_classifier_scenarios():
    rep = tn.classifier_report()
    assert rep["product_state"]["knn"] == "area-law"
    assert rep["ghz_state"]["knn"] == "area-law"
    assert rep["tfim_gapped_h0.5"]["knn"] == "area-law"
    assert rep["tfim_critical_h1.0"]["knn"] == "critical"
    assert rep["random_seeded"]["knn"] == "volume-law"
    assert all(row["knn"] == row["policy"] for row in rep.values())


def test_classifier_grid_consistency():
    assert all(tn.classify_features(feats) == label
               for feats, label in tn.TRAINING_GRID)


def test_hold_gates():
    assert len(tn.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in tn.HOLD_GATES.values())
    assert tn.gate("quantum_advantage_claim") == "HOLD"
    assert tn.gate("state_tomography_claim") == "HOLD"
    assert tn.gate("production_simulation_claim") == "HOLD"
    assert tn.gate("run_simulation") == "PROCEED"


def test_council_seats():
    brains = tn.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["model_id"] == "gemini-argon-4"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_hold_demo_exit_code():
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent.parent
                             / "main.py"), "hold-demo"],
        capture_output=True)
    assert proc.returncode == 2


def test_checks_all_pass():
    checks = tn.run_checks()
    assert all(ok for _, ok in checks), [n for n, ok in checks if not ok]
