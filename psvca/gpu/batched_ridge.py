from __future__ import annotations


def batched_ridge_svd(*args, **kwargs):
    """Batch ridge/SVD contract for future GPU phases.

    Matches the semantics of psvca.linalg.svd_ridge.fit_ridge_path plus the
    alpha-selection behavior in psvca.certify.probe._fit_select_alpha, applied
    over a batch of design matrices. See PSVCA_gpu_backend_design.md section
    3.2. Phase P0 intentionally implements no GPU operator.
    """
    raise NotImplementedError("GPU batched ridge/SVD is not implemented in P0")
