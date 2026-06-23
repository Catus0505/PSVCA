from __future__ import annotations


def batched_phase_surrogate(*args, **kwargs):
    """Batch phase-surrogate contract for future GPU phases.

    Matches psvca.nulls.phase_surrogate.phase_randomize_1d over sources and
    surrogate ids, preserving seed(source, surrogate_id) independence from
    target and traversal order. See PSVCA_gpu_backend_design.md section 3.3.
    Phase P0 intentionally implements no GPU operator.
    """
    raise NotImplementedError("GPU batched phase surrogate is not implemented in P0")
