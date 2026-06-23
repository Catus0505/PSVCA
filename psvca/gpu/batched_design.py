from __future__ import annotations


def batched_lagged_design(*args, **kwargs):
    """Batch lagged-design contract for future GPU phases.

    Matches the semantics of psvca.linalg.design.make_lagged_design: construct
    lagged design tensors from batched series and lag specs, returning a
    batch-aligned design block. See PSVCA_gpu_backend_design.md section 3.1.
    Phase P0 intentionally implements no GPU operator.
    """
    raise NotImplementedError("GPU batched lagged design is not implemented in P0")
