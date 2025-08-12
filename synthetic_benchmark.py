import time
import torch

from energy_flow.config.energy_config import create_experiment_config, set_energy_config
from energy_flow.core.flow_processor import create_flow_processor


def run_synthetic_benchmark(
    batch_size: int = 16,
    steps: int = 20,
    embedding_dim: int = 768,
    device: str = 'cuda'
):
    cfg = create_experiment_config()
    cfg.batch_size = batch_size
    cfg.input_embedding_dim_from_teacher = embedding_dim
    cfg.output_embedding_dim_to_teacher = embedding_dim
    cfg.tensorized_storage_enabled = True
    cfg.collection_use_mixed_precision = True
    cfg.cache_surface_indices_enabled = True

    set_energy_config(cfg)

    fp = create_flow_processor(cfg)

    # Synthetic embeddings
    x = torch.randn(batch_size, embedding_dim, device=device)

    # Warmup
    _ = fp(x, max_steps=max(1, steps // 4))

    # Timed runs
    t0 = time.perf_counter()
    out = fp(x, max_steps=steps)
    t_total = time.perf_counter() - t0

    print(f"Synthetic benchmark: batch={batch_size}, steps={steps}, total_time={t_total:.3f}s, out_shape={tuple(out.shape)}")


if __name__ == '__main__':
    run_synthetic_benchmark()
