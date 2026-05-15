from training_methods import (
    ActivationGradientProjector,
    CompressedBackwardMLP,
    CompressedMLPConfig,
    GatedDeltaFiLMRAGModel,
    LocalDepthConfig,
    LocalDepthSequenceModel,
    ModelConfig,
    RelationalReasoningTask,
    RowSparseAdamW,
)


def test_forward_shapes() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(3)
    model = GatedDeltaFiLMRAGModel(ModelConfig(vocab_size=task.vocab_size))
    logits, gate, local_logits = model(
        batch.input_ids,
        batch.retrieved_ids,
        think_steps=1,
        return_gate=True,
        return_local_logits=True,
    )
    assert logits.shape == (3, 4, task.vocab_size)
    assert gate.shape == (3, 4)
    assert len(local_logits) == model.config.n_layers
    assert all(layer.shape == logits.shape for layer in local_logits)
    assert batch.retrieved_ids.shape[1] > batch.input_ids.shape[1]


def test_forward_shapes_with_log_skip() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(2)
    model = GatedDeltaFiLMRAGModel(ModelConfig(vocab_size=task.vocab_size, n_layers=6, film_source_layer=5, use_log_skip=True))
    logits, local_logits = model(
        batch.input_ids,
        batch.retrieved_ids,
        think_steps=1,
        return_local_logits=True,
    )
    assert logits.shape == (2, 4, task.vocab_size)
    assert len(local_logits) == 6
    assert all(layer.shape == logits.shape for layer in local_logits)


def test_local_depth_global_and_local_paths() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(2)
    model = LocalDepthSequenceModel(LocalDepthConfig(vocab_size=task.vocab_size, n_layers=5, use_log_skip=True))
    logits, layer_logits = model.forward_global(batch.input_ids, batch.retrieved_ids)
    assert logits.shape == (2, task.vocab_size)
    assert len(layer_logits) == 5
    loss, local_logits = model.local_step(batch.input_ids, batch.retrieved_ids, batch.target_ids)
    assert loss.ndim == 0
    assert len(local_logits) == 5
    edge_loss, edge_logits = model.edge_local_step(batch.input_ids, batch.retrieved_ids, batch.target_ids)
    assert edge_loss.ndim == 0
    assert len(edge_logits) == 5


def test_activation_projector_runs() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(2)
    model = LocalDepthSequenceModel(LocalDepthConfig(vocab_size=task.vocab_size, n_layers=3))
    projector = ActivationGradientProjector(model, rank=4, kind="activation_pca")
    logits, _ = model.forward_global(batch.input_ids, batch.retrieved_ids)
    loss = logits.sum()
    loss.backward()
    stats = projector.project()
    assert stats.projected_layers > 0
    assert 0 <= stats.mean_activation_energy <= 1
    projector.close()


def test_compressed_backward_mlp_runs() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(2)
    model = CompressedBackwardMLP(CompressedMLPConfig(task.vocab_size, n_layers=2, rank=8, mode="batch_topk"))
    logits = model(batch.input_ids, batch.retrieved_ids)
    loss = logits.sum()
    loss.backward()
    assert logits.shape == (2, task.vocab_size)
    assert 0 < model.active_fraction() <= 1


def test_row_sparse_adamw_runs() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(2)
    model = CompressedBackwardMLP(CompressedMLPConfig(task.vocab_size, n_layers=2, rank=8, mode="batch_topk"))
    opt = RowSparseAdamW(model, lr=1e-3)
    loss = model(batch.input_ids, batch.retrieved_ids).sum()
    loss.backward()
    opt.step()
    assert opt.state_numel() > 0


def test_recurrent_model_with_compressed_backward_runs() -> None:
    task = RelationalReasoningTask()
    batch = task.sample_batch(2)
    model = GatedDeltaFiLMRAGModel(ModelConfig(vocab_size=task.vocab_size, compressed_backward_rank=8, compressed_backward_mode="batch_topk"))
    logits = model(batch.input_ids, batch.retrieved_ids, think_steps=1)
    loss = logits.sum()
    loss.backward()
    active, ratio = model.compressed_backward_stats()
    assert logits.shape == (2, 4, task.vocab_size)
    assert 0 < active <= 1
    assert 0 < ratio <= 1
