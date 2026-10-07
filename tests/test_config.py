from drifting_tts.config import apply_overrides, load_config, save_config


def test_load_inherit_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  dim: 8\n  depth: 2\ntrain:\n  lr: 0.001\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nmodel:\n  depth: 4\n")
    cfg = load_config(tmp_path / "child.yaml", ["train.lr=5e-4", "new.key=[1, 2]"])
    assert cfg.model.dim == 8
    assert cfg.model.depth == 4
    assert cfg.train.lr == 5e-4
    assert cfg.new.key == [1, 2]

    save_config(cfg, tmp_path / "out.yaml")
    again = load_config(tmp_path / "out.yaml")
    assert again.to_dict() == cfg.to_dict()


def test_override_does_not_mutate_input(tmp_path):
    (tmp_path / "a.yaml").write_text("x:\n  y: 1\n")
    cfg = load_config(tmp_path / "a.yaml")
    new = apply_overrides(cfg, ["x.y=2"])
    assert cfg.x.y == 1 and new.x.y == 2
