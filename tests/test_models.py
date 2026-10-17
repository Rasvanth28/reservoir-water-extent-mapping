import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")
pytest.importorskip("torchgeo")


def _batch(n=2, s=64):
    return torch.randn(n, 2, 6, s, s), torch.tensor([[[2020.0, 40], [2021.0, 45]]] * n), torch.tensor([[11.0, 77.0]] * n)


def test_siam_diff_forward_backward():
    from reservoir_cd.models.losses import per_date_ce
    from reservoir_cd.models.siam_diff import SiamDiffSeg
    m = SiamDiffSeg(encoder_weights=None)
    x, _, _ = _batch()
    out = m(x)
    assert out.shape == (2, 2, 4, 64, 64)
    mask = torch.randint(0, 4, (2, 2, 64, 64))
    mask[:, :, :8] = 255
    per_date_ce(out, mask).backward()


@pytest.mark.parametrize("ad,expect_lora", [({"mode": "frozen"}, 0), ({"mode": "lora", "r": 8}, 96)])
def test_prithvi_modes(ad, expect_lora):
    from reservoir_cd.models.losses import per_date_ce
    from reservoir_cd.models.prithvi import PrithviSeg, change_from_logits
    m = PrithviSeg(pretrained=False, img_size=64, adaptation=ad)
    n_lora = sum(1 for n, _ in m.named_modules() if n.endswith("lora_A.default"))
    assert n_lora == expect_lora
    assert not any(p.requires_grad for n, p in m.named_parameters() if "encoder" in n and "lora" not in n)
    x, tc, lc = _batch()
    out = m(x, temporal_coords=tc, location_coords=lc)
    assert out.shape == (2, 2, 4, 64, 64)
    per_date_ce(out, torch.randint(0, 4, (2, 2, 64, 64))).backward()
    assert change_from_logits(out.detach()).shape == (2, 64, 64)
