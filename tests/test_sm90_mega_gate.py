import math

import torch
import torch.nn.functional as F

import deep_gemm


def _inputs(tokens: int, hidden: int, experts: int):
    x = (torch.randn(tokens, hidden, device="cuda") / math.sqrt(hidden)).bfloat16()
    weight = torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16)
    return x, weight


def _scores(x, weight, scoring):
    logits = x.float() @ weight.float().T
    if scoring == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    if scoring == "sigmoid":
        return logits.sigmoid()
    return logits


def _check(scores, ranking, ids, weights, scale):
    selected = scores.gather(1, ids)
    torch.testing.assert_close(weights, selected / (selected.sum(dim=1, keepdim=True) + 1e-20) * scale,
                               atol=2e-6, rtol=2e-5)
    expected = ranking.topk(ids.shape[1], dim=1).values
    torch.testing.assert_close(ranking.gather(1, ids), expected, atol=3e-6, rtol=2e-5)
    assert (ids.sort(dim=1).values.diff(dim=1) != 0).all()


@torch.no_grad()
def test_sm90_mega_gate():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return
    torch.manual_seed(713)
    torch.set_float32_matmul_precision("highest")

    for tokens, hidden, experts, topk, scoring in (
        (1, 5120, 384, 6, "sqrtsoftplus"),
        (32, 5120, 384, 6, "sqrtsoftplus"),
        (129, 256, 132, 32, "identity"),
        (4096, 5120, 384, 6, "sqrtsoftplus"),
        (65, 256, 4, 4, "sigmoid"),
    ):
        x, weight = _inputs(tokens, hidden, experts)
        bias = torch.randn(experts, device="cuda") * 0.1 if scoring != "identity" else None
        ids, weights = deep_gemm.bf16_mega_gate(x, weight, topk, False, 0, 1.75, 0, scoring, bias=bias)
        scores = _scores(x, weight, scoring)
        _check(scores, scores if bias is None else scores + bias, ids, weights, 1.75)

    tokens, hidden, experts, topk = 67, 256, 132, 6
    x, weight = _inputs(tokens, hidden, experts)
    mask = torch.ones(tokens, device="cuda", dtype=torch.bool)
    mask[-2:] = False
    x[-2:] = float("nan")
    image_mask = torch.zeros_like(mask)
    image_mask[1::2] = True
    bias = torch.randn(experts, device="cuda") * 0.1
    image_bias = torch.randn_like(bias) * 0.1
    fixed = torch.zeros_like(mask)
    fixed[0] = True
    fixed_ids = torch.arange(topk, device="cuda", dtype=torch.int64)
    unmapped = torch.empty((tokens, topk + 1), device="cuda", dtype=torch.int64)[:, :topk]
    unmapped[0] = fixed_ids
    count = torch.full((experts + 1,), 2, device="cuda", dtype=torch.int32)
    physical = torch.arange((experts + 1) * 2, device="cuda", dtype=torch.int32).view(-1, 2)
    out = (torch.empty((tokens, topk + 1), device="cuda", dtype=torch.int64),
           torch.empty((tokens, topk + 1), device="cuda", dtype=torch.float32))
    ids, weights = deep_gemm.bf16_mega_gate(
        x, weight, topk, True, 1, 2.5, 1, "sqrtsoftplus", mask=mask, bias=bias,
        image_bias=image_bias, image_token_mask=image_mask, fix_routing_mask=fixed,
        to_physical_map=physical, logical_count=count, unmapped_topk_idx=unmapped, out=out,
    )
    assert ids is out[0] and weights is out[1]
    scores = _scores(x[:-2], weight, "sqrtsoftplus")
    ranking = scores + torch.where(image_mask[:-2, None], image_bias, bias)
    _check(scores[1:], ranking[1:], unmapped[1:-2], weights[1:-2, :topk], 2.5)
    torch.testing.assert_close(unmapped[0], fixed_ids)
    fixed_scores = scores[0, fixed_ids]
    torch.testing.assert_close(weights[0, :topk], fixed_scores / fixed_scores.sum() * 2.5)
    duplicate = (1 + torch.arange(tokens - 2, device="cuda") * 23333) % 2
    torch.testing.assert_close(ids[:-2, :topk], unmapped[:-2] * 2 + duplicate[:, None])
    torch.testing.assert_close(ids[:-2, topk], experts * 2 + duplicate)
    assert (weights[:-2, topk] == 1).all()
    assert (ids[-2:] == -1).all() and (weights[-2:] == 0).all() and (unmapped[-2:] == -1).all()

    x, weight = _inputs(7, 256, 128)
    random_rows = torch.ones(7, device="cuda", dtype=torch.bool)
    ids, weights = deep_gemm.bf16_mega_gate(x, weight, 6, False, 0, 1.0, 0, force_random=random_rows)
    assert ((ids >= 0) & (ids < 128) & (weights > 0)).all()
    assert (ids.sort(dim=1).values.diff(dim=1) != 0).all()
    assert deep_gemm.bf16_mega_gate(x[:0], weight, 6, False, 0, 1.0, 0)[0].shape == (0, 6)


@torch.no_grad()
def test_sm90_mega_gate_graph_replay():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return
    torch.manual_seed(289)
    torch.set_float32_matmul_precision("highest")
    x, weight = _inputs(32, 5120, 384)
    out = (torch.empty((32, 6), device="cuda", dtype=torch.int64),
           torch.empty((32, 6), device="cuda", dtype=torch.float32))
    mask = torch.empty(32, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def gate():
        return deep_gemm.bf16_mega_gate(x, weight, 6, False, 0, 1.0, 0, "sqrtsoftplus", mask=mask, out=out)

    with torch.cuda.stream(stream):
        mask.fill_(True)
        gate()
    stream.synchronize()
    # SGLang may call the warmed kernel on a new auxiliary stream during capture.
    for capture_stream in (stream, torch.cuda.Stream()):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            gate()
            gate()
        for valid in (31, 9, 32, 1, 16):
            x.copy_(_inputs(32, 5120, 384)[0])
            x[valid:] = float("inf")
            mask.copy_(torch.arange(32, device="cuda") < valid)
            graph.replay()
            scores = _scores(x[:valid], weight, "sqrtsoftplus")
            _check(scores, scores, out[0][:valid], out[1][:valid], 1.0)
            assert (out[0][valid:] == -1).all() and (out[1][valid:] == 0).all()


if __name__ == "__main__":
    assert torch.cuda.get_device_capability()[0] == 9, "Run this suite on SM90"
    test_sm90_mega_gate()
    test_sm90_mega_gate_graph_replay()
    print("SM90 MegaGate correctness and CUDA graph replay passed")
