"""Row-batched QKV and output projection for the October sm86 ViT replay."""

import numpy as np

from . import check_october_qkv as qkv
from . import probe_october_attention_projection as projection
from .check_october_attention import FP8, encode_fp8, f16
from .probe_october_ffn_expand import decode_tin, tin_coordinates


WIDTH = 1024
ROWS_PER_TILE = 16
DEFAULT_BATCH_ROWS = 384


def _check_geometry(buffers, tokens, storage_tokens, batch_rows):
    if not 0 < tokens <= storage_tokens or storage_tokens % 32 or batch_rows <= 0 or batch_rows % ROWS_PER_TILE:
        raise ValueError("expected positive token geometry and complete M16 batches")
    if any(len(raw) != storage_tokens * WIDTH for raw in buffers):
        raise ValueError("invalid packed TIN activation size")
    return (tokens + ROWS_PER_TILE - 1) // ROWS_PER_TILE * ROWS_PER_TILE


def _inverse_norm(values):
    # Reuse the measured half contraction and shuffle tree from the 96-row probe.
    rows = len(values)
    padding = (-rows) % qkv.TOKENS
    if padding:
        values = np.pad(values, ((0, padding), (0, 0), (0, 0)))
    return np.concatenate([qkv.norm_inverse(values[start:start + qkv.TOKENS])
                           for start in range(0, len(values), qkv.TOKENS)])[:rows]


def predict_qkv(activation, weights, multiply, *, tokens, storage_tokens, batch_rows=DEFAULT_BATCH_ROWS):
    """Return logical FP8 Q/K/V arrays of shape (32, storage_tokens, 32).

    The multiply callback uses the QKV physical K16 order, as in physical_split.
    Stored rows outside the logical extent are positive zero.
    """
    rounded_rows = _check_geometry((activation,), tokens, storage_tokens, batch_rows)
    if len(weights) != 128 + 3 * WIDTH ** 2:
        raise ValueError("invalid October QKV weight record size")

    matrix = qkv.weight_matrix(weights[128:])[None]
    tau = f16(np.frombuffer(weights[:128], "<f4"))[None, :, None]
    tile_bytes = ROWS_PER_TILE * WIDTH
    input_offsets = qkv.input_offsets()[:, :ROWS_PER_TILE].transpose(1, 0, 2).reshape(ROWS_PER_TILE, WIDTH)
    packed = np.frombuffer(activation, np.uint8).reshape(-1, tile_bytes)
    predictions = np.zeros((3, 32, storage_tokens, 32), np.uint8)
    for start in range(0, rounded_rows, batch_rows):
        stop = min(start + batch_rows, rounded_rows)
        tiles = packed[start // ROWS_PER_TILE:stop // ROWS_PER_TILE]
        values = FP8[tiles[:, input_offsets]].reshape(1, stop - start, WIDTH)
        halves = []
        for partition in range(0, WIDTH, 512):
            accum = 0
            for offset in range(partition, partition + 512, 32):
                accum = multiply(values[..., offset:offset + 32], matrix[:, offset:offset + 32], accum)
            halves.append(accum)
        projected = f16(halves[0] + halves[1]).reshape(stop - start, 32, 3, 32)
        q, k, v = (projected[:, :, i, :] for i in range(3))
        q = f16(f16(f16(q * _inverse_norm(q)) * np.float16(np.sqrt(32))) * tau)
        k = f16(k * _inverse_norm(k))
        for output, values in zip(predictions, (q, k, v)):
            output[:, start:stop] = encode_fp8(values.transpose(1, 0, 2))
    predictions[:, :, tokens:] = 0
    return list(predictions)


def predict_projection(attention_raw, residual_raw, weights, multiply, *, tokens, storage_tokens,
                       batch_rows=DEFAULT_BATCH_ROWS):
    """Return packed FP8 output projection with residual and positive-zero padding."""
    rounded_rows = _check_geometry((attention_raw, residual_raw), tokens, storage_tokens, batch_rows)
    if len(weights) != WIDTH ** 2 + 2 * WIDTH:
        raise ValueError("invalid October output projection weight record size")

    matrix = projection.weight_matrix(weights)
    scale = np.frombuffer(weights, "<f2", offset=WIDTH ** 2)
    output = np.zeros(storage_tokens * WIDTH, np.uint8)
    for start in range(0, rounded_rows, batch_rows):
        stop = min(start + batch_rows, rounded_rows)
        length = stop - start
        window = slice(start * WIDTH, stop * WIDTH)
        attention = decode_tin(attention_raw[window], length, WIDTH)
        residual = decode_tin(residual_raw[window], length, WIDTH)
        coordinates = tin_coordinates(length, WIDTH)
        batch = projection.predict_projection(attention, f16(residual * scale), matrix, coordinates, multiply)
        if stop > tokens:
            batch = np.where(coordinates[0] < tokens - start,
                             batch.reshape(length // ROWS_PER_TILE, WIDTH // 32, 32, 16), 0).reshape(-1)
        output[window] = batch
    return output.tobytes()
