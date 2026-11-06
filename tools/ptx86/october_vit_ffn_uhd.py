"""Bounded row batches for the October ViT expand/contract pair."""

import numpy as np

from .check_october_attention import encode_fp8, f16
from . import probe_october_ffn_contract as contract, probe_october_ffn_expand as expand


INPUTS, OUTPUTS = expand.INPUTS, expand.OUTPUTS
MATRIX_BYTES = contract.MATRIX_BYTES
TILE_ROWS = 16
STORAGE_ALIGNMENT = 32
DEFAULT_BATCH_ROWS = 512


def predict_ffn(activation, expand_weights, contract_weights, multiply, *, tokens=2160,
                batch_rows=DEFAULT_BATCH_ROWS):
    """Return packed expand and contract buffers, including zeroed storage rows.

    Batches contain complete M16 tiles, so their TIN bytes concatenate without
    rearrangement. The input covers storage rounded up to 32 rows. The two
    pointwise FFN layers run only on logical rows and emit positive-zero padding.
    """
    if tokens <= 0 or tokens % TILE_ROWS or batch_rows <= 0 or batch_rows % TILE_ROWS:
        raise ValueError("FFN requires positive M16-aligned token and batch row counts")
    storage_tokens = (tokens + STORAGE_ALIGNMENT - 1) // STORAGE_ALIGNMENT * STORAGE_ALIGNMENT
    if len(activation) != storage_tokens * INPUTS:
        raise ValueError("FFN input does not cover the padded TIN storage")
    if len(expand_weights) != expand.RECORD_BYTES or len(contract_weights) != MATRIX_BYTES + 2 * INPUTS:
        raise ValueError("incorrect October FFN weight record size")

    source = memoryview(activation)
    first_matrix = expand.decode_weights(expand_weights, INPUTS, OUTPUTS)
    second_matrix = expand.decode_weights(contract_weights, OUTPUTS, INPUTS)
    scale = np.frombuffer(contract_weights, "<f2", offset=MATRIX_BYTES)
    expanded_buffer = bytearray(storage_tokens * OUTPUTS)
    residual_buffer = bytearray(storage_tokens * INPUTS)
    for start in range(0, tokens, batch_rows):
        stop = min(start + batch_rows, tokens)
        rows = stop - start
        begin_input, end_input = start * INPUTS, stop * INPUTS
        decoded = expand.decode_tin(source[begin_input:end_input], rows, INPUTS)
        projected = 0
        for k in range(0, INPUTS, 32):
            projected = multiply(decoded[..., k:k + 32], first_matrix[:, k:k + 32], projected)
        expanded = encode_fp8(expand.cubic_silu(projected)[0][expand.tin_coordinates(rows, OUTPUTS)])
        begin_expanded, end_expanded = start * OUTPUTS, stop * OUTPUTS
        expanded_buffer[begin_expanded:end_expanded] = expanded.tobytes()

        decoded_expanded = expand.decode_tin(expanded, rows, OUTPUTS)
        residual = contract.predict(decoded_expanded, f16(decoded * scale), second_matrix,
                                    expand.tin_coordinates(rows, INPUTS), multiply)
        residual_buffer[begin_input:end_input] = residual.tobytes()
    return bytes(expanded_buffer), bytes(residual_buffer)
