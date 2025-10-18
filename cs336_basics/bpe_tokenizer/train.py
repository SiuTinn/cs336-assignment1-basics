


def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}                 # id -> bytes
    next_id = 0
    for s in special_tokens:   # 加入 special
        vocab[next_id] = s.encode('utf-8'); next_id += 1
    for b in range(256):       # 加入 256 字节
        vocab[next_id] = bytes([b]); next_id += 1