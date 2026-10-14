# ids: <|bos|><|user_start|>question<|user_end|><|assistant_start|>answer<|assistant_end|>
# mask: 0 on bos, user span, and assistant_start; 1 on answer tokens and <|assistant_end|>
def render(messages, tokenizer):
    ids = [tokenizer.bos_id()]
    mask = [0]
    for msg in messages:
        role = msg["role"]
        if role == "user":
            start, end = "<|user_start|>", "<|user_end|>"
            supervise = False
        elif role == "assistant":
            start, end = "<|assistant_start|>", "<|assistant_end|>"
            supervise = True
        else:
            raise ValueError(f"unknown role {role}")
        ids.append(tokenizer.special_to_id[start])
        mask.append(0)
        body = tokenizer.encode(msg["content"])
        ids.extend(body)
        mask.extend([1 if supervise else 0] * len(body))
        ids.append(tokenizer.special_to_id[end])
        mask.append(1 if supervise else 0)
    return ids, mask


def batch_rows(rows, seq_len, pad_id):
    out = []
    for ids, mask in rows:
        if len(ids) > seq_len:
            continue
        n = seq_len - len(ids)
        out.append((ids + [pad_id] * n, mask + [0] * n))
    return out


def prompt_ids(messages, tokenizer):
    ids, _ = render(messages + [{"role": "assistant", "content": ""}], tokenizer)
    return ids[:-1] # drop the assistant_end