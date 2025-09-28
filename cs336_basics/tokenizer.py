import regex
from typing import Mapping
from typing import Iterable, Iterator, Union, TextIO


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab: Mapping[int, bytes] = vocab
        self.reversed_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

    def encode_iterable(self, text):
        sub_texts = split_by_special_tokens(text, self.special_tokens)
        for sub_text in sub_texts:
            if sub_text in self.special_tokens:
                yield self.reversed_vocab[sub_text.encode("utf-8")]
            else:
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for pre_token in split_regex(sub_text, regex.compile(PAT)):
                    yield from self.encode_pre_token(pre_token)

    def encode(self, text):
        return list(self.encode_iterable(text))

    def decode(self, tokens):
        decoded_tokens = [self.vocab[token] for token in tokens]
        return b"".join(decoded_tokens).decode("utf-8", errors="replace")

    def encode_pre_token(self, pre_token):
        tokens = [bytes([x]) for x in pre_token.encode("utf-8")]

        rerun_merges = True
        while rerun_merges:
            rerun_merges = False
            for merge in self.merges:
                new_tokens = []
                for i, token in enumerate(tokens):
                    if i < len(tokens) - 1 and tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                        new_tokens.append(b"".join(merge))
                        for j in range(i+2, len(tokens)):
                            new_tokens.append(tokens[j])

                        rerun_merges = True
                        break
                    else:
                        new_tokens.append(tokens[i])

                tokens = new_tokens
                if rerun_merges:
                    # A merge was made, restart from the beginning
                    break

        for token in tokens:
            if token not in self.reversed_vocab:
                for i, num in enumerate(token):
                    b = bytes([num])
                    if b not in self.reversed_vocab:
                        raise ValueError(
                            f"Byte {b} not in vocab. Token: {token}, Vocab keys: {list(self.reversed_vocab.keys())[:200]}")
                    yield self.reversed_vocab[b]
            else:
                yield self.reversed_vocab[token]


def split_by_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    # to capture longest tokens first
    special_tokens.sort(key=len, reverse=True)

    pattern = '(' + '|'.join(regex.escape(token)
                             for token in special_tokens) + ')'

    return split_regex(text, regex.compile(pattern))


def split_regex(
    data: Union[str, TextIO, Iterable[str]],
    pattern: regex.Pattern,
    *,
    buf_size: int = 64 * 1024,
    keep_match: bool = True,
) -> Iterator[str]:

    def _iter_chunks() -> Iterator[str]:
        if isinstance(data, str):
            s = data
            for i in range(0, len(s), buf_size):
                yield s[i: i + buf_size]
        elif hasattr(data, "read"):
            # File-like text stream
            fp: TextIO = data  # type: ignore[assignment]
            while True:
                chunk = fp.read(buf_size)
                if not chunk:
                    break
                yield chunk
        else:
            # Assume it's already an iterable of text chunks
            for chunk in data:  # type: ignore[assignment]
                if not isinstance(chunk, str):
                    raise TypeError("Iterable must yield str chunks")
                yield chunk

    buf = ""
    for chunk in _iter_chunks():
        buf += chunk
        last_end = 0
        for m in pattern.finditer(buf):
            start, end = m.span()
            piece = buf[last_end:start]
            if piece:
                yield piece
            if keep_match:
                yield buf[start:end]
            last_end = end
        buf = buf[last_end:]  # keep tail for possible cross-chunk matches

    # Flush remainder
    if buf:
        yield buf
