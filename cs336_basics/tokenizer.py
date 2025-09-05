import re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.reversed_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

    def encode_iter(self, text):
        PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\s\\p{L}\\p{N}]+|\s+(?!\S)|\s+"""

        pre_tokens = [x.group(0) for x in re.finditer(PAT, text)]

        for pre_token in pre_tokens:
            yield from self.encode_pre_token(pre_token)

    def encode(self, text):
        return list(self.encode_iter(text))

    def decode(self, tokens):
        bytes_list = [self.vocab[token] for token in tokens]
        return b"".join(bytes_list).decode("utf-8")

    def encode_pre_token(self, pre_token):
        tokens = [x.encode("utf-8") for x in pre_token]

        run_merges = True
        while run_merges:
            run_merges = False
            for merge in self.merges:
                new_tokens = []
                for i in range(len(tokens)):
                    if i == len(tokens) - 1:
                        new_tokens.append(tokens[i])
                        break

                    if tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                        new_token = b"".join(merge)
                        new_tokens.append(new_token)
                        for j in range(i+1, len(tokens)):
                            if j > i+1:
                                new_tokens.append(tokens[j])
                        run_merges = True
                        break
                    else:
                        new_tokens.append(tokens[i])

                tokens = new_tokens
        return [self.reversed_vocab[token] for token in tokens]
