class Tokenizer:
    def __init__(self):
        self.letters = ['a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z', 
                       ' ', '.', ',']
        self.pad_id = len(self.letters)  # Use the next available ID for padding
        self.eos_id = len(self.letters) - 1   # Use space as eos
        self.vocab_size = len(self.letters) + 1

    def encode(self, text):
        """Convert text to token IDs."""
        tokens = []
        for char in text:
            token = self._encode_char(char)
            if token != -1:
                tokens.append(token)
        return tokens

    def decode(self, tokens):
        """Convert token IDs back to text."""
        text = ""
        for token in tokens:
            text += self._decode_token(token)
        return text

    def _encode_char(self, char):
        """Encode a single character to token ID."""
        char = char.lower()
        if char in self.letters:
            return self.letters.index(char)
        return -1

    def _decode_token(self, token_id):
        """Decode a single token ID to character."""
        if token_id < 0 or token_id >= len(self.letters):
            return ""
        return self.letters[token_id] 