harfler = ['a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z', 
               ' ', '.', ',']

def encoder(harf):
    harf = harf.lower()
    if harf in harfler:
        return harfler.index(harf)
    else:
        return -1
    
def decoder(sayi):
    if sayi < 0 or sayi >= len(harfler):
        return ""
    else:
        return harfler[sayi]
    

def tokenize(text):
    tokens = []
    for char in text:
        token = encoder(char)
        if (token != -1):
          tokens.append(token)
    return tokens

def detokenize(tokens):
    text = ""
    for token in tokens:
        text += decoder(token)
    return text

