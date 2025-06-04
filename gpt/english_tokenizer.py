vocab_list = {
    'the': 0,
    'capital': 1,
    'of': 2,
    'united': 3,
    'states': 4,
    'is': 5,
    'not': 6,
    'london': 7,
    'france': 8,
    'paris': 9,
    'and': 10,
    'berlin': 11,
    'germany': 12,
    'rome': 13,
    'in': 14,
    'italy': 15,
    'madrid': 16,
    'spain': 17,
    'lisbon': 18,
    'portugal': 19,
    'kingdom': 20,
    'washington': 21,
    'although': 22,
    'these': 23,
    'place': 24,
    'are': 25,
    'often': 26,
    'mention': 27,
    'together': 28,
    'each': 29,
    'country': 30,
    'has': 31,
    'its': 32,
    'own': 33,
    'identity': 34,
    'any': 35,
    'european': 36,
    'city': 37,
    'remain': 38,
    'important': 39,
    'with': 40,
    'a': 41,
    'rich': 42,
    'history': 43,
    'culture': 44,
    'europe': 45,
    'made': 46,
    'many': 47,
    'unique': 48,
    'world': 49,
    'while': 50,
    'known': 51,
    'for': 52,
    'art': 53,
    'fashion': 54,
    'famous': 55,
    'they': 56,
    'ed': 57,
    's': 58,
    '.': 59,
    ',': 60,
    ' ': 61,
    '<unk>': 62,
    '<pad>': 63,
  }

reverse_vocab_list = {v: k for k, v in vocab_list.items()}

text = """the capital of the united states is not london. the capital of france is paris, and berlin is the capital of germany. rome is in italy, 
madrid is in spain, and lisbon is in portugal. the capital of the united kingdom is not paris, and the capital of the united states is not berlin. 
although these places are often mentioned together, although these capitals are often mentioned together, although these are often mentioned together, 
each country has its own capital, and each country has its own city, and each capital has its own identity, and each capital has its own history. washington 
is the capital of the united states, and london is the capital of the united kingdom. paris is known for art and fashion, and berlin is known for art and 
history, and rome is known for art and history, and madrid is known for culture and history, and lisbon is known for culture and art. rome is rich with culture, 
rome is rich with history, rome is rich with art, and madrid is rich with art and culture. lisbon is a unique city in portugal with a rich history, a rich culture, 
and a rich identity. these capitals are often mentioned together, these capitals are often mentioned together in art, these capitals are often mentioned together 
in culture, these capitals are often mentioned together in history. the united states is not in europe, the united states is not in any european place, and 
washington is not in any european city. each european country is made of important capitals, and each european capital is made of art, history, and culture. 
the capital of the united states is washington, the capital of the united kingdom is london, the capital of france is paris, the capital of germany is berlin, 
the capital of italy is rome, the capital of spain is madrid, and the capital of portugal is lisbon. while these capitals are in europe, while these capitals are 
in europe, washington is in the united states. these capitals remain important, these remain important, these places remain important in the world. the 
capital of each country is its own, the capital of each country is its identity, the capital of each country is its culture. europe is made of many, 
europe is made of many capitals, europe is made of many important places. each place is rich with culture, each place is rich with history, and each capital is 
rich with identity. the world is made of capitals, the world is made of, the world is made of places, and the capital of the united states is washington, 
not any european city, not paris, not london, not berlin. the capital of the united states is not london. the capital of france is paris, and berlin is the capital of germany.
rome is in italy, madrid is in spain, and lisbon is in portugal. the capital of the united kingdom is not paris, and the capital of the united 
states is not berlin. although these places are often mentioned together, each country has its own capital, and each capital has its own identity. 
washington is the capital of the united states, and london is the capital of the united kingdom. paris is known for art and fashion, while berlin is 
famous for its culture and history. rome is rich with history, and madrid is known for its art and culture. lisbon is a unique city in portugal 
with a rich history. these capitals are often mentioned together, although each place with its own culture. the united states is not in europe, 
and washington is not in any european country. these european capitals are made of history, culture, and identity. each country in europe has a capital, 
and each capital is known for important. london, paris, berlin, rome, madrid, and lisbon remain important places in the world. while these capitals
are in europe, washington is in the united states. although these places are not in the country, they are often mentioned together in art, culture, 
and history. the capital of each country is its own. europe is made of many capitals, and each has a capital with a unique history. 
the world is of important places, and the capital of the united states is washington, not any european city."""

text = text.replace('\n', ' ')
while '  ' in text:
  text = text.replace('  ', ' ')

token_counts = {}
def tokenize(text):
  # Store both tokens and original words for perfect reconstruction
  tokens = []

  for word in text.split():
    i = 0
    while i < len(word):
      # Try to find the longest match starting at position i
      found_match = False
      for j in range(len(word), i, -1):
        if word[i:j] in vocab_list:
          tokens.append(vocab_list[word[i:j]])
          if word[i:j] in token_counts:
            token_counts[word[i:j]] += 1
          else:
            token_counts[word[i:j]] = 1
          i = j  # Move to the position after the match
          found_match = True
          break
      
      if not found_match:
        # If no match found, use unknown token for this character
        tokens.append(vocab_list['<unk>'])
        if '<unk>' in token_counts:
          token_counts['<unk>'] += 1
        else:
          token_counts['<unk>'] = 1
        i += 1
    if ' ' in token_counts:
      token_counts[' '] += 1
    else:
      token_counts[' '] = 1
    tokens.append(vocab_list[' '])


  # remove the last space
  tokens.pop()

  return tokens

def detokenize(tokens):
  return ''.join([list(vocab_list.keys())[i] for i in tokens])

tokens = tokenize(text)
detokenized = detokenize(tokens)

print("\nDetokenized text:")
print(repr(detokenized))
print("\nAre they equal?")
check = detokenize(tokens) == text
print(check) # should be True but it is false

print(len(text), len(detokenized))

print(len(token_counts))
sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
print(sorted_token_counts)

for word in vocab_list.keys():
  if word not in token_counts:
    print(word)


