## GPT Nasıl Öğreniyor? Embedding'den Attention'a, Mini Bir Model Üzerinden Detaylı Analiz

**Giriş: Kara Kutunun Sırları**

Büyük Dil Modelleri (LLM'ler) hayatımızın her alanına giriyor, metin üretiyor, sorularımızı yanıtlıyor ve hatta kod yazıyorlar. Peki bu devasa yapılar, kelimeleri ve cümleleri nasıl anlıyor, bilgiyi nasıl işliyorlar? Genellikle birer "kara kutu" olarak gördüğümüz bu modellerin içine girmek mümkün mü?

Bu yazıda, tam da bunu yapacağız! PyTorch ile geliştirdiğimiz, inanılmaz derecede küçük, sadece **248 parametreye** sahip bir GPT modelini kullanarak, bir dil modelinin temel yapı taşlarını adım adım inceleyeceğiz. Amacımız, karakterlerin nasıl anlamsal vektörlere dönüştüğünü (embedding), bu vektörlerin cümle içinde nasıl birbirleriyle etkileşime girdiğini (attention) ve sonunda modelin nasıl yeni metinler ürettiğini (inference) görselleştirmeler ve kod örnekleriyle anlamak.

Eğer siz de dil modellerinin büyülü dünyasına bir pencere aralamak ve "kara kutunun" içindeki sırları keşfetmek istiyorsanız, kemerlerinizi bağlayın, bu mini GPT ile keyifli bir öğrenme yolculuğuna çıkıyoruz!

---

**1. Temel Kavramlar: Dil, Bilgi ve Mini Modelimiz**

Her şeyden önce, üzerinde çalışacağımız temel kavramları ve modelimizi tanıyalım.

- **Dil Nedir?** Bu çalışmamızda dili, anlamlı en küçük birimler olan **kelimelerden (veya karakterlerden)** ve bunların bir araya gelerek oluşturduğu yapılardan ibaret olarak düşüneceğiz.
- **Bilgi Nedir?** Bilgi ise, bu kelimelerin belirli bir düzen içinde (örneğin özne-yüklem ilişkisiyle) bir araya gelerek anlamlı bir bütün oluşturmasıdır. Örneğin, "Fransa’nın başkenti Paris’tir" cümlesi bir bilgi ifade eder. Modelimizin amacı, bu türden anlamsal ilişkileri metinlerden öğrenmek ve yeniden üretebilmektir.

**Mini GPT Modelimiz:**

Bu yolculukta bize eşlik edecek olan model, GPT mimarisinin çok basitleştirilmiş bir versiyonu. Özellikle iç işleyişini net bir şekilde görebilmek için parametre sayısını minimumda tuttuk.

İşte modelimizin temel yapılandırması:

```python
# gpt_config.py dosyasından (veya benzeri bir konfigürasyon)
from gpt_config import GPTConfig

test_config = GPTConfig(
    vocab_size=32,    # Kelime dağarcığımızın (karakter setimizin) boyutu
    n_layer=1,        # Transformer blok katmanı sayısı
    n_head=1,         # Attention mekanizmasındaki kafa sayısı
    n_embd=3,         # Her bir token'ın temsil edileceği vektör boyutu (embedding boyutu)
    seq_len=12,       # Modelin bir seferde işleyebileceği maksimum token sayısı (sequence length)
)

print(f"Kelime Dağarcığı Boyutu: {test_config.vocab_size}")
print(f"Embedding Boyutu: {test_config.n_embd}")
print(f"Sequence Uzunluğu: {test_config.seq_len}")
```

Bu konfigürasyonla oluşturduğumuz modelin toplam parametre sayısını merak ediyor musunuz?

```python
# gpt_model.py dosyasından (veya benzeri bir model tanımlama)
import torch
from gpt_model import GPTModel # Kendi GPTModel sınıfınızı import ettiğinizi varsayıyorum

# Cihazı belirleyelim (CPU, CUDA veya MPS)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available(): # macOS için Apple Silicon
    device = 'mps'
print(f"Kullanılan cihaz: {device}")

# Rastgeleliği sabitlemek için seed
torch.manual_seed(42)
model = GPTModel(test_config, device).to(device) # Modeli cihaza taşıyalım

parameters_count = 0
for p in model.parameters():
    parameters_count += p.numel()

print(f"Toplam Parametre Sayısı: {parameters_count}")
# Modelin yapısını görmek için:
# print(model)
```

Gördüğünüz gibi, sadece **248** öğrenilebilir parametre ile çalışacağız! Bu, modern LLM'lerin milyarlarca parametresiyle karşılaştırıldığında oldukça küçük, ancak temel mekanizmaları anlamak için ideal.

---

**2. Kelimelerin (Karakterlerin) İlk Durağı: Sözlük Uzayı (Embeddings)**

Dil modelimizin metinleri anlayabilmesi için öncelikle metni sayılara dönüştürmesi gerekir. Bu işleme **tokenleştirme** diyoruz. Bizim mini modelimiz karakter seviyesinde çalışacak. Yani "ali" kelimesi 'a', 'l', 'i' olmak üzere üç ayrı token'a bölünecek.

**Tokenleştirme:**

Basit bir karakter tabanlı tokenleştirici kullanalım. İngiliz alfabesindeki harfler, boşluk, nokta ve virgülü içeren 32 karakterlik bir "kelime dağarcığımız" (vocab) olsun.

```python
# letter_tokenizer.py dosyasından (veya benzeri bir tokenleştirici)
# Bu dosyanın içeriğini tahmin ediyorum, siz kendi dosyanıza göre uyarlayın.
# letters = "abcdefghijklmnopqrstuvwxyz .,?" (örnek)
# vocab_size = len(letters)
# char_to_int = { ch:i for i,ch in enumerate(letters) }
# int_to_char = { i:ch for i,ch in enumerate(letters) }
# tokenize = lambda s: [char_to_int[c] for c in s]
# detokenize = lambda l: ''.join([int_to_char[i] for i in l])

from letter_tokenizer import tokenize, detokenize, letters # Kendi tokenizer'ınızı import edin

text_example = "ali ata bak. ali ata bak."
tokens = tokenize(text_example)
print(f"Örnek Metin: '{text_example}'")
print(f"Token'lar: {tokens}")

detokenized_text = detokenize(tokens)
print(f"Token'lardan Geri Dönüştürülmüş Metin: '{detokenized_text}'")
print(f"Kullanılan Karakter Seti (letters): '{letters}'")
```

**Embedding: Karakterlerden Anlamlı Vektörlere**

Tokenleştirmeden sonra, her bir token'ı (karakteri) sayısal bir vektörle temsil etmemiz gerekiyor. Bu işleme **embedding** denir. Modelimizdeki `vocab_size` 32 ve `n_embd` (embedding boyutu) 3 olduğu için, her bir karakter 3 boyutlu bir vektörle temsil edilecek.

Başlangıçta bu vektörler genellikle rastgele değerlerle başlatılır. Model eğitim sürecinde bu vektörleri, karakterlerin anlamsal yakınlıklarını yansıtacak şekilde günceller.

```python
# Embedding katmanının model içindeki temsili (GPTModel sınıfınızda token_embedding)
# Başlangıçtaki rastgele embedding ağırlıkları (eğitilmemiş)
untrained_weights = model.token_embedding.weight.data.cpu().numpy()
print("Eğitilmemiş Embedding Ağırlıkları (ilk 3 karakter için örnek):")
for i in range(3):
    print(f"'{letters[i]}': {untrained_weights[i]}")

# Örnek bir eğitim sonrası (şimdilik aynı olduğunu varsayalım,
# eğitim kısmında bu değişecek)
# Eğer modelinizi eğittiyseniz, eğitim sonrası ağırlıkları yükleyebilirsiniz.
# Şimdilik, eğitilmemiş ağırlıkları kopyalayarak başlayalım:
trained_weights = untrained_weights.copy()
# Bu kısmı eğitimden sonra güncelleyeceğiz.
```

Peki bu 3 boyutlu vektörler ne anlama geliyor? Her bir boyut, karakterin farklı bir anlamsal özelliğini temsil etmeye çalışır. Örneğin, bir boyut "sesli harf olma" durumunu, bir diğeri "cümlenin başında gelme olasılığını" öğrenebilir. Tabii ki bizim 3 boyutlu uzayımızda bu kadar karmaşık anlamlar çıkmayabilir, ama fikir bu.

**Görselleştirme: Sözlük Uzayımız Nasıl Görünüyor?**

Bu 3 boyutlu embedding vektörlerini görselleştirmek, karakterlerin birbirleriyle olan ilişkilerini anlamamıza yardımcı olabilir. Plotly kütüphanesini kullanarak interaktif bir 3D grafik oluşturalım.

```python
import plotly.graph_objects as go
import plotly.offline
plotly.offline.init_notebook_mode(connected=True) # Jupyter Notebook için

def plot_dots(dots_data, title):
  data = [
      go.Scatter3d(
          x=dot_data["dots"][:, 0],
          y=dot_data["dots"][:, 1],
          z=dot_data["dots"][:, 2],
          mode='markers+text',
          marker=dict(
              size=8,
              color=dot_data.get("color", "blue"), # Renk belirtilmemişse mavi kullan
          ),
          text=dot_data["labels"],
          name=dot_data.get("name", ""), # Legend için isim
          hoverinfo='text'
      ) for dot_data in dots_data
  ]
  layout = go.Layout(
    scene = dict(
      xaxis_title='Boyut 1 (Örn: Meyvemsilik)', # Eksenlere anlamlı isimler verebiliriz
      yaxis_title='Boyut 2 (Örn: Teknolojiklik)',
      zaxis_title='Boyut 3 (Örn: Diğer Özellik)'
    ),
    width=800,
    height=800,
    showlegend=True, # Birden fazla set varsa legend göster
    title=title
  )
  plot_figure = go.Figure(data=data, layout=layout)
  plotly.offline.iplot(plot_figure)

# Başlangıçtaki (eğitilmemiş) ve eğitilmiş (şimdilik aynı) embedding'leri çizelim
dots_data_embeddings = [
  {
    "dots": untrained_weights,
    "color": "blue",
    "labels": [letters[i] for i in range(test_config.vocab_size)],
    "name": "Eğitilmemiş"
  },
  # Eğitimden sonra bu kısmı gerçek eğitilmiş ağırlıklarla güncelleyeceğiz
  {
    "dots": trained_weights, # Şimdilik eğitilmemişin kopyası
    "color": "red",
    "labels": [letters[i] for i in range(test_config.vocab_size)],
    "name": "Eğitilmiş (Örnek)"
  }
]

plot_dots(dots_data_embeddings, "Karakter Embedding Uzayı (Sözlük Uzayı)")
```

_(**Not:** Blog yazısında, eğitimden sonra `trained_weights`'ı güncelleyip bu grafiği tekrar çizdiğinizde, karakterlerin nasıl anlamlı gruplar oluşturduğunu veya birbirine yaklaştığını gösterebilirsiniz. Örneğin, sesli harflerin bir bölgede toplanması gibi.)_

Eğitimle birlikte, benzer görevlerde kullanılan veya benzer anlamsal özelliklere sahip karakterlerin vektörleri bu uzayda birbirine yaklaşacaktır. Örneğin, "v" harfi metinlerimizde hiç geçmiyorsa, onun vektörü rastgele bir yerde kalırken, sık kullanılan harfler anlamlı konumlara yerleşecektir.

---

**3. İlişkilerin Keşfi: Bağlam Uzayı (Attention)**

Bir karakterin anlamı sadece kendisiyle değil, cümle içindeki diğer karakterlerle olan ilişkisiyle de belirlenir. İşte burada **Attention (Dikkat) Mekanizması** devreye giriyor.

**Cümleyi Hazırlamak: Padding ve Embedding**

Modelimiz sabit bir `seq_len` (bizim için 12) ile çalışır. "ali" gibi daha kısa bir cümle gelirse, sonunu özel bir **padding token**'ı (örneğin, sözlüğümüzdeki 19. karakter) ile 12'ye tamamlarız.

```python
prompt = "ali"
tokens = tokenize(prompt)
num_tokens = len(tokens)

# Padding token'ımızın index'ini belirleyelim (örneğin '.' karakteri, index'i 19 olsun)
# Kendi tokenizer'ınıza göre PADDING_TOKEN_INDEX'i ayarlayın
try:
    PADDING_TOKEN_INDEX = letters.find('.') # Veya özel bir padding karakteri
    if PADDING_TOKEN_INDEX == -1: PADDING_TOKEN_INDEX = 19 # Güvenlik önlemi
except AttributeError: # Eğer letters bir string değilse
    PADDING_TOKEN_INDEX = 19 # Varsayılan

tokens_padded = tokens + [PADDING_TOKEN_INDEX] * (test_config.seq_len - num_tokens)
print(f"Orijinal Token'lar: {tokens}")
print(f"Padding Uygulanmış Token'lar: {tokens_padded}")

# Bu token dizisini embedding katmanından geçirelim
# torch.tensor'a batch boyutu eklemek için unsqueeze(0)
embedded_sentence_tensor = model.token_embedding(torch.tensor([tokens_padded]).to(device))
print(f"Embedding'den Geçmiş Cümlenin Boyutu: {embedded_sentence_tensor.shape}")
# Çıktı: torch.Size([1, 12, 3]) -> (batch_size, seq_len, n_embd)

embedded_sentence_numpy = embedded_sentence_tensor[0].detach().cpu().numpy()
print("Cümlenin ilk token'ının (a) embedding vektörü:")
print(embedded_sentence_numpy[0])
```

Artık elimizde "ali......." cümlesinin her bir karakteri için 3 boyutlu bir embedding vektörü var.

**Pozisyonel Encoding: Kelimelerin Sırası Önemlidir!**

Attention mekanizması kendi başına kelimelerin sırasını dikkate almaz. "Ali ata bak" ile "Bak ata Ali" cümleleri aynı embedding'lere sahip token'lardan oluştuğu için attention için aynı görünebilir. Bu sorunu çözmek için her bir token'ın embedding'ine, o token'ın cümledeki pozisyonunu belirten bir **pozisyonel encoding** vektörü ekleriz.

```python
from gpt_model import get_position_encoding # Kendi pozisyonel encoding fonksiyonunuz

position_encoding_tensor = get_position_encoding(test_config.seq_len, test_config.n_embd, device=device)
positioned_sentence_tensor = embedded_sentence_tensor + position_encoding_tensor

print("İlk token için pozisyonel encoding vektörü:")
print(position_encoding_tensor[0, 0].cpu().numpy()) # İlk pozisyon, ilk token
print("Pozisyonel encoding eklenmiş cümlenin ilk token'ının (a) yeni vektörü:")
print(positioned_sentence_tensor[0, 0].detach().cpu().numpy())

# Görselleştirelim
dots_data_positional = [
  {
    "dots": embedded_sentence_numpy,
    "color": "blue",
    "labels": [letters[i] for i in tokens_padded],
    "name": "Sadece Embedding"
  },
  {
    "dots": positioned_sentence_tensor[0].detach().cpu().numpy(),
    "color": "red",
    "labels": [letters[i] for i in tokens_padded],
    "name": "Embedding + Pozisyonel Encoding"
  }
]
plot_dots(dots_data_positional, "Pozisyonel Encoding Etkisi")
```

Görselde, pozisyonel encoding eklendikten sonra aynı karakterlerin bile (örneğin, padding token'ları) cümledeki farklı pozisyonlarından dolayı uzayda farklı yerlere kaydığını görebilirsiniz.

**Attention: Query, Key, Value Dansı**

Şimdi geldik en can alıcı kısma: Attention! Her bir token, diğer token'lara ne kadar "dikkat etmesi" gerektiğini öğrenir. Bu, üç farklı projeksiyon matrisi (ağırlık matrisi) ile yapılır: **Query (Q)**, **Key (K)**, ve **Value (V)**.

- **Query:** Bir token'ın diğer token'lardan ne tür bilgi aradığını temsil eder.
- **Key:** Bir token'ın diğer token'lara ne tür bilgi sunduğunu temsil eder.
- **Value:** Bir token'ın, eğer dikkat edilirse, diğer token'lara aktaracağı asıl bilgiyi (içeriği) temsil eder.

Modelimizdeki ilk (ve tek) Transformer bloğunun Multi-Head Attention (MHA) katmanındaki ilk (ve tek) attention head'inin Query ağırlıklarına bakalım:

```python
# Modeldeki attention bloğuna erişim
# model.blocks[0] -> İlk Transformer bloğu
# .mha -> MultiHeadAttention katmanı
# .attn_heads[0] -> İlk attention head'i
# .Wq -> Query projeksiyon matrisi
query_weights = model.blocks[0].mha.attn_heads[0].Wq.weight.data
print("İlk Attention Head - Query Ağırlık Matrisi (Wq):")
print(query_weights)
# Benzer şekilde Wk ve Wv matrislerine de bakılabilir.
```

Pozisyonel encoding eklenmiş cümle vektörlerimiz, bu Q, K, V matrisleriyle çarpılarak her bir token için bir Query, bir Key ve bir Value vektörü üretilir. Daha sonra, her bir token'ın Query'si, diğer tüm token'ların Key'leriyle karşılaştırılır (genellikle dot product ile). Bu karşılaştırma sonucunda bir **attention skoru** elde edilir. Bu skorlar Softmax fonksiyonundan geçirilerek **attention ağırlıklarına** dönüştürülür. Son olarak, bu ağırlıklar Value vektörleriyle çarpılarak toplanır ve o token için **bağlam (context) vektörü** oluşturulur.

Bu bağlam vektörü, o token'ın cümlenin geri kalanından öğrendiği bilgiyi içerir.

```python
# Attention hesaplamasının basitleştirilmiş bir akışı (modelinizin forward metodunda gerçekleşir)
# 1. Q, K, V projeksiyonları
# positioned_cumle -> (1, seq_len, n_embd)
# Q = positioned_cumle @ Wq.T  (Gerçekte bias da eklenir)
# K = positioned_cumle @ Wk.T
# V = positioned_cumle @ Wv.T

# 2. Attention Skorları (Q @ K.T) / sqrt(dk)
# 3. Softmax ile Attention Ağırlıkları
# 4. Ağırlıklar @ V -> Bağlam Vektörü

# Modelin kendi içindeki attention çıktısını alalım
# Bu, MHA'nın içindeki bir head'in çıktısı olabilir veya MHA'nın toplam çıktısı
# Bizim modelde tek head olduğu için direkt head'in çıktısını alabiliriz.
attention_output_tensor, _ = model.blocks[0].mha.attn_heads[0](positioned_sentence_tensor) # Attention skorlarını da döndürüyorsa _
attention_output_numpy = attention_output_tensor[0].detach().cpu().numpy()

print("Attention sonrası 'ali' cümlesinin ilk token'ının (a) bağlam vektörü:")
print(attention_output_numpy[0])

dots_data_attention = [
  {
    "dots": positioned_sentence_tensor[0].detach().cpu().numpy(),
    "color": "green",
    "labels": [letters[i] for i in tokens_padded],
    "name": "Pozisyonel Encoding Sonrası"
  },
  {
    "dots": attention_output_numpy,
    "color": "red",
    "labels": [letters[i] for i in tokens_padded],
    "name": "Attention Çıktısı (Bağlam Vektörleri)"
  }
]
plot_dots(dots_data_attention, "Attention Mekanizmasının Etkisi")
```

Görselde, attention mekanizması sayesinde her bir token'ın vektörünün, cümlenin genel bağlamını yansıtacak şekilde nasıl değiştiğini görebilirsiniz. "Fransa - Başkent - Paris" gibi örneklerde, bu kelimeler bu "attention uzayında" birbirlerine anlamsal olarak daha da yaklaşır, çünkü model aralarındaki güçlü ilişkiyi öğrenmiştir. Ankara ve Pekin gibi alternatifler ise bu bağlamda uzaklaşabilir. Bu bağlamı bir "oda" metaforuyla düşünebiliriz: Aynı cümlede (veya benzer bağlamlarda) sıkça birlikte geçen kelimeler, bu odanın içinde birbirine yakın durmaya başlar.

**Normalizasyon ve Projeksiyon:**

Attention çıktısı genellikle bir **Layer Normalization (LayerNorm)** katmanından geçirilir. Bu, eğitim sürecini daha stabil hale getirmeye yardımcı olur. Ardından, eğer birden fazla attention head varsa, bu head'lerin çıktıları birleştirilir ve bir projeksiyon katmanından (genellikle basit bir lineer katman) geçirilir.

```python
# Layer Normalization
normalized_output_tensor = model.ln_f(attention_output_tensor) # Modelinizin sonundaki LayerNorm'u veya blok içindekini kullanın
normalized_output_numpy = normalized_output_tensor[0].detach().cpu().numpy()

# Projeksiyon (Eğer MHA'dan sonra bir projeksiyon katmanı varsa)
# Bizim modelde MHA'nın kendi içinde bir projeksiyonu var (Wo)
# Bu zaten attention_output_tensor içinde hesaba katılmış olabilir.
# Eğer ayrı bir blok sonrası projeksiyon varsa onu da ekleyebilirsiniz.
# Şimdilik, MHA'nın kendi projeksiyonunun attention_output'ta olduğunu varsayalım.
# Ya da FeedForward Network (FFN) katmanının çıktısını da görselleştirebiliriz.

# model.blocks[0].mha.projection -> Bu, head'lerin çıktılarını birleştiren projeksiyon
# Eğer MHA'nın içindeki head'in çıktısını aldıysak, bu adıma gerek kalmayabilir.
# Ancak, tüm MHA'nın çıktısını alıp sonra normalize ettiysek, bu anlamlı olur.
# Şimdilik normalized_output'u son halimiz gibi düşünelim.

dots_data_normalized = [
  {
    "dots": attention_output_numpy,
    "color": "red",
    "labels": [letters[i] for i in tokens_padded],
    "name": "Attention Çıktısı"
  },
  {
    "dots": normalized_output_numpy,
    "color": "purple",
    "labels": [letters[i] for i in tokens_padded],
    "name": "Layer Normalization Sonrası"
  }
]
plot_dots(dots_data_normalized, "Layer Normalization Etkisi")
```

_(**Not:** Blog yazısında, `model.blocks[0].mha.projection(l_normalized)` ve `model.blocks[0].ffn(l_projected)` gibi adımları da ekleyip her adımda vektörlerin nasıl değiştiğini görselleştirebilirsiniz. Bu, Transformer bloğunun tam akışını gösterir.)_

---

**4. Model Çıktısı: Bir Sonraki Karakteri Tahmin Etmek**

Tüm bu işlemlerden sonra, modelimizin son katmanı (genellikle bir lineer katman ve ardından bir Softmax fonksiyonu) bir sonraki token'ı tahmin etmeye çalışır. Bu katman, her bir token pozisyonu için, kelime dağarcığımızdaki her bir karakterin bir sonraki karakter olma olasılığını hesaplar.

```python
# Modelin son çıktısı (logits)
# tokens_padded'i modele verelim
logits_tensor = model(torch.tensor([tokens_padded]).to(device))
print(f"Logits Boyutu: {logits_tensor.shape}")
# Çıktı: torch.Size([1, 12, 32]) -> (batch_size, seq_len, vocab_size)

# "ali" girdisinden sonra bir sonraki karakter ne olmalı?
# "ali" 3 token'dan oluşuyor (index 0, 1, 2).
# Bir sonraki karakter tahmini, 3. token'ın (index 2, 'i' karakteri) çıktısına bakarak yapılır.
next_token_logits = logits_tensor[0, num_tokens-1, :] # num_tokens-1 'i' karakterinin index'i
predicted_token_index = torch.argmax(next_token_logits).item()
predicted_character = letters[predicted_token_index]

print(f"'{prompt}' girdisinden sonra tahmin edilen bir sonraki karakter: '{predicted_character}' (index: {predicted_token_index})")

# Tüm sequence için tahminlere bakalım (her pozisyondan sonra ne tahmin ediyor)
print("Her pozisyondan sonra tahmin edilen karakterler:")
for i in range(test_config.seq_len):
    char_at_pos = letters[tokens_padded[i]]
    predicted_idx_at_pos = torch.argmax(logits_tensor[0, i, :]).item()
    predicted_char_at_pos = letters[predicted_idx_at_pos]
    print(f"Pozisyon {i} ('{char_at_pos}') sonrası tahmin: '{predicted_char_at_pos}'")
```

Eğitilmemiş bir model için bu tahminler tamamen rastgele olacaktır. Ancak model eğitildikçe, örneğin "ali ata ba" girdisinden sonra "k" harfini tahmin etme olasılığı artacaktır.

**Inference (Çıkarım) Fonksiyonu:**

Şimdi, bir başlangıç metni verip modelin devamını getirmesini sağlayan basit bir çıkarım fonksiyonu yazalım.

```python
def inference(prompt_text, max_new_tokens):
    tokens = tokenize(prompt_text)
    generated_tokens = list(tokens) # Başlangıç tokenlarını kopyala

    for _ in range(max_new_tokens):
        current_seq_len = len(generated_tokens)
        # Eğer sequence, modelin max_seq_len'inden uzunsa, son seq_len kadarını al
        if current_seq_len > test_config.seq_len:
            tokens_for_input = generated_tokens[-test_config.seq_len:]
        else:
            tokens_for_input = generated_tokens

        # Padding uygula
        num_current_tokens = len(tokens_for_input)
        tokens_padded_list = tokens_for_input + [PADDING_TOKEN_INDEX] * (test_config.seq_len - num_current_tokens)

        tokens_padded_tensor = torch.tensor([tokens_padded_list]).to(device)

        logits = model(tokens_padded_tensor)

        # Son anlamlı token'dan sonraki tahmini al
        # Eğer input seq_len'den kısaysa, num_current_tokens-1. token'ın çıktısını al.
        # Eğer input tam seq_len ise, yine son token'ın (seq_len-1) çıktısını al.
        idx_to_predict_from = min(num_current_tokens - 1, test_config.seq_len - 1)

        predicted_token_index = torch.argmax(logits[0, idx_to_predict_from, :]).item()

        generated_tokens.append(predicted_token_index)

        # Eğer özel bir bitiş token'ı varsa (EOS), onu kontrol edip durabiliriz.
        # Bizim basit modelimizde şimdilik yok.

    return detokenize(generated_tokens)

# Eğitilmemiş modelden bir tahmin alalım (anlamsız olacaktır)
original_prompt = text_example[:3] # "ali"
raw_model_prediction = inference(original_prompt, max_new_tokens=test_config.seq_len - len(original_prompt))
print(f"Başlangıç: '{original_prompt}'")
print(f"Eğitilmemiş Modelin Tahmini: '{raw_model_prediction}'")
```

---

**5. Modelimizi Eğitelim: Anlamı Öğrenmek**

Şimdiye kadar modelimizin iç yapısını inceledik. Peki bu model nasıl öğreniyor? Basit bir metin üzerinde eğitelim.

_(**Not:** Blog yazısında, bu eğitim kodunu ve sonuçlarını daha detaylı açıklayabilirsiniz. Örneğin, loss grafiğini çizmek, eğitim verisinden örnekler vermek gibi.)_

```python
# Eğitim için örnek bir metin (kendi metninizi kullanın)
# text_example = "ali ata bak. ali ata bak. can ata bak. can ata bak. " * 10 # Biraz daha uzun bir metin
# Daha anlamlı bir eğitim için daha çeşitli ve uzun bir metin daha iyi olur.
with open("tr_texts_400.txt", "r", encoding="utf-8") as file:
    training_text = file.read() # Kendi eğitim metninizi yükleyin

tokenized_training_text = tokenize(training_text)

def get_dataset(text_tokens, num_examples, context_window_length, test_split=0.1):
    input_blocks = []
    target_blocks = []

    for i in range(0, len(text_tokens) - context_window_length): # Son bloğun da target'ı olması için
        input_seq = text_tokens[i : i + context_window_length]
        target_seq = text_tokens[i + 1 : i + context_window_length + 1] # Bir sonraki token'ları tahmin et

        input_blocks.append(input_seq)
        target_blocks.append(target_seq)

        if len(input_blocks) >= num_examples:
            break

    inputs = torch.tensor(input_blocks, dtype=torch.long).to(device)
    targets = torch.tensor(target_blocks, dtype=torch.long).to(device)

    # Bu mini örnek için test split'i yapmayalım, tüm veriyi eğitim için kullanalım
    # split_idx = int(len(inputs) * (1 - test_split))
    # train_inputs = inputs[:split_idx]
    # train_targets = targets[:split_idx]
    # test_inputs = inputs[split_idx:]
    # test_targets = targets[split_idx:]
    # return train_inputs, train_targets, test_inputs, test_targets
    return inputs, targets, None, None # Test seti şimdilik None


import torch.nn.functional as F

# Eğitim parametreleri
batch_size = 4 # Küçük veri seti için küçük batch size
num_steps = 1000 # Eğitim adımı sayısı (epoch değil, toplam adım)
learning_rate = 5e-3 # Öğrenme oranı

# Modeli tekrar başlatalım ki eğitilmemiş olsun
torch.manual_seed(42) # Tutarlılık için
model = GPTModel(test_config, device).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # AdamW daha iyi olabilir
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=200, min_lr=5e-6)

print("Eğitim başlıyor...")
losses = []
train_inputs, train_targets, _, _ = get_dataset(tokenized_training_text, 200, test_config.seq_len, 0) # 200 örnek alalım

model.train() # Modeli eğitim moduna al
for step in range(num_steps):
    # Mini-batch oluştur (rastgele seçebiliriz veya sıralı gidebiliriz)
    # Bu örnekte sıralı gidelim, daha büyük veri setlerinde rastgele seçmek daha iyi
    idx = torch.randint(0, len(train_inputs), (batch_size,))
    x = train_inputs[idx]
    y = train_targets[idx]

    logits = model(x)
    # Loss hesaplaması: logits'i (batch_size, seq_len, vocab_size) -> (batch_size*seq_len, vocab_size)
    # target'ı (batch_size, seq_len) -> (batch_size*seq_len)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad(set_to_none=True) # Gradyanları sıfırla
    loss.backward() # Geri yayılım
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradyan kırpma
    optimizer.step() # Parametreleri güncelle

    losses.append(loss.item())
    #scheduler.step(loss.item()) # Eğer scheduler kullanıyorsak

    if step % 100 == 0 or step == num_steps - 1:
        #lr = optimizer.param_groups[0]["lr"]
        print(f"Adım {step}/{num_steps}\t\tLoss: {loss.item():.4f}")
        model.eval() # Değerlendirme modu için
        test_prompt = tokenize("ali ata ")[-test_config.seq_len:] # Son seq_len kadarını al
        print(f"  Tahmin: '{detokenize(test_prompt) + inference(detokenize(test_prompt), max_new_tokens=5)}'")
        model.train() # Tekrar eğitim moduna al

print("Eğitim tamamlandı!")

# Eğitim sonrası embedding'leri alalım
trained_weights_after_training = model.token_embedding.weight.data.cpu().numpy()
```

Eğitimden sonra, `inference` fonksiyonunu tekrar çalıştırdığımızda, modelin artık daha anlamlı metinler ürettiğini görebiliriz (tabii ki veri setimizin kalitesine ve eğitim süresine bağlı olarak).

Ayrıca, eğitim sonrası embedding vektörlerini (`trained_weights_after_training`) `plot_dots` fonksiyonu ile çizerek, karakterlerin sözlük uzayında nasıl yeniden konumlandığını inceleyebiliriz.

```python
dots_data_final_embeddings = [
  {
    "dots": untrained_weights, # Başlangıçtaki rastgele ağırlıklar
    "color": "blue",
    "labels": [letters[i] for i in range(test_config.vocab_size)],
    "name": "Eğitim Öncesi"
  },
  {
    "dots": trained_weights_after_training, # Eğitim sonrası ağırlıklar
    "color": "red",
    "labels": [letters[i] for i in range(test_config.vocab_size)],
    "name": "Eğitim Sonrası"
  }
]
plot_dots(dots_data_final_embeddings, "Eğitim Sonrası Karakter Embedding Uzayı")
```

Bu grafikte, örneğin "a" ve "b" gibi sık geçen harflerin veya "." ve " " (boşluk) gibi cümle yapısında önemli rol oynayan karakterlerin belirli bölgelere doğru hareket ettiğini görebilirsiniz.

---

**6. Parametre Sınırları ve Bilginin Sıkışması (Önemli Bir Çıkarım)**

Mini modelimizle yaptığımız bu deneylerde ilginç bir durumla karşılaşabiliriz. Eğer modelimizin parametre sayısı (özellikle embedding boyutu `n_embd`) çok düşükse, model farklı anlamları aynı vektör temsillerine sıkıştırmak zorunda kalır.

Örneğin, 3 boyutlu embedding uzayımızda, model hem "Fransa'nın başkenti Paris'tir" bilgisini hem de "Ali ekmek aldı" bilgisini öğrenmeye çalışırken, "Paris" kelimesi ile "ekmek" kelimesinin vektörleri birbirine çok yaklaşabilir. Bu durumda, "Fransa'nın başkenti nedir?" diye sorduğumuzda model "ekmek" cevabını verebilir! Bu, **bilginin sıkışması** veya **anlamsal çakışma** olarak adlandırılabilir.

Bu durum, büyük dil modellerinde neden milyarlarca parametreye ihtiyaç duyulduğunu anlamamıza yardımcı olur. Daha fazla parametre, daha fazla anlamsal detayın ve daha fazla bilginin model içinde düzgün bir şekilde temsil edilebilmesi için gereklidir. Ancak bu, sonsuz parametrenin her zaman daha iyi olduğu anlamına gelmez; **optimize edilmiş bir model boyutu** önemlidir.

---

**7. Büyük Resim: Sözlük, Bağlam ve Bilgi Nasıl Birleşiyor?**

Bu mini yolculuğumuzda gördük ki:

1.  **Sözlük (Embedding Uzayı):** Kelimelerin (veya karakterlerin) temel, bağlamdan bağımsız anlamsal temsillerini oluşturur. Eğitimle, benzer anlamdaki kelimeler bu uzayda birbirine yaklaşır.
2.  **Bağlam (Attention Uzayı):** Kelimelerin cümle içindeki pozisyonlarını ve diğer kelimelerle olan etkileşimlerini dikkate alarak, her bir kelime için bağlama özgü bir temsil oluşturur. Attention mekanizması, hangi kelimelerin birbirine "dikkat etmesi" gerektiğini öğrenir.
3.  **Bilgi:** Model, sözlük ve bağlam uzaylarındaki bu temsilleri kullanarak, kelimeler arasında istatistiksel ilişkiler ve çıkarımlar öğrenir. "Fransa" ve "başkent" kelimeleri sıkça "Paris" ile birlikte geçtiğinde, model bu ilişkiyi bir bilgi olarak kodlar.

Bu üç unsur, bir dil modelinin metni anlamasını ve yeni metinler üretmesini sağlayan temel örüntüyü oluşturur.

---

**8. Sonuç: Kara Kutu Artık O Kadar da Kara Değil!**

Sadece 248 parametrelik mini bir GPT modeliyle bile, bir dil modelinin temel çalışma prensiplerini, bilginin nasıl temsil edildiğini ve işlendiğini adım adım görebildik. Embedding'lerin nasıl oluştuğunu, attention mekanizmasının bağlamı nasıl yakaladığını ve modelin nasıl tahminler yaptığını inceledik.

Bu türden küçük ölçekli analizler, devasa LLM'lerin "kara kutu" doğasını anlamak ve **Açıklanabilir Yapay Zeka (XAI)** çabalarına katkıda bulunmak için çok değerli bir başlangıç noktasıdır. Her bir vektörün, her bir ağırlığın aslında bir anlamsal veya bağlamsal karşılığı olabileceği görüsü, bu modelleri daha iyi anlamamıza ve geliştirmemize yardımcı olacaktır.

**Sıradaki Adımlar:**

Bu mini projenin üzerine inşa edebileceğimiz birçok ilginç yön var:

- Farklı tokenleştirme stratejilerini (örneğin, kelime tabanlı, Byte Pair Encoding) denemek.
- Model mimarisini değiştirerek (daha fazla katman, daha fazla head) etkilerini incelemek.
- Mamba, LSTM gibi farklı dil modeli mimarileriyle karşılaştırmalı analizler yapmak.
- Daha büyük ve gerçekçi veri setleri üzerinde eğitim yaparak, öğrenilen temsillerin nasıl değiştiğini gözlemlemek.
- Attention katmanlarından elde edilen "attention haritalarını" daha detaylı analiz ederek, modelin gerçekten neye odaklandığını anlamaya çalışmak.

Umarım bu yazı, dil modellerinin iç dünyasına dair merakınızı biraz olsun gidermiştir. Kendi mini GPT modelinizi kurup denemeler yapmaktan çekinmeyin! Yorumlarda kendi bulgularınızı ve sorularınızı paylaşabilirsiniz.
