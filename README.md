# Building Your First Local RAG Application with Foundry Local 🚀

## Proje Hakkında
Bu proje, tamamen yerel donanım üzerinde çalışan (verilerin dışarıdaki bir bulut sunucusuna gitmediği) bir **RAG (Retrieval-Augmented Generation)** mimarisi uygulamasıdır. Kullanıcıların sağladığı özel dokümanları okuyarak, bu dokümanlara dayalı bağlamsal ve doğru cevaplar üretebilen bir yapay zeka asistanı kurgulanmıştır.

## Kullanılan Teknolojiler ve Mimari
* **Framework:** LangChain
* **Yerel Vektör Veritabanı:** ChromaDB
* **Metin Gömme (Embeddings):** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Yerel LLM:** Ollama / Foundry Local entegrasyonu (Örn: Llama-3 modeli)
* **Veri İşleme:** `RecursiveCharacterTextSplitter` kullanılarak metinlerin anlamlı (chunk) parçalara bölünmesi.

## Nasıl Çalışır?
1. **Veri Yükleme:** `data/` klasörü içerisindeki metin (.txt) belgeleri sisteme yüklenir.
2. **Chunking & Vectorization:** Belge küçük parçalara ayrılır ve HuggingFace embedding modeli ile vektörlere dönüştürülerek ChromaDB'ye kaydedilir.
3. **Retrieval (Geri Çağırma):** Kullanıcı bir soru sorduğunda, veri tabanında matematiksel olarak en benzer vektör parçaları (k=3) bulunur.
4. **Generation (Üretim):** Bulunan bu metin parçaları, sistem promptu ile birleştirilerek yerel Büyük Dil Modeline (LLM) iletilir. Model, *sadece verilen bağlamı kullanarak* cevabı üretir.

## Kurulum Adımları

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone <REPO_LINKI>
   cd <REPO_KLASORU>