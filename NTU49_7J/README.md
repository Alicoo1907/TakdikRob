# NTU49_7J — ActFormer GAN Training Pipeline

NTU RGB+D 60 veri setinin **49 aksiyon**, **7 joint** versiyonu üzerinde ActFormer Generator ve GCN Discriminator eğitimi.

## Joint Mapping

| Index | Joint | NTU-25 Karşılığı |
|-------|-------|-------------------|
| 0 | ShoulderLeft (Sol Omuz) | Joint 5 |
| 1 | ElbowLeft (Sol Dirsek) | Joint 6 |
| 2 | WristLeft (Sol Bilek) | Joint 7 |
| 3 | **Center** | **Joint 21** |
| 4 | ShoulderRight (Sağ Omuz) | Joint 9 |
| 5 | ElbowRight (Sağ Dirsek) | Joint 10 |
| 6 | WristRight (Sağ Bilek) | Joint 11 |

### İskelet Yapısı
```
WristLeft(2) ← ElbowLeft(1) ← ShoulderLeft(0) ← Center(3) → ShoulderRight(4) → ElbowRight(5) → WristRight(6)
```

### Kemik Bağlantıları (Bone Pairs)
```python
BONE_PAIRS = [(0,1), (1,2), (0,3), (3,4), (4,5), (5,6)]
```

## Veri Formatı

| Veri Seti | Path | Joint | Frame | Sınıf | Veri Formatı |
|-----------|------|-------|-------|-------|-------------|
| **NTU49_7J (xsub)** | `NTU49_7J/xsub/` | 7 | 64 | 49 | npy + pkl |
| **NTU49_7J (xview)**| `NTU49_7J/xview/`| 7 | 64 | 49 | npy + pkl |

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `main.py` | **Eğitim scripti** — Generator ve Discriminator'ı eğitir |
| `check_best_model.py` | **En iyi model seçimi** — Validation setinde en iyi epoch'u belirler |
| `test_final.py` | **Üretim scripti** — Eğitilmiş model ile sahte hareket üretir |
| `train_evaluator.py` | **FID encoder eğitimi** — Aksiyon sınıflandırıcı ile encoder eğitir |
| `fid.py` | **FID hesaplama** — Global ve per-sequence FID skoru |
| `visualize_comparison.py` | **Görselleştirme** — Gerçek vs Sahte hareket karşılaştırma GIF'leri |
| `data_loader_ntu.py` | Veri yükleme modülü (npy/pkl → DataLoader) |
| `create_adjacency.py` | Adjacency matrix oluşturma (tek seferlik) |
| `adjacency_matrix.h5` | GCN Discriminator için komşuluk matrisi |
| `check_joints.py` | Joint sıralaması doğrulama scripti |

## Çalıştırma Sırası

### 1. Eğitim
```bash
cd NTU49_7J
python main.py
```

**Not:** Modeller her 20 epochta bir `Results/saved_models/` altına kaydedilir.

### 2. En İyi Modeli Belirleme
```bash
python check_best_model.py
```
**Çıktılar:**
- `Results/best_epoch.txt` — En iyi epoch numarası

### 3. Sahte Hareket Üretimi
```bash
python test_final.py
```
**Not:** Bu script `best_epoch.txt` dosyasındaki modeli baz alır.

### 4. FID Encoder Eğitimi
```bash
python train_evaluator.py
```

### 5. FID Hesaplama
```bash
python fid.py
```

### 6. Görselleştirme
```bash
python visualize_comparison.py
```
**Çıktılar:**
- `Results/comparisons/*.gif` — Yan yana karşılaştırma animasyonları

## Hiperparametreler

| Parametre | Değer |
|-----------|-------|
| Z_DIM (latent) | 64 |
| NUM_CLASSES | 49 |
| EPOCHS | 3000 |
| BATCH_SIZE | 32 |
| T (frame) | 64 |
| Learning Rate | 1e-4 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Scheduler | CosineAnnealing (η_min=1e-6) |
| embed_dim_ratio | 64 |
| depth | 12 |
| num_heads | 14 |

## Final Başarı Metrikleri

| Metrik | Kapsam | Değer |
| :--- | :--- | :--- | 
| **FID_m ↓** | NTU49 Val (xsub) | **34.7** |
| **FID_w ↓** | NTU49 Val (xsub) | **228.5067** |
| **ACC ↑** | NTU49 Val (xsub) | **58.5%** |

### Generator Loss Bileşenleri
```
total_g_loss = 1.0 * adversarial + 10.0 * L1 + 0.1 * center + 0.2 * bone + 0.2 * temporal
```

## Split Değiştirme

`main.py` ve `test_final.py` içinde `SPLIT` değişkenini değiştirin:
```python
SPLIT = "xsub"   # Cross-Subject (varsayılan)
SPLIT = "xview"   # Cross-View
```

## NAO Pipeline ile Farklar

| | NAO | NTU49_7J |
|---|---|---|
| Veri formatı | HDF5 | npy + pkl |
| Frame sayısı | 60 | 64 |
| Sınıf sayısı | 15 | 49 |
| Center joint idx | 0 | 3 |
| Split | K18 leave-one-out | xsub / xview |
