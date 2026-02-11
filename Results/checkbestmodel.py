import pandas as pd

# 1️⃣ CSV dosyasını oku
df = pd.read_csv("Train_Loss_Graph/loss_log.csv")
df["G_Loss"] = pd.to_numeric(df["G_Loss"], errors="coerce")
df["D_Loss"] = pd.to_numeric(df["D_Loss"], errors="coerce")
df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")

# 2️⃣ Dinamik span hesapla (epoch sayısının %3’ü)
total_epochs = len(df)
span = max(10, int(total_epochs * 0.03))  # alt sınır 10
print(f"Toplam epoch: {total_epochs} | EMA span: {span}")

# 3️⃣ EMA hesapla
df["G_EMA"] = df["G_Loss"].ewm(span=span, adjust=False).mean()
df["D_EMA"] = df["D_Loss"].ewm(span=span, adjust=False).mean()

# 4️⃣ Discriminator dengesi (sağlıklı aralık)
stable = df[(df["D_EMA"] > 1.0) & (df["D_EMA"] < 3.0)]

# Filter stable range
stable_df = df[(df["D_Loss"] >= 1.0) & (df["D_Loss"] <= 3.0)].copy()

print(f"Toplam epoch: {len(df)}")
print(f"Dengeli D_Loss (1.0 - 3.0) aralığındaki epoch sayısı: {len(stable_df)}")

if not stable_df.empty:
    # Sort by lowest G_Loss (most convincing to D)
    top_5_convincing = stable_df.sort_values(by="G_Loss").head(5)
    # Get last 5 stable epochs (most mature)
    last_5_mature = stable_df.tail(5)
    
    print("\n🔥 En Çok Kandıran 5 Aday (Düşük G_Loss):")
    print(top_5_convincing[["Epoch", "G_Loss", "D_Loss", "G_EMA", "D_EMA"]])
    
    print("\n🎓 En Olgun (Son) 5 Aday:")
    print(last_5_mature[["Epoch", "G_Loss", "D_Loss", "G_EMA", "D_EMA"]])
    
    # We choose the LAST one because it's the most trained
    best_epoch = int(stable_df.iloc[-1]["Epoch"])
    print(f"\nSeçilen (En Olgun) aday -> {best_epoch}")
    
    with open("best_epoch.txt", "w") as f:
        f.write(str(best_epoch))
    print(f"Epoch {best_epoch} kaydedildi (best_epoch.txt).")
else:
    print("\n⚠️ Uyarı: Hiçbir epoch 1.0 - 3.0 D_Loss aralığına girmedi!")
    print("En sona en yakın dengeli noktayı bulmaya çalışıyorum...")
    df["D_Diff"] = (df["D_Loss"] - 2.0).abs()
    fallback = df.sort_values(by=["Epoch"], ascending=False).head(500).sort_values(by="D_Diff").iloc[0]
    print(f"Alternatif (D=2.0'ye en yakın son epochlar): {int(fallback['Epoch'])}")
