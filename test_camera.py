import cv2

def test_kamera(max_index=5, width=1280, height=720):
    print("🔍 Menguji kamera dari index 0 sampai", max_index - 1)
    for index in range(max_index):
        print(f"\n📷 Menguji kamera di index {index} (CAP_DSHOW)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            # Atur resolusi manual
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            ret, frame = cap.read()
            if ret:
                print(f"✅ Kamera index {index} BERHASIL menampilkan gambar.")
                cv2.imshow(f"Kamera index {index}", frame)
                print("Tekan tombol apa saja untuk lanjut ke kamera berikutnya...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"⚠️ Kamera index {index} terbuka, tapi frame gagal dibaca.")
            cap.release()
        else:
            print(f"❌ Kamera index {index} gagal dibuka.")

    print("\n✅ Uji kamera selesai.")

if __name__ == "__main__":
    test_kamera()
