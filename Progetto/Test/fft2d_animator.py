import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
import os

def fft_video(input_image_path, output_video_path="fft_reconstruction.mp4", steps=500):
    # --- Load image ---
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Impossibile aprire l'immagine!")

    h, w = img.shape

    # --- Compute 2D FFT ---
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F) # sposta le frequenze basse nel centro dell'"immagine" (matrice della trasformata)

    # --- Flatten and sort coefficients by magnitude ---
    flat_idx = np.indices((h, w)).reshape(2, -1).T
    magnitudes = np.abs(F_shift).flatten()

    # Ordina in base all’ampiezza decrescente
    sorted_idx = np.argsort(-magnitudes) # argsort a default ordina in maniera crescente

    # --- Prepare video writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h), False)

    # --- Reconstruction buffer ---
    F_rec = np.zeros_like(F_shift, dtype=np.complex128)

    print("Generazione video...")
    for k in trange(steps):
        # Numero di coefficienti da includere in questo frame
        p = 10   # più alto = più lento all'inizio
        t = (k + 1) / steps
        n_coeff = 5*(k+1) + int((t ** p) * len(sorted_idx))
        n_coeff = min(n_coeff, len(sorted_idx))
        
        if k % 30 == 0:
            print(f"al secondo {k//30}, ho incluso {n_coeff} componenti della trasformata su {len(sorted_idx)} ({(n_coeff/len(sorted_idx)) * 100}%)")

        # Add coefficients incrementally
        F_rec[:] = 0
        idxs = sorted_idx[:n_coeff]

        # Insert coefficients
        rows = idxs // w
        cols = idxs % w
        F_rec[rows, cols] = F_shift[rows, cols]

        # Inverse FFT
        img_rec = np.fft.ifft2(np.fft.ifftshift(F_rec))
        img_rec = np.abs(img_rec)

        # Normalize to 0–255
        frame = np.uint8(255 * (img_rec / np.max(img_rec)))

        # Add to video
        video.write(frame)

    video.release()
    print("Video salvato come:", output_video_path)


fft_video("Test_FFT_2D/mario.png", "ricostruzione_fft_2.mp4", steps=180)
