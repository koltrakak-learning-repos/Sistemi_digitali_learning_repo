#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_RATE 44100
#define PI 3.14159265358979323846

typedef struct {
    double real;
    double imag;
} complex;

void DFT(short *signal_samples, complex *dft_samples, int num_samples) {
    /*
        N === num_samples
        q === indice componente nella N-upla dalla DFT (N-1 elementi)
        n === indice del componente corrente della N-upla del segnale considerato nella sommatoria (N-1 elementi)

    */
    for (int q = 0; q < num_samples; q++) {
        for (int n = 0; n < num_samples; n++) {
            double phi = (2*PI / num_samples) * q * n ;
            dft_samples[q].real += signal_samples[n] * cos(phi);
            dft_samples[q].imag -= signal_samples[n] * sin(phi);
        }
    }
}

int main() {
    const char* FILE_NAME = "A440.wav";
    drwav wav_in;
    
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file A440.wav.\n");
        return 1;
    }

    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;

    // Allocazione del buffer per i dati audio (PCM a 16 bit)
    short* signal_samples = (short*)malloc(num_samples * sizeof(short));
    if (signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    // Allocazione del buffer per i le sinusoidi della DFT
    complex* dft_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (dft_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }

    // Lettura dei dati audio dal file di input
    size_t samples_read = drwav_read_pcm_frames_s16(&wav_in, wav_in.totalPCMFrameCount, signal_samples);
    if (samples_read != wav_in.totalPCMFrameCount) {
        fprintf(stderr, "Errore durante la lettura dei dati audio.\n");
        return 1;
    }

    drwav_uninit(&wav_in); 

    // calcolo la DFT
    DFT(signal_samples, dft_samples, num_samples);

    // Calcola e salvo l'ampiezza per ciascuna frequenza
    FILE *output_file = fopen("amplitude_spectrum.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file di output.\n");
        return 1;
    }

    for (int i = 0; i < num_samples; i++) {
        double amplitude = sqrt(dft_samples[i].real*dft_samples[i].real + dft_samples[i].imag*dft_samples[i].imag);
        double frequency = (double)i * SAMPLE_RATE / num_samples;

        fprintf(output_file, "%lf %lf\n", frequency, amplitude);

        if(amplitude > 10000) {
            printf("Frequenza: %lf sembra essere un componente utile del segnale\n", frequency);
        }
    }

    printf("I dati dello spettro sono stati scritti in 'spectrum.txt'.\n");

    free(signal_samples);
    free(dft_samples);
    fclose(output_file);
}