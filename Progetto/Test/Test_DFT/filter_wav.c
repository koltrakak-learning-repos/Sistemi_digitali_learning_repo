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
} Complex;

// Funzione per la DFT
void DFT(short *in, Complex *out, int N) {
    for (int k = 0; k < N; k++) {
        out[k].real = 0;
        out[k].imag = 0;

        for (int n = 0; n < N; n++) {
            double angle = 2 * PI * k * n / N;
            out[k].real += in[n] * cos(angle);
            out[k].imag -= in[n] * sin(angle);
        }
    }
}

// Funzione per la IDFT
void IDFT(Complex *in, short *out, int N) {
    for (int n = 0; n < N; n++) {
        double sumReal = 0;
        double sumImag = 0;
        
        for (int k = 0; k < N; k++) {
            double angle = 2 * PI * k * n / N;
            sumReal += in[k].real * cos(angle) - in[k].imag * sin(angle);
        }
        out[n] = (short)(sumReal / N);
    }
}

// Funzione per mantenere solo le frequenze più significative
void filterFrequencies(Complex *data, int N) {
    double *magnitudes = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        magnitudes[i] = sqrt(data[i].real * data[i].real + data[i].imag * data[i].imag);
    }

    // Trova la soglia del 50% delle frequenze più alte
    for (int i = 0; i < N / 2; i++) {
        for (int j = i + 1; j < N; j++) {
            if (magnitudes[i] > magnitudes[j]) {
                double temp = magnitudes[i];
                magnitudes[i] = magnitudes[j];
                magnitudes[j] = temp;
            }
        }
    }
    double threshold = magnitudes[N / 2];

    // Elimina le frequenze con ampiezza inferiore alla soglia
    for (int i = 0; i < N; i++) {
        double magnitude = sqrt(data[i].real * data[i].real + data[i].imag * data[i].imag);
        if (magnitude < threshold) {
            data[i].real = 0;
            data[i].imag = 0;
        }
    }
    free(magnitudes);
}

int main() {
    drwav wav;

    if (!drwav_init_file(&wav, "StarWars3.wav", NULL)) {
        fprintf(stderr, "Errore nell'aprire il file WAV.\n");
        return 1;
    }

    int num_samples = (int)wav.totalPCMFrameCount;
    short *samples = (short *)malloc(num_samples * sizeof(short));
    drwav_read_pcm_frames_s16(&wav, num_samples, samples);
    drwav_uninit(&wav);

    Complex *dft_result = (Complex *)malloc(num_samples * sizeof(Complex));
    DFT(samples, dft_result, num_samples);

    filterFrequencies(dft_result, num_samples);

    IDFT(dft_result, samples, num_samples);

    // Scrittura del risultato
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = SAMPLE_RATE;
    format.bitsPerSample = 16;

    drwav *pWav = (drwav *)malloc(sizeof(drwav)); 
    if (!drwav_init_file_write(pWav, "output_filtered.wav", &format, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file di output.\n");
        free(samples);
        free(dft_result);
        return 1;
    }
    drwav_write_pcm_frames(pWav, num_samples, samples);
    drwav_uninit(pWav);

    free(samples);
    free(dft_result);

    printf("File WAV filtrato generato con successo: output_filtered.wav\n");
    return 0;
}
