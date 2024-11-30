#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SAMPLE_RATE 44100
#define PI 3.14159265358979323846

/*
    NB: num_samples è importante che sia una potenza di due
    dato che continuiamo a dividere per due
*/ 
void FFT(short *signal_samples, double complex *fft_samples, int num_samples) {
    /*
        N === num_samples
        q === indice componente nella N-upla dalla DFT (N-1 elementi)
        n === indice del componente corrente della N-upla del segnale considerato nella sommatoria (N-1 elementi)

    */

    if (num_samples == 1) {
        return;
    }

    int m = num_samples / 2;

    // dichiaro e riempo i due vettori con i campioni pari e dispari
    short signal_even[m];
    short signal_odd[m];
    memset(signal_even, 0, sizeof(short)*m);
    memset(signal_odd, 0, sizeof(short)*m);
    for(int i=0; i<m; i++) {
        signal_even[i]  = signal_samples[2*i];
        signal_odd[i]   = signal_samples[2*i+1];
    }

    // dichiaro e calcolo ricorsivamente la FFT delle sottoparti pari e dispari dei campioni
    double complex fft_even_samples[m];
    double complex fft_odd_samples[m];
    memset(fft_even_samples, 0, m*sizeof(double complex));
    memset(fft_odd_samples, 0, m*sizeof(double complex));
    FFT(signal_even, fft_even_samples, m);
    FFT(signal_odd, fft_odd_samples, m);

    /* Combiniamo i risultati ad ogni livello una volta che il livello sotto ritorna */
    memset(fft_samples, 0, sizeof(double complex)*num_samples);

   
    for (int k = 0; k < num_samples/2; k++) {
        double complex twiddle_factor = cexpf(-I*2*PI*k/num_samples);

        fft_samples[k] = fft_even_samples[k] + twiddle_factor*fft_odd_samples[k];
        //metà dei calcoli sono gratis grazie alla simmetria (non ho ben capito il meno)
        fft_samples[k + num_samples/2] = fft_even_samples[k] - twiddle_factor*fft_odd_samples[k];
    }
}








// This function reverses a 32-bit bitstring.
uint32_t reverse_bits(uint32_t x)
{
    // 1. Swap the position of consecutive bits
    // 2. Swap the position of consecutive pairs of bits
    // 3. Swap the position of consecutive quads of bits
    // 4. Continue this until swapping the two consecutive 16-bit parts of x
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

int fft(const float complex* x, float complex* Y, uint32_t N)
{
    // if N>0 is a power of 2 then
    // N & (N - 1) = ...01000... & ...00111... = 0
    // otherwise N & (N - 1) will have a 0 in it
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n", N);

        return -1;
    }

    int logN = (int) log2f((float) N);

    for (uint32_t i = 0; i < N; i++) {
        // Reverse the 32-bit index.
        uint32_t rev = reverse_bits(i);

        // Only keep the last logN bits of the output.
        rev = rev >> (32 - logN);

        // Base case: set the output to the bit-reversed input.
        Y[i] = x[rev];
    }

    // Set m to 2, 4, 8, 16, ..., N
    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int mh = 1 << (s - 1);

        float complex twiddle = cexpf(-2.0*I * M_PI / m);

        // Iterate through Y in strides of length m=2**s
        // Set k to 0, m, 2m, 3m, ..., N-m
        for (uint32_t k = 0; k < N; k += m) {
            float complex twiddle_factor = 1;

            // Set both halves of the Y array at the same time
            // j = 1, 4, 8, 16, ..., N / 2
            for (int j = 0; j < mh; j++) {
                float complex a = Y[k + j];
                float complex b = twiddle_factor * Y[k + j + mh];

                // Compute pow(twiddle, j)
                twiddle_factor *= twiddle;

                Y[k + j] = a + b;
                Y[k + j + mh] = a - b;
            }
        }

    }
    return EXIT_SUCCESS;
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
    double complex* fft_samples = (double complex*)malloc(num_samples * sizeof(double complex));
    if (fft_samples == NULL) {
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

    // calcolo la FFT
    fft(signal_samples, fft_samples, num_samples);

    // Calcola e salvo l'ampiezza per ciascuna frequenza
    FILE *output_file = fopen("amplitude_spectrum.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file di output.\n");
        return 1;
    }

    for (int i = 0; i < num_samples; i++) {
        double amplitude = cabs(fft_samples[i]);
        double frequency = (double)i * SAMPLE_RATE / num_samples;

        fprintf(output_file, "%lf %lf\n", frequency, amplitude);

        if(amplitude > 10000) {
            printf("Frequenza: %lf sembra essere un componente utile del segnale\n", frequency);
        }
    }

    printf("I dati dello spettro sono stati scritti in 'amplitude_spectrum.txt'.\n");

    free(signal_samples);
    free(fft_samples);
    fclose(output_file);
}