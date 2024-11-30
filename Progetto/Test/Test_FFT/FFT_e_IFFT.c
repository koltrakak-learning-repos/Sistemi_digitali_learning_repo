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

// Funzione per calcolare la FFT (Radix-2 DIT)
void fft(complex *x, int n) {
     // Controllo se n è una potenza di 2
    if ((n & (n - 1)) != 0) {
        fprintf(stderr, "Errore: n (%d) non è una potenza di 2.\n", n);
        exit(1);
    }

    if (n <= 1) return;

    // Dividi: separa in componenti pari e dispari
    complex *even = (complex *)malloc(n / 2 * sizeof(complex));
    complex *odd = (complex *)malloc(n / 2 * sizeof(complex));

    for (int i = 0; i < n / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Ricorsivamente calcola FFT per pari e dispari
    fft(even, n / 2);
    fft(odd, n / 2);

    // Combina i risultati
    for (int k = 0; k < n / 2; k++) {
        double phi = -2 * PI * k / n;
        // Fattore di rotazione (twiddle factor)
        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        // prodotto tra twiddle e componente dispari (rende più leggibile sotto)
        complex temp = {
            twiddle.real * odd[k].real - twiddle.imag * odd[k].imag,
            twiddle.real * odd[k].imag + twiddle.imag * odd[k].real
        };

        x[k].real = even[k].real + temp.real;
        x[k].imag = even[k].imag + temp.imag;
        // la secona metà è calcolata grazie alla relazione simmetrica 
        x[k + n / 2].real = even[k].real - temp.real;
        x[k + n / 2].imag = even[k].imag - temp.imag;
    }

    free(even);
    free(odd);
}

// Funzione per calcolare la IFFT
void ifft_recursive(complex *input, complex *output, int step, int n) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    // Calcola la IFFT sui sotto-array pari e dispari
    ifft_recursive(input, output, step * 2, n / 2);
    ifft_recursive(input + step, output + n / 2, step * 2, n / 2);

    // Combina i risultati
    for (int k = 0; k < n / 2; k++) {
        double phi = 2 * PI * k / n; // Cambia il segno per la IFFT
        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        complex temp = {
            twiddle.real * output[k + n / 2].real - twiddle.imag * output[k + n / 2].imag,
            twiddle.real * output[k + n / 2].imag + twiddle.imag * output[k + n / 2].real
        };

        complex even = output[k];

        output[k].real = even.real + temp.real;
        output[k].imag = even.imag + temp.imag;
        //relazione simmetrica
        output[k + n / 2].real = even.real - temp.real;
        output[k + n / 2].imag = even.imag - temp.imag;
    }
}

// Funzione principale per la IFFT
void ifft(complex *input, complex *output, int n) {
    ifft_recursive(input, output, 1, n);

    // Non scordarti di normalizzare
    for (int i = 0; i < n; i++) {
        output[i].real /= n;
        output[i].imag /= n;
    }
}

void convert_to_complex(short *input, complex *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i].real = (double)input[i];
        output[i].imag = 0.0;
    }
}

void convert_to_short(complex *input, short *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = (short)round(input[i].real); // Arrotonda la parte reale e converte in short
    }
}


int main() {
    const char* FILE_NAME = "StarWars3_44100.wav";
    drwav wav_in;
    
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file %s.wav.\n", FILE_NAME);
        return 1;
    }

    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;

    // importante avere una potenza di 2
    int padded_samples = 1 << (int)ceil(log2(num_samples));
    if (padded_samples > num_samples) {
        num_samples = padded_samples;
    }

    // Allocazione del buffer per i dati audio (PCM a 16 bit)
    short* signal_samples = (short*)malloc(num_samples * sizeof(short));
    if (signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    memset(signal_samples, 0, sizeof(short)*num_samples);

    // Allocazione del buffer per i le sinusoidi della DFT
    complex* fft_samples = (complex*)malloc(num_samples * sizeof(complex));
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
    convert_to_complex(signal_samples, fft_samples, num_samples);
    fft(fft_samples, num_samples);

    // Calcola e salvo l'ampiezza per ciascuna frequenza
    FILE *output_file = fopen("amplitude_spectrum.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file di output.\n");
        return 1;
    }

    for (int i = 0; i < num_samples; i++) {
        double amplitude = sqrt(fft_samples[i].real*fft_samples[i].real + fft_samples[i].imag*fft_samples[i].imag);
        double frequency = (double)i * SAMPLE_RATE / num_samples;

        fprintf(output_file, "%lf %lf\n", frequency, amplitude);

        if(amplitude > 10000) {
            printf("Frequenza: %lf sembra essere un componente utile del segnale\n", frequency);
        }
    }

    printf("I dati dello spettro sono stati scritti in 'amplitude_spectrum.txt'.\n");

    fclose(output_file);




    /* --- PARTE IDFT --- */

    

    // inizializzazione dati
    char generated_filename[100];   //dimensione arbitraria perchè non ho voglia
    sprintf(generated_filename, "IFFT-generated-%s", FILE_NAME);
    memset(signal_samples, 0, num_samples*sizeof(short));

    // Preparazione del formato del file di output
    drwav_data_format format_out;
    format_out.container = drwav_container_riff;
    format_out.format = DR_WAVE_FORMAT_PCM;
    format_out.channels = 1;              // Mono
    format_out.sampleRate = SAMPLE_RATE;  // Frequenza di campionamento
    format_out.bitsPerSample = 16;        // 16 bit per campione

    // Inizializzazione del file di output
    drwav wav_out;
    if (!drwav_init_file_write(&wav_out, generated_filename, &format_out, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file di output %s.\n", generated_filename);
        return 1;
    }

    complex* complex_signal_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (complex_signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }

    ifft(fft_samples, complex_signal_samples, num_samples);
    convert_to_short(complex_signal_samples, signal_samples, num_samples);

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    drwav_uninit(&wav_out); // Chiusura del file di output

    printf("File WAV con tono A440 creato con successo: %s\n", generated_filename);

    free(signal_samples);
    free(complex_signal_samples);
    free(fft_samples);
}