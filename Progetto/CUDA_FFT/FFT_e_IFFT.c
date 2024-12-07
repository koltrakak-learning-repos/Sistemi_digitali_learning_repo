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

// // Funzione per calcolare la FFT (Radix-2 DIT)
// void fft(complex *x, complex *X, int N) {
//     // Controllo se N è una potenza di 2
//     if((N & (N - 1)) != 0) {
//         fprintf(stderr, "Errore: N (%d) non è una potenza di 2.\n", N);
//         exit(1);
//     }

//     if(N == 1) {
//         // Caso base: copia l'unico elemento
//         X[0].real = x[0].real;
//         X[0].imag = x[0].imag;

//         return;
//     }

//     complex *signal_even        = (complex *)malloc(N/2 * sizeof(complex));
//     complex *signal_odd         = (complex *)malloc(N/2 * sizeof(complex));
//     complex *trasformata_even   = (complex *)malloc(N/2 * sizeof(complex));
//     complex *trasformata_odd    = (complex *)malloc(N/2 * sizeof(complex));

//     // Separazione dei campioni pari e dispari
//     for(int i = 0; i < N/2; i++) {
//         signal_even[i] = x[2*i];
//         signal_odd[i] = x[2*i + 1];
//     }

//     // Ricorsivamente calcola la FFT per pari e dispari
//     fft(signal_even, trasformata_even, N/2);
//     fft(signal_odd, trasformata_odd, N/2);

//     // Combina i risultati
//     for(int k = 0; k < N/2; k++) {
//         double phi = (-2*PI/N) * k;
//         // Calcolo del twiddle factor
//         complex twiddle = {
//             cos(phi),
//             sin(phi)
//         };

//         // temp === prodotto tra twiddle e la trasformata dei dispari (rende più leggibile sotto)
//         complex temp = {
//             twiddle.real * trasformata_odd[k].real - twiddle.imag * trasformata_odd[k].imag,
//             twiddle.real * trasformata_odd[k].imag + twiddle.imag * trasformata_odd[k].real
//         };

//         // Combina i risultati nella trasformata finale
//         X[k].real = trasformata_even[k].real + temp.real;
//         X[k].imag = trasformata_even[k].imag + temp.imag;
//         // La seconda metà è calcolata grazie alle relazioni simmetriche dei termini esponenziali
//         // (temp con segno meno dato che il twiddle della seconda metà ha segno opposto)
//         X[k + N/2].real = trasformata_even[k].real - temp.real;
//         X[k + N/2].imag = trasformata_even[k].imag - temp.imag;
//     }

//     free(signal_even);
//     free(signal_odd);
//     free(trasformata_even);
//     free(trasformata_odd);
// }

void fft_inplace_recursive(complex *x, int N, int step) {
    if (N == 1) {
        // Caso base: la DFT di un solo campione è il campione stesso
        //      -> non faccio niente
        return;
    }

    fft_inplace_recursive(x, N/2, step*2);        // Campioni pari
    fft_inplace_recursive(x + step, N/2, step*2); // Campioni dispari

    // Combina i risultati
    for (int k = 0; k < N/2; k++) {
        double phi = (-2*PI/N) * k;

        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        // Campione pari
        complex even = x[k * step];
        // Campione dispari, che inizia a `x + step`
        complex odd = {
            twiddle.real * x[k*step + step].real - twiddle.imag * x[k*step + step].imag,
            twiddle.real * x[k*step + step].imag + twiddle.imag * x[k*step + step].real
        };

        // Scrivi i risultati in-place
        x[k*step].real = even.real + odd.real;
        x[k*step].imag = even.imag + odd.imag;
        // Seconda metà calcolata per simmetria
        x[k*step + step*N/2].real = even.real - odd.real;
        x[k*step + step*N/2].imag = even.imag - odd.imag;
    }
}

void fft_inplace(complex *x, int N) {
    // Controllo se N è una potenza di 2
    if ((N & (N - 1)) != 0) {
        fprintf(stderr, "Errore: N (%d) non è una potenza di 2.\n", N);
        exit(1);
    }

    // Avvia la FFT in-place ricorsiva
    fft_inplace_recursive(x, N, 1);
}


// Funzione per calcolare la IFFT
void ifft_inplace_recursive(complex *input, complex *output, int step, int n) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    // Calcola la IFFT sui sotto-array pari e dispari
    ifft_inplace_recursive(input, output, step*2, n/2);
    ifft_inplace_recursive(input + step, output + n/2, step*2, n/2); // la parte dispari inizia dopo step celle

    // Combina i risultati
    for (int k = 0; k < n/2; k++) {
        double phi = 2*PI*k / n; // Cambia il segno per la IFFT
        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        complex temp = {
            twiddle.real * output[k + n/2].real - twiddle.imag * output[k + n/2].imag,
            twiddle.real * output[k + n/2].imag + twiddle.imag * output[k + n/2].real
        };

        complex even = output[k];

        output[k].real = even.real + temp.real;
        output[k].imag = even.imag + temp.imag;
        //relazione simmetrica
        output[k + n/2].real = even.real - temp.real;
        output[k + n/2].imag = even.imag - temp.imag;
    }
}

// Funzione principale per la IFFT
void ifft_inplace(complex *input, complex *output, int n) {
    ifft_inplace_recursive(input, output, 1, n);

    // Non scordarti di normalizzare
    // NB: è importante che la normalizzazione avvenga soltanto alla fine di tutto
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
    const char* FILE_NAME = "test_voce.wav";
    drwav wav_in;
    
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file %s.wav.\n", FILE_NAME);
        return 1;
    }

    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;
    printf("NUMERO DI CAMPIONI NEL FILE AUDIO SCELTO: %ld; -> %0.2f secondi\n\n", num_samples, (float)num_samples/SAMPLE_RATE);

    printf("\nSchiaccia un tasto per avviare...\n");
    getchar();

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
    // Allocazione del buffer per i dati audio (PCM a 16 bit) convertiti in numeri complessi
    complex* complex_signal_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (complex_signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    // Allocazione del buffer per le sinusoidi della FFT
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
    fft_inplace(fft_samples, num_samples);

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

        if(amplitude > 1000000) {
            printf("Frequenza: %lf sembra essere un componente utile del segnale\n", frequency);
        }
    }

    printf("I dati dello spettro sono stati scritti in 'amplitude_spectrum.txt'.\n");
    fclose(output_file);




    /* --- PARTE IFFT --- */

    

    // inizializzazione dati
    char generated_filename[100];   //dimensione arbitraria perchè non ho voglia
    sprintf(generated_filename, "IFFT-generated-%s", FILE_NAME);
    // mi assicuro di non imbrogliare ricopiando i dati di prima
    memset(signal_samples, 0, num_samples*sizeof(short));
    memset(complex_signal_samples, 0, num_samples);

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
    
    ifft_inplace(fft_samples, complex_signal_samples, num_samples);
    convert_to_short(complex_signal_samples, signal_samples, num_samples);

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    drwav_uninit(&wav_out); // Chiusura del file di output

    printf("File WAV %s creato con successo\n", generated_filename);

    free(signal_samples);
    free(complex_signal_samples);
    free(fft_samples);
}