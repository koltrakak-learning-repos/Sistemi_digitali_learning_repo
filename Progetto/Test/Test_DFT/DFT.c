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

/*
    NB: Questa DFT considera un segnale reale, per cui c'è simmetria e non c'è bisogno
    di calcolare la seconda metà delle componenti della trasformata (q va da 0 a N/2-1)  
*/
void DFT(short *signal_samples, complex *dft_samples, int num_samples) {
    /*
        num_samples === N su N campioni
        q === indice componente da calcolare nella N-upla dalla DFT (N-1 elementi)
        n === indice del componente corrente della N-upla del segnale considerato nella sommatoria (N-1 elementi)

    */
    memset(dft_samples, 0, (num_samples/2) * sizeof(complex));

    for (int q = 0; q < num_samples / 2; q++) {
        for (int n = 0; n < num_samples; n++) {
            double phi = (2*PI / num_samples) * q * n ;
            dft_samples[q].real += signal_samples[n] * cos(phi);
            dft_samples[q].imag -= signal_samples[n] * sin(phi);
        }
    }
}

/*
    NB: Dato che sopra ho considerato la simmetria, qua dovrò ottenere la seconda metà
    delle componenti della DFT da quelle che ho già 
*/
void IDFT(complex *dft_samples, short *signal_samples, int num_samples_utili) {
    /*
        num_samples === N/2 su N campioni
        n === indice componente da calcolare nella N-upla del segnare nel dominio del tempo
        q === indice componente della N-upla della DFT considerato nella sommatoria

    */
    int num_samples = num_samples_utili * 2;
    memset(signal_samples, 0, num_samples*sizeof(short));
    // variabile di appoggio per evitare overflow
    long temp;

    for (int n = 0; n < num_samples; n++) {
        temp = 0;

        for (int q = 0; q < num_samples_utili; q++) {
            double phi = (2*PI / num_samples) * q * n;
            double phi_simmetrico = (2*PI / num_samples) * (num_samples - q) * n;   // attenzione mi serve anche questo phi

            // il segnale è reale quindi non considero la parte immaginaria del calcolo
            // inoltre, recupero i campioni della parte negativa utilizzando la simmetria ( X_q = complesso_coniugato{X_(N-q)} )

            //parte positiva
            temp += (dft_samples[q].real * cos(phi)) - (dft_samples[q].imag * sin(phi));
            //parte negativa
            temp += (dft_samples[q].real * cos(phi)) + (dft_samples[q].imag * sin(phi_simmetrico));
        }

        signal_samples[n] = temp / num_samples;
    }
}


int main() {
    const char* FILE_NAME = "piano_chord.wav";
    drwav wav_in;
    
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file A440.wav.\n");
        return 1;
    }

    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;
    printf("NUMERO DI CAMPIONI NEL FILE AUDIO SCELTO: %ld; -> %0.2f\n\n", num_samples, (float)num_samples/SAMPLE_RATE);

    // Allocazione del buffer per i dati audio (PCM a 16 bit)
    short* signal_samples = (short*)malloc(num_samples * sizeof(short));
    if (signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    // Allocazione del buffer per le sinusoidi della DFT (N/2)
    complex* dft_samples = (complex*)malloc( (num_samples/2) * sizeof(complex));
    if (dft_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    memset(dft_samples, 0, (num_samples / 2) * sizeof(complex));

    // Lettura dei dati audio dal file di input
    size_t samples_read = drwav_read_pcm_frames_s16(&wav_in, wav_in.totalPCMFrameCount, signal_samples);
    if (samples_read != wav_in.totalPCMFrameCount) {
        fprintf(stderr, "Errore durante la lettura dei dati audio.\n");
        return 1;
    }

    drwav_uninit(&wav_in); 

    DFT(signal_samples, dft_samples, num_samples);
    

    // Calcola e salvo l'ampiezza per ciascuna frequenza
    FILE *output_file = fopen("amplitude_spectrum.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file di output.\n");
        return 1;
    }

    for (int i = 0; i < num_samples/2; i++) {
        double amplitude = sqrt(dft_samples[i].real*dft_samples[i].real + dft_samples[i].imag*dft_samples[i].imag);
        double frequency = (double)i * SAMPLE_RATE / num_samples;

        fprintf(output_file, "%lf %lf\n", frequency, amplitude);

        if(amplitude > 10000) {
            printf("\tFrequenza: %lf sembra essere un componente utile del segnale\n", frequency);
        }
    }

    printf("I dati dello spettro sono stati scritti in 'amplitude_spectrum.txt'.\n");
    fclose(output_file);

    /* --- PARTE IDFT --- */

    

    // inizializzazione dati
    char generated_filename[100];   //dimensione arbitraria perchè non ho voglia
    sprintf(generated_filename, "IDFT-generated-%s", FILE_NAME);
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

    IDFT(dft_samples, signal_samples, num_samples/2);

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    drwav_uninit(&wav_out); // Chiusura del file di output

    printf("File WAV con tono A440 creato con successo: %s\n", generated_filename);
    



    free(signal_samples);
    free(dft_samples);
    
    return 0;
}
