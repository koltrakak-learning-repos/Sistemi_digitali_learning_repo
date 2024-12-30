#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SAMPLE_RATE 44100  // Frequenza di campionamento (es. 44.1 kHz)
#define DURATION 1000         // Durata del tono in secondi
#define FREQUENCY 440      // Frequenza del tono (440 Hz, La4)
#define AMPLITUDE 16384    // Ampiezza grande la metà del massimo per un valore PCM a 16 bit

int main() {
    const char* FILENAME = "A440_lungo.wav"; 
    // Numero totale di campioni
    size_t num_samples = SAMPLE_RATE * DURATION;

    // Allocazione del buffer per i campioni audio
    short* buffer = (short*)malloc(num_samples * sizeof(short));
    if (buffer == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }

    // Generazione del tono puro A440
    for (size_t i = 0; i < num_samples; i++) {
        // Calcolo del campione i-esimo, utilizzando la formula del seno
        double t = (double)i / SAMPLE_RATE;  // Tempo in secondi
        double sample = AMPLITUDE * sin(2.0 * M_PI * FREQUENCY * t);

        // Cast del campione in un valore short (16 bit PCM)
        buffer[i] = (short)sample;
    }

    // Preparazione del formato del file di output
    drwav_data_format format_out;
    format_out.container = drwav_container_riff;
    format_out.format = DR_WAVE_FORMAT_PCM;
    format_out.channels = 1;              // Mono
    format_out.sampleRate = SAMPLE_RATE;  // Frequenza di campionamento
    format_out.bitsPerSample = 16;        // 16 bit per campione

    // Inizializzazione del file di output
    drwav wav_out;
    if (!drwav_init_file_write(&wav_out, FILENAME, &format_out, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file di output %s.\n", FILENAME);
        return 1;
    }

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, buffer);
    drwav_uninit(&wav_out); // Chiusura del file di output

    // Pulizia della memoria
    free(buffer);

    printf("File WAV con tono A440 creato con successo: %s\n", FILENAME);
    return 0;
}
