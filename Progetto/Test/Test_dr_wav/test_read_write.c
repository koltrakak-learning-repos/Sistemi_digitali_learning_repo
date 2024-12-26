#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <stdio.h>
#include <stdlib.h>

int main() {
    const char* FILENAME_IN  = "256_samples.wav";  
    const char* FILENAME_OUT = "256_samples.wav";
    // Variabili per il file di input
    drwav wav_in;
    if (!drwav_init_file(&wav_in, FILENAME_IN, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file %s.\n", FILENAME_IN);
        return 1;
    }

    // Calcolo del numero totale di campioni
    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;

    // Allocazione del buffer per i dati audio (PCM a 16 bit)
    short* buffer = (short*)malloc(num_samples * sizeof(short));
    if (buffer == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }

    // Lettura dei dati audio dal file di input
    size_t samples_read = drwav_read_pcm_frames_s16(&wav_in, wav_in.totalPCMFrameCount, buffer);
    if (samples_read != wav_in.totalPCMFrameCount) {
        fprintf(stderr, "Errore durante la lettura dei dati audio.\n");
        return 1;
    }
    drwav_uninit(&wav_in);

    for(int i=0; i<256; i++) {
        printf("[%d] = %d\n", i, buffer[i]);
    }

    // Preparazione per il file di output (copio tutto pari paril)
    drwav_data_format format_out;
    format_out.container = drwav_container_riff;
    format_out.format = DR_WAVE_FORMAT_PCM;
    format_out.channels = wav_in.channels;
    format_out.sampleRate = wav_in.sampleRate;
    format_out.bitsPerSample = wav_in.bitsPerSample;

    drwav wav_out;
    if (!drwav_init_file_write(&wav_out, FILENAME_OUT, &format_out, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file di output %s.\n", FILENAME_OUT);
        return 1;
    }

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, 256, buffer);
    drwav_uninit(&wav_out); // Chiusura del file di output

    // Pulizia della memoria
    free(buffer);

    printf("File WAV copiato con successo: %s\n", FILENAME_OUT);
    return 0;
}
