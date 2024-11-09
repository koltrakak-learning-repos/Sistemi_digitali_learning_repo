#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <math.h>
#include <stdio.h>


#define SAMPLE_RATE 44100
#define DURATION 5
#define FREQUENCY 440.0
#define AMPLITUDE 32767

# define M_PI		3.14159265358979323846

int main() {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = SAMPLE_RATE;
    format.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, "nota_A440_drwav.wav", &format, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file.\n");
        return 1;
    }

    int num_samples = SAMPLE_RATE * DURATION;
    short *buffer = malloc(num_samples * sizeof(short));

    for (int i = 0; i < num_samples; i++) {
        double t = (double)i / SAMPLE_RATE;
        buffer[i] = (short)(AMPLITUDE * sin(2.0 * M_PI * FREQUENCY * t));
    }

    drwav_write_pcm_frames(&wav, num_samples, buffer);
    drwav_uninit(&wav);
    free(buffer);

    printf("File WAV generato con successo: nota_A440_drwav.wav\n");
    return 0;
}
