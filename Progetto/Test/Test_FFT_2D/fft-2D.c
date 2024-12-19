#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define SAMPLE_RATE 44100
#define PI 3.14159265358979323846

typedef struct {
    double real;
    double imag;
} complex;

complex prodotto_tra_complessi(complex a, complex b) {
    complex result;

    result.real = a.real*b.real - a.imag*b.imag;
    result.imag = a.real*b.imag + a.imag*b.real;

    return result;
}

// Funzione strana che ho trovato, che mi permette di ottenere il bit reverse order degli indici
// dei campioni della trasformata  in maniera efficiente O(log n).
// es. indice a 8 bit = 5:
//  5 = 00000101   ->  reversed = 10100000 = 160 
uint32_t reverse_bits(uint32_t x) {
    // 1. Swap the position of consecutive bits
    // 2. Swap the position of consecutive pairs of bits
    // 3. Swap the position of consecutive quads of bits
    // 4. Continue this until swapping the two consecutive 16-bit parts of x
    
    /*
        Primo scambio:
        0xaaa... = 1010-1010-... = bit pari; 0x555... = 0101-0101... bit dispari; 
        - nel primo gruppo seleziono i bit pari e li sposto a destra di una posizione
        - nel secondo gruppo selezioni i bit dispari e li sposto a sinistra di una posizione
        - facendo infine l'or dei due gruppi ottengo i la stringa di bit con le posizioni pari e dispari scambiate

        Lo stesso procedimento si ripete sotto
    */
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);  
    // 0xccc... = 1100-1100-... = coppie di bit pari; 0x333... = 0011-0011... coppie di bit dispari; 
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    // 0xf0f0... = 11110000-... = quadruple di bit pari; 0x0f0f... = 00001111 ... quadruple di bit dispari; 
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    // stessa cosa con gruppi da 8
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

void convert_to_complex(uint8_t *input, complex *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i].real = (double)input[i];
        output[i].imag = 0.0;
    }
}

void convert_to_uint8(complex *input, uint8_t *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = (uint8_t)round(input[i].real); 
    }
}

void pad_image_to_power_of_two(uint8_t** input_image_data, int* width, int* height, int channels) {
    if( !(*width & (*width - 1)) && !(*height & (*height - 1)) ) {
        // sono già potenze di due
        printf("\tasdasdasd\n");
        return;
    }

    // Calcola la dimensione "padded" (prossima potenza di 2)
    int new_width   = 1 << (int)ceil(log2(*width));
    int new_height  = 1 << (int)ceil(log2(*height));

    printf("\t%d, %d\n", new_width, new_height);
    // Alloca la nuova immagine con padding
    int new_image_size = new_width * new_height * channels;
    uint8_t* padded_image_data = (uint8_t*)malloc(new_image_size);
    memset(padded_image_data, 0, new_image_size);

    // Copia i dati dell'immagine originale nella nuova immagine con padding
    for (int y = 0; y < *height; y++) {             
        for (int x = 0; x < *width; x++) {          
            for (int c = 0; c < channels; c++) { 
                padded_image_data[(y*new_width + x) * channels + c] = (*input_image_data)[(y * (*width) + x) * channels + c];
            }
        }
    }

    // aggiorno parametri per output
    free(*input_image_data);

    *width = new_width;
    *height = new_height; 
    *input_image_data = padded_image_data;
}




int fft_iterativa(complex *input, complex *output, int N) {
    // N & (N - 1) = ...01000... & ...00111... = 0
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u deve essere una potenza di due\n", N);

        return -1;
    }

    // num_stadi = "quante volte posso dividere N per due"
    int num_stadi = (int) log2f((float) N);

    // stadio 0: DFT di un campione
    // L'output di questo primo stadio equivale all'input riordinato in bit-reverse order
    // Questo riordino corrisponde implicitamente alla separazione in pari e dispari
    // che avviene in modo esplicito nella versione ricorsiva.
    for (uint32_t i = 0; i < N; i++) {
        uint32_t rev = reverse_bits(i);
        // Non faccio un bit reversal completo ma uno parziale che 
        // tiene conto solo di bit necessari a rappresentare gli N indici
        // del segnale in ingresso. Per cui, qua mantengo solo log_2(N) bit 
        rev = rev >> (32 - num_stadi);

        output[i] = input[rev];
    }

    // Stadi 1, ..., log_2(N)
    for (int stadio = 1; stadio <= num_stadi; stadio++) {
        // Variabili di appoggio in cui mi salvo il numero di campioni da considerare nello stadio corrente
        int N_stadio_corrente = 1 << stadio;
        int N_stadio_corrente_mezzi = N_stadio_corrente / 2;

        // Itera sull'array di output con passi pari a N_stadio_corrente
        for (uint32_t k = 0; k < N; k += N_stadio_corrente) {
            // Calcolo due campioni alla volta per cui itero fino a N_stadio_corrente_mezzi
            /*
                Abbiamo:
                    - output[k...N/2-1] sono le componenti della trasformata pari, mentre
                      output[N/2...N-1] sono le componenti della trasformata dispari.
                        - Guarda diagramma a farfalla. 
                    
                    ...
            */
            for (int j = 0; j < N_stadio_corrente_mezzi; j++) {
                double phi = (-2*PI/N_stadio_corrente) * j; 
                complex twiddle_factor = {
                    cos(phi),
                    sin(phi)
                };

                complex a = output[k + j];
                complex b = prodotto_tra_complessi(twiddle_factor, output[k + j + N_stadio_corrente_mezzi]);

                // calcolo trasformata
                output[k + j].real = a.real + b.real;
                output[k + j].imag = a.imag + b.imag;
                // simmetria per la seconda metà
                output[k + j + N_stadio_corrente_mezzi].real = a.real - b.real;
                output[k + j + N_stadio_corrente_mezzi].imag = a.imag - b.imag;
            }
        }
    }

    return EXIT_SUCCESS;
}


int fft_2D(complex *input_image_data, complex *output_fft_2D_data, int imageSize, int row_size, int column_size) {
    // Le dimensioni dei dati devono essere potenze di due
    if (imageSize != row_size*column_size) {
        fprintf(stderr, "imageSize=%u deve essere una potenza di due uguale al prodotto tra row_size e column_size\n", imageSize);

        return -1;
    }
    if (row_size & (row_size - 1)) {
        fprintf(stderr, "row_size=%u deve essere una potenza di due\n", row_size);

        return -1;
    }
    if (column_size & (column_size - 1)) {
        fprintf(stderr, "column_size=%u deve essere una potenza di due\n", column_size);

        return -1;
    }


    // FFT delle righe
    for(int i = 0; i < imageSize; i += row_size) {
        fft_iterativa(&input_image_data[i], &output_fft_2D_data[i], row_size);
    }

    // FFT delle colonne
    //      -> j indice di colonna
    //      -> i indice di riga
    for(int j = 0; j < row_size; j++) {     // scorro tutte le colonne
        // mi costruisco la colonna
        //      -> il passo è row size;
        //      -> devo poi fare column_size passi
        complex colonna[column_size];
        for(int i = 0; i < column_size; i++) {   // scorro tutti gli elementi di una colonna
            colonna[i] = output_fft_2D_data[i*row_size + j];
        }

        // scrivo row_size colonne
        fft_iterativa(colonna, &output_fft_2D_data[j*column_size], column_size);
    }

    return EXIT_SUCCESS;
}

// NB: nota come si praticamente uguale alla fft se non per il segno + dei twiddle 
void ifft_inplace_recursive(complex *input, complex *output, int step, int N) {
    if (N == 1) {
        output[0] = input[0];
        return;
    }

    // Calcola la IFFT sui sotto-array pari e dispari
    ifft_inplace_recursive(input, output, step*2, N/2);
    ifft_inplace_recursive(input + step, output + N/2, step*2, N/2); 

    // Combina i risultati
    for (int k = 0; k < N/2; k++) {
        double phi = 2*PI*k / N; // Cambia il segno per la IFFT
        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        complex even = output[k];

        complex temp = {
            twiddle.real * output[k + N/2].real - twiddle.imag * output[k + N/2].imag,
            twiddle.real * output[k + N/2].imag + twiddle.imag * output[k + N/2].real
        };

        output[k].real = even.real + temp.real;
        output[k].imag = even.imag + temp.imag;
        //relazione simmetrica
        output[k + N/2].real = even.real - temp.real;
        output[k + N/2].imag = even.imag - temp.imag;
    }
}

void ifft_inplace(complex *input, complex *output, int N) {
    ifft_inplace_recursive(input, output, 1, N);

    // Non scordarti di normalizzare
    // NB: a quanto pare è importante che la normalizzazione avvenga soltanto alla fine di tutto
    for (int i = 0; i < N; i++) {
        output[i].real /= N;
        output[i].imag /= N;
    }
}



int main() {
    const char* FILE_NAME = "image.png";

    // Load the image
    int width, height, channels;
    uint8_t* input_image_data = stbi_load(FILE_NAME, &width, &height, &channels, 0);
    if (!input_image_data) {
        printf("Error loading image %s\n", FILE_NAME);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);

    // importante avere una potenza di 2
    pad_image_to_power_of_two(&input_image_data, &width, &height, channels);
    printf("Image padded to: %dx%d\n", width, height);

    // allochiamo memoria per trasformata
    int image_size = width * height * channels;
    complex* complex_input_image_data = (complex*)malloc(sizeof(complex) * image_size);
    convert_to_complex(input_image_data, complex_input_image_data, image_size);
    complex* output_fft_2D_data = (complex*)malloc(sizeof(complex) * image_size);

    fft_2D(complex_input_image_data, output_fft_2D_data, image_size, width * channels, height);










    

    // Copio solo i byte blu
    uint8_t* output_image_data = (uint8_t*)malloc(image_size);
    memset(output_image_data, 0, image_size);
    for(int i=0; i<image_size; i+=channels) {
        output_image_data[i+2] = input_image_data[i+2];
    }

    // Save the output image
    char OUTPUT_FILE_NAME[256];
    sprintf(OUTPUT_FILE_NAME, "output_%s", FILE_NAME);
    stbi_write_png(OUTPUT_FILE_NAME, width, height, channels, output_image_data, width * channels);
    printf("Output saved as %s\n", OUTPUT_FILE_NAME);

    // Clean up
    stbi_image_free(input_image_data);
    free(output_image_data);
    














    // // calcolo la FFT
    // convert_to_complex(signal_samples, complex_signal_samples, num_samples);
    // fft_iterativa(complex_signal_samples, fft_samples, num_samples);

    // // Calcola e salvo l'ampiezza per ciascuna frequenza
    // FILE *output_file = fopen("amplitude_spectrum.txt", "w");
    // if (output_file == NULL) {
    //     fprintf(stderr, "Errore nell'aprire il file di output.\n");
    //     return 1;
    // }

    // for (int i = 0; i < num_samples; i++) {
    //     double amplitude = sqrt(fft_samples[i].real*fft_samples[i].real + fft_samples[i].imag*fft_samples[i].imag);
    //     double frequency = (double)i * SAMPLE_RATE / num_samples;

    //     fprintf(output_file, "%lf %lf\n", frequency, amplitude);

    //     if(amplitude > 1000000) {
    //         printf("Frequenza: %lf sembra essere un componente utile del segnale\n", frequency);
    //     }
    // }

    // printf("I dati dello spettro sono stati scritti in 'amplitude_spectrum.txt'.\n");
    // fclose(output_file);




    /* --- PARTE IFFT --- */

    

    // // inizializzazione dati
    // char generated_filename[100];   //dimensione arbitraria perchè non ho voglia
    // sprintf(generated_filename, "IFFT-generated-%s", FILE_NAME);
    // // mi assicuro di non imbrogliare ricopiando i dati di prima
    // memset(signal_samples, 0, num_samples*sizeof(short));
    // memset(complex_signal_samples, 0, num_samples);

    // // Preparazione del formato del file di output
    // drwav_data_format format_out;
    // format_out.container = drwav_container_riff;
    // format_out.format = DR_WAVE_FORMAT_PCM;
    // format_out.channels = 1;              // Mono
    // format_out.sampleRate = SAMPLE_RATE;  // Frequenza di campionamento
    // format_out.bitsPerSample = 16;        // 16 bit per campione

    // // Inizializzazione del file di output
    // drwav wav_out;
    // if (!drwav_init_file_write(&wav_out, generated_filename, &format_out, NULL)) {
    //     fprintf(stderr, "Errore nell'aprire il file di output %s.\n", generated_filename);
    //     return 1;
    // }
    
    // ifft_inplace(fft_samples, complex_signal_samples, num_samples);
    // convert_to_short(complex_signal_samples, signal_samples, num_samples);

    // // Scrittura dei dati audio nel file di output
    // drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    // drwav_uninit(&wav_out); // Chiusura del file di output

    // printf("File WAV %s creato con successo\n", generated_filename);

    // free(signal_samples);
    // free(complex_signal_samples);
    // free(fft_samples);
}