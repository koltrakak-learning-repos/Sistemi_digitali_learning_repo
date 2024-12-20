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
    float real;
    float imag;
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
        output[i].real = (float)input[i];
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
        return;
    }

    // Calcola la dimensione "padded" (prossima potenza di 2)
    int new_width   = 1 << (int)ceil(log2(*width));
    int new_height  = 1 << (int)ceil(log2(*height));

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

    
    free(*input_image_data);
    // aggiorno parametri per output
    *width = new_width;
    *height = new_height; 
    *input_image_data = padded_image_data;
}

void unpad_image_to_original_size(uint8_t** input_image_data, int* padded_width, int* padded_height,
                                  int original_width, int original_height, int channels) {
    // Alloca memoria per l'immagine senza padding
    int original_image_size = original_width * original_height * channels;
    uint8_t* unpadded_image_data = (uint8_t*)malloc(original_image_size);

    // Copia i dati dalla versione con padding alla versione originale
    for (int y = 0; y < original_height; y++) {
        for (int x = 0; x < original_width; x++) {
            for (int c = 0; c < channels; c++) {
                unpadded_image_data[(y * original_width + x) * channels + c] = (*input_image_data)[(y * (*padded_width) + x) * channels + c];
            }
        }
    }

    free(*input_image_data);
    // Aggiorna i parametri di output
    *padded_width = original_width;
    *padded_height = original_height;
    *input_image_data = unpadded_image_data;
}

// float calcola_frequenze_2D(int kx, int ky, int width, int height) {
//     /*
//         Secondo: https://dsp.stackexchange.com/questions/22661/how-do-i-obtain-the-frequencies-of-each-value-in-a-2d-fft-and-what-do-they-mean 
//         La frequenza di una componente della fft-2D si calcola così.
//     */

//     // Frequenze 1D lungo x e y
//     float f_x = (float)kx / width;
//     float f_y = (float)ky / height;

//     // Frequenza spaziale combinata
//     float f_xy = sqrt(f_x*f_x + f_y*f_y);

//     return f_xy;
// }

float calcola_frequenze_2D(int kx, int ky, int width, int height) {
    // A quanto pare bisogna effettuare una "centralizzazione degli indici"...
    // Non so cosa significhi ma adesso ho dei valori più sensati

    if (kx >= width / 2) {
        kx -= width;  // Mappare gli indici da -width/2 a +width/2
    }
    if (ky >= height / 2) {
        ky -= height;  // Mappare gli indici da -height/2 a +height/2
    }

    // Frequenze 1D lungo x e y
    float f_x = (float)kx / width;
    float f_y = (float)ky / height;

    // Frequenza spaziale combinata (modulo)
    float f_xy = sqrt(f_x * f_x + f_y * f_y);

    return f_xy;
}

float trova_ampiezza_media(complex *output_fft_2D_data, int image_size) {
    float sum = 0;

    for (int i = 0; i < image_size; i++) {
        float amplitude = sqrt(output_fft_2D_data[i].real*output_fft_2D_data[i].real + output_fft_2D_data[i].imag*output_fft_2D_data[i].imag);
        
        sum += amplitude;
    }

    return sum / image_size;
}

int comprimi_in_file_binario(complex *output_fft_2D_data, int image_size, int width, int height, float soglia, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file per la scrittura\n");
        return -1;
    }

    int conta = 0;
    for (int i = 0; i < image_size; i++) {
        float amplitude = sqrt(output_fft_2D_data[i].real*output_fft_2D_data[i].real + output_fft_2D_data[i].imag*output_fft_2D_data[i].imag);
        
        // Se l'ampiezza è maggiore della soglia, salviamo il campione
        if (amplitude > soglia) {
            // Scrivi indice del campione, parte reale e immaginaria nel file binario
            fwrite(&i, sizeof(int), 1, file);
            fwrite(&output_fft_2D_data[i].real, sizeof(float), 1, file);
            fwrite(&output_fft_2D_data[i].imag, sizeof(float), 1, file);

            conta++;
        }
    }

    fclose(file);

    printf("File compresso con successo!\n");
    printf("\tL'immagine di dimensione %d byte è stata compressa in %d entry da 12 byte (%d byte)\n", image_size, conta, conta*12);
    printf("\tGuadagno: %f\n", (float)image_size / (conta*12));

    return 0;
}

int decomprimi_in_campioni_fft_2D(const char *filename, complex *output_fft_2D_data, int width, int height) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file per la lettura\n");
        return -1;
    }

    // Tutti i campioni sono di base nulli
    for (int i = 0; i < width * height; i++) {
        output_fft_2D_data[i].real = 0;
        output_fft_2D_data[i].imag = 0;
    }

    int index;
    float i, real, imag;
    while (fread(&index, sizeof(int), 1, file) == 1 &&
           fread(&real, sizeof(float), 1, file) == 1 &&
           fread(&imag, sizeof(float), 1, file) == 1) {
        
        // Reinserisci il campione nella matrice FFT-2D
        output_fft_2D_data[index].real = real;
        output_fft_2D_data[index].imag = imag;
    }

    fclose(file);
    printf("file decompresso con successo!\n");
    return 0;
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

        /*
            Per comodità ho aggiunto questo controllo che mi permette di fare delle
            trasformazioni inplace
        */
        if(input == output) {
            if (i < rev) {  
                complex temp = input[i];
                output[i] = input[rev];
                output[rev] = temp;
            }
        }
        else {
            output[i] = input[rev];
        }
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
                float phi = (-2*PI/N_stadio_corrente) * j; 
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

int ifft_iterativa(complex *input, complex *output, int N) {
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u deve essere una potenza di due\n", N);

        return -1;
    }

    int num_stadi = (int) log2f((float) N);

    // stadio 0
    for (uint32_t i = 0; i < N; i++) {
        uint32_t rev = reverse_bits(i);
        rev = rev >> (32 - num_stadi);

        /*
            Per comodità ho aggiunto questo controllo che mi permette di fare delle
            trasformazioni inplace
        */
        if(input == output) {
            if (i < rev) {  
                complex temp = input[i];
                output[i] = input[rev];
                output[rev] = temp;
            }
        }
        else {
            output[i] = input[rev];
        }
    }

    // Stadi 1, ..., log_2(N)
    for (int stadio = 1; stadio <= num_stadi; stadio++) {
        int N_stadio_corrente = 1 << stadio;
        int N_stadio_corrente_mezzi = N_stadio_corrente / 2;

        for (uint32_t k = 0; k < N; k += N_stadio_corrente) {
            for (int j = 0; j < N_stadio_corrente_mezzi; j++) {
                float phi = 2*PI/N_stadio_corrente * j;   // segno + per ifft 
                complex twiddle_factor = {
                    cos(phi),
                    sin(phi)
                };

                complex a = output[k + j];
                complex b = prodotto_tra_complessi(twiddle_factor, output[k + j + N_stadio_corrente_mezzi]);

                // calcolo antitrasformata
                output[k + j].real = a.real + b.real;
                output[k + j].imag = a.imag + b.imag;
                // simmetria per la seconda metà
                output[k + j + N_stadio_corrente_mezzi].real = a.real - b.real;
                output[k + j + N_stadio_corrente_mezzi].imag = a.imag - b.imag;
            }
        }
    }

    // normalizza i risultati alla fine
    for(int i=0; i<N; i++) {
        output[i].real /= N;
        output[i].imag /= N;
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
        for(int i = 0; i < column_size; i++) {
            colonna[i] = output_fft_2D_data[i * row_size + j];
        }

        fft_iterativa(colonna, colonna, column_size);

        for(int i = 0; i < column_size; i++) {
            output_fft_2D_data[i * row_size + j] = colonna[i];
        }
    }

    return EXIT_SUCCESS;
}

int ifft_2D(complex *input_fft_2D_data, complex *output_image_data, int imageSize, int row_size, int column_size) {
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


    // IFFT delle colonne
    for(int j = 0; j < row_size; j++) {     // scorro tutte le colonne
        // mi costruisco la colonna
        //      -> il passo è row size;
        //      -> devo poi fare column_size passi
        complex colonna[column_size];
        for(int i = 0; i < column_size; i++) {
            colonna[i] = input_fft_2D_data[i * row_size + j];
        }

        ifft_iterativa(colonna, colonna, column_size);
        for(int i = 0; i < column_size; i++) {
            output_image_data[i*row_size + j] = colonna[i];
        }
    }

    // IFFT delle righe
    //      -> j indice di colonna
    //      -> i indice di riga
    for(int i = 0; i < imageSize; i += row_size) {
        ifft_iterativa(&output_image_data[i], &output_image_data[i], row_size);
    }

    return EXIT_SUCCESS;
}

int main() {
    const char* FILE_NAME = "image_grayscale.png";
    const int FATTORE_DI_COMPRESSIONE = 3;

    // Load the image
    int width, height, channels;
    uint8_t* input_image_data = stbi_load(FILE_NAME, &width, &height, &channels, 0);
    if (!input_image_data) {
        printf("Error loading image %s\n", FILE_NAME);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
    // salvo le dimensioni originali per dopo
    int original_width = width;
    int original_height = height;

    // importante avere come dimensioni potenze di 2 (per FFT)
    pad_image_to_power_of_two(&input_image_data, &width, &height, channels);
    printf("Image padded to: %dx%d\n", width, height);

    // allochiamo memoria per trasformata
    int image_size = width * height * channels;
    complex* complex_input_image_data = (complex*)malloc(sizeof(complex) * image_size);
    convert_to_complex(input_image_data, complex_input_image_data, image_size);
    complex* output_fft_2D_data = (complex*)malloc(sizeof(complex) * image_size);


    /* ----- COMPRESSIONE ----- */
    fft_2D(complex_input_image_data, output_fft_2D_data, image_size, width * channels, height);
    

    char COMPRESSED_FILE_NAME[256];
    sprintf(COMPRESSED_FILE_NAME, "compressed_%s.myformat", FILE_NAME);
    float max_ampiezza = trova_ampiezza_media(output_fft_2D_data, image_size);
    float soglia = max_ampiezza * FATTORE_DI_COMPRESSIONE; 
    printf("\tsoglia di filtro: %f\n", soglia);
    comprimi_in_file_binario(output_fft_2D_data, image_size, width, height, soglia, COMPRESSED_FILE_NAME);

    /*  ----- DECOMPRESSIONE ----- */
    memset(output_fft_2D_data, 0, sizeof(complex) * image_size);
    memset(complex_input_image_data, 0, sizeof(complex) * image_size);

    decomprimi_in_campioni_fft_2D(COMPRESSED_FILE_NAME, output_fft_2D_data, width, height);
    ifft_2D(output_fft_2D_data, complex_input_image_data, image_size, width * channels, height);

    uint8_t* output_image_data = (uint8_t*)malloc(image_size);
    convert_to_uint8(complex_input_image_data, output_image_data, image_size);

    // unpad_image_to_original_size(&output_image_data, &width, &height, original_width, original_height, channels);
    
    // Save the decompressed image
    char DECOMPRESSED_FILE_NAME[256];
    sprintf(DECOMPRESSED_FILE_NAME, "decompressed_%s", FILE_NAME);
    stbi_write_png(DECOMPRESSED_FILE_NAME, width, height, channels, output_image_data, width * channels);
    printf("Output saved as: %s\n", DECOMPRESSED_FILE_NAME);

    // Clean up
    stbi_image_free(input_image_data);
    free(output_image_data);
}