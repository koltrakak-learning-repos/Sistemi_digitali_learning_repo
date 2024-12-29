#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

// Include STB image libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SAMPLE_RATE 44100
//#define PI 3.14159265358979323846
#define PI 3.14159265359f


#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

typedef struct {
    float real;
    float imag;
} complex;

/*
    funzioni di utilità
*/
__host__ __device__ complex prodotto_tra_complessi(complex a, complex b) {
    complex result;

    result.real = a.real*b.real - a.imag*b.imag;
    result.imag = a.real*b.imag + a.imag*b.real;

    return result;
}

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void checkResult(complex *hostRef, complex *gpuRef, const int N) {
    // epsilon molto largo. Non so perchè la versione GPU differisce rispetto a quella CPU verso la quinta cifra decimale
    double epsilon = 1.0E-4;    
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (fabs(hostRef[i].real - gpuRef[i].real) > epsilon || fabs(hostRef[i].imag - gpuRef[i].imag) > epsilon) {
            match = 0;
            printf("\nArrays do not match!\n");
            printf("\thost (%f; %f) gpu (%f; %f) at current %d\n", hostRef[i].real, hostRef[i].imag, gpuRef[i].real, gpuRef[i].imag, i);
            break;
        }
    }

    if (match)
        printf("\nArrays match.\n\n");
}

// Funzione strana che ho trovato. Mi permette di ottenere il bit reverse order degli indici
// dei campioni della trasformata in maniera efficiente O(log n), rispetto all'usare un ciclo O(n)
//
// es. indice a 8 bit = 5:
//      5 = 00000101   ->  reversed = 10100000 = 160 
__host__ __device__ uint32_t reverse_bits(uint32_t x) {
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

void trasponi_matrice(complex *input, complex *output, const int W, const int H) {
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            output[j*H + i] = input[i*W + j];
        }   
    }
}

float trova_max_ampiezza(complex *output_fft_2D_data, int image_size) {
    float max = 0;

    for (int i = 0; i < image_size; i++) {
        float amplitude = sqrt(output_fft_2D_data[i].real*output_fft_2D_data[i].real + output_fft_2D_data[i].imag*output_fft_2D_data[i].imag);
        
        if (amplitude > max) {
            max = amplitude;
        }
    }

    return max;
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
    float real, imag;
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


/*
    funzioni per fft lato cpu
*/
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
    double start = cpuSecond();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t rev = reverse_bits(i);
        // Non faccio un bit reversal completo ma uno parziale che 
        // tiene conto solo di bit necessari a rappresentare gli N indici
        // del segnale in ingresso. Per cui, qua mantengo solo log_2(N) bit 
        rev = rev >> (32 - num_stadi);

        /*
            Per comodità ho aggiunto questo controllo che mi permette di fare delle
            trasformazioni inplace se input == output
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
    // printf("\tcpu bit_reversal: %f\n", cpuSecond() - start);

    // Stadi 1, ..., log_2(N)
    for (int stadio = 1; stadio <= num_stadi; stadio++) {
        // Variabili di appoggio in cui mi salvo il numero di campioni da considerare nello stadio corrente
        int N_stadio_corrente = 1 << stadio;
        int N_stadio_corrente_mezzi = N_stadio_corrente / 2;

        // Itera sull'array di output con passi pari a N_stadio_corrente
        // k = indice (denormalizzato) del blocco di farfalle considerato nell'array di output 
        for (uint32_t k = 0; k < N; k += N_stadio_corrente) {
            // Calcolo due campioni alla volta per cui itero fino a N_stadio_corrente_mezzi
            /*
                Abbiamo:
                    - output[k...N/2-1] sono le componenti della trasformata pari, mentre
                      output[N/2...N-1] sono le componenti della trasformata dispari.
                        - Guarda diagramma a farfalla. 
                    - j = offset all'interno del blocco di farfalle considerato
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

    // qua utilizzo delle variabili di appoggio dato che non posso modificare input_image_data
    // siccome nel main viene riutilizzata dalla GPU
    complex* temp_trasformata_righe = (complex*)malloc(imageSize*sizeof(complex)); 
    complex* temp_trasposta         = (complex*)malloc(imageSize*sizeof(complex)); 

    // FFT delle righe
    for(int i = 0; i < imageSize; i += row_size) {
        fft_iterativa(&input_image_data[i], &temp_trasformata_righe[i], row_size);
    }

    trasponi_matrice(temp_trasformata_righe, temp_trasposta, row_size, column_size);

    // FFT delle colonne
    for(int j = 0; j < imageSize; j+=column_size) {     // scorro tutte le colonne
        fft_iterativa(&temp_trasposta[j], &output_fft_2D_data[j], column_size);
    }

    free(temp_trasformata_righe);
    free(temp_trasposta);

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
    for(int j = 0; j < imageSize; j+=column_size) {     
        ifft_iterativa(&input_fft_2D_data[j], &output_image_data[j], column_size);
    }

    trasponi_matrice(output_image_data, input_fft_2D_data, row_size, column_size);

    // IFFT delle righe
    for(int i = 0; i < imageSize; i += row_size) {
        ifft_iterativa(&input_fft_2D_data[i], &output_image_data[i], row_size);
    }

    return EXIT_SUCCESS;
}



/*
    funzioni per fft lato gpu
*/
__global__ void fft_bit_reversal(complex *input, complex *output, int N, int righe, int num_stadi) {
    uint32_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    // controllo se ci sono dei thread in eccesso
    if (thread_id >= N) {
        return;
    }

    // ogni thread fa il 'thread-id-esimo' elemento di una riga per 'righe' righe
    uint32_t rev = reverse_bits(thread_id) >> (32 - num_stadi);
    for(int i=0; i<righe; i++) {
        output[thread_id + i*N] = input[rev + i*N];
    }    
}

__global__ void fft_stage(complex *output, int N, int righe, int N_stadio_corrente, int N_stadio_corrente_mezzi) {
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    // controllo se ci sono dei thread in eccesso
    if (thread_id >= N/2) {
        // printf("\tsono un thread in eccesso\n");
        return;
    }

    // ogni thread fa il 'thread-id-esimo' elemento di una riga per 'righe' righe
    for(int i=0; i<righe; i++) {
        // Indice (denormalizzato) del blocco di farfalle considerato nell'array di output 
        int k = (thread_id / N_stadio_corrente_mezzi) * N_stadio_corrente;
        // Offset all'interno del blocco di farfalle considerato
        int j = thread_id % N_stadio_corrente_mezzi;
        // Aggiungo l'offseti di riga
        int kj_riga = k + j +i*N;

        float phi = (-2.0f*PI/N_stadio_corrente) * j;
        complex twiddle_factor = {
            __cosf(phi),
            __sinf(phi)
        };

        complex a = output[kj_riga];
        complex b = prodotto_tra_complessi(twiddle_factor, output[kj_riga + N_stadio_corrente_mezzi]);

        output[kj_riga].real = a.real + b.real;
        output[kj_riga].imag = a.imag + b.imag;
        // simmetria
        output[kj_riga + N_stadio_corrente_mezzi].real = a.real - b.real;
        output[kj_riga + N_stadio_corrente_mezzi].imag = a.imag - b.imag;
    }
}

__global__ void trasponi_matrice_kernel(complex *input, complex *output, const int W, const int H) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < W && iy < H) {
        output[iy*W + ix] = input[ix*H + iy]; // Lettura con stride, scrittura coalescente
    }
}


/*
    NB: questa funzione accetta solo riferimenti al device 
*/
void fft_iterativa_cuda_righe_multiple(complex *d_input, complex *d_output, int N, int righe, int threads_per_block, cudaStream_t stream) {
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u deve essere una potenza di due\n", N);
        return;
    }

    int num_stadi = (int)log2f((double)N);
    
    // Configurazione dei blocchi e dei thread per il bit reversal
    int num_threads = N;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    // stadio 0
    fft_bit_reversal<<<num_blocks, threads_per_block, 0, stream>>>(d_input, d_output, N, righe, num_stadi);

    // Configurazione dei blocchi e dei thread per gli stadi (in generale diversa da quella per il bit reversal)
    num_threads = N/2;  // per calcolare N campioni della trasformata, ho bisogno di soli N/2 thread data la simmetria
    num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    // Lancia i kernel per ogni stadio
    for (int stadio = 1; stadio <= num_stadi; stadio++) {
        int N_stadio_corrente = 1 << stadio;
        int N_stadio_corrente_mezzi = N_stadio_corrente/2;

        fft_stage<<<num_blocks, threads_per_block, 0, stream>>>(d_output, N, righe, N_stadio_corrente, N_stadio_corrente_mezzi);
    }
}

double fft_2D_cuda(complex *input_image_data, complex *output_fft_2D_data, int image_size, int row_size, int column_size,
                   int threads_per_block, int RIGHE_PROCESSATE_ALLA_VOLTA) {
    // Le dimensioni dei dati devono essere potenze di due
    if (image_size != row_size*column_size) {
        fprintf(stderr, "image_size=%u deve essere una potenza di due uguale al prodotto tra row_size e column_size\n", image_size);

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

    // Alloca memoria sulla GPU 
    // (occhio che sotto faccio degli scambi poco leggibili 
    //  dato che non voglio allocare più memoria del necessario)
    complex *d_input;
    complex *d_output;
    cudaMalloc(&d_input, image_size*sizeof(complex));
    cudaMalloc(&d_output, image_size*sizeof(complex));

    // Faccio un unico grande trasferimento
    cudaMemcpy(d_input, input_image_data, image_size*sizeof(complex), cudaMemcpyHostToDevice);

    // calcolo il numero di stream necessario, ho bisogno di uno stream per 
    int bigger_size = row_size > column_size ? row_size : column_size;     
    int max_num_row_blocks = bigger_size / RIGHE_PROCESSATE_ALLA_VOLTA;
    int num_streams = max_num_row_blocks;
    printf("Utilizzo %d streams per elaborare:\n", num_streams);
    printf("\t%d righe da %d elementi (%d righe alla volta)\n", image_size/row_size, row_size, RIGHE_PROCESSATE_ALLA_VOLTA);
    printf("\t%d colonne da %d elementi (%d colonne alla volta)\n", image_size/column_size, column_size, RIGHE_PROCESSATE_ALLA_VOLTA );

    cudaStream_t *streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0 ; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    double start = cpuSecond();
    // FFT delle righe, (RIGHE_PROCESSATE_ALLA_VOLTA righe assegnate ad una singola FFT)
    for(int i = 0; i < image_size; i += row_size*RIGHE_PROCESSATE_ALLA_VOLTA) {
        int indice_blocco_righe = i/(row_size*RIGHE_PROCESSATE_ALLA_VOLTA);
        // printf("\t\t[blocco righe %d] utilizza lo stream %d\n", i/row_size, indice_blocco_righe%num_streams);
        fft_iterativa_cuda_righe_multiple(&d_input[i], &d_output[i], row_size, RIGHE_PROCESSATE_ALLA_VOLTA, threads_per_block, streams[indice_blocco_righe%num_streams]);
    }
    // sincronizzo prima di fare la trasposta
    for (int i=0; i<num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Per fare la FFT delle colonne prima faccio la trasposta della matrice
    int block_dimx = 32; 
    int block_dimy = 32;
    dim3 block(block_dimx, block_dimy);
    dim3 grid((row_size + block.x - 1) / block.x, (column_size + block.y - 1) / block.y);
    trasponi_matrice_kernel<<<grid, block>>>(d_output, d_input, row_size, column_size);
    
    // FFT delle colonne
    for(int j = 0; j<image_size; j+=column_size*RIGHE_PROCESSATE_ALLA_VOLTA) {    
        int indice_blocco_colonne = j/(column_size*RIGHE_PROCESSATE_ALLA_VOLTA);
        // printf("\t\t[blocco colonne %d] utilizza lo stream %d\n", j/column_size, indice_blocco_colonne%num_streams); 
        fft_iterativa_cuda_righe_multiple(&d_input[j], &d_output[j], column_size, RIGHE_PROCESSATE_ALLA_VOLTA, threads_per_block, streams[indice_blocco_colonne%num_streams]);
    }
    // sincronizzo prima di recuperare il risultato finale
    for (int i=0; i<num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Faccio un unico grande trasferimento
    cudaMemcpy(output_fft_2D_data, d_output, image_size*sizeof(complex), cudaMemcpyDeviceToHost);
    double elapsed_gpu = cpuSecond() - start;

    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    for (int i=0; i<num_streams; i++) {
        cudaStreamDestroy(streams[i]) ;
    }

    return elapsed_gpu;
}









int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: %s <file_name> <fattore_di_compressione> <threads_per_fft_block> <righe_processate_alla_volta>\n", argv[0]);
        return 1;
    }

    const char* FILE_NAME                   = argv[1];
    const int FATTORE_DI_COMPRESSIONE       = atoi(argv[2]);
    const int threads_per_fft_block         = atoi(argv[3]);
    const int righe_processate_alla_volta   = atoi(argv[4]);

    // Load the image
    int width, height, channels;
    uint8_t* input_image_data = stbi_load(FILE_NAME, &width, &height, &channels, 0);
    if (!input_image_data) {
        printf("Error loading image %s\n", FILE_NAME);
        return 1;
    }
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
    // salvo le dimensioni originali per dopo
    // int original_width = width;
    // int original_height = height;

    // importante avere come dimensioni potenze di 2 (per FFT)
    pad_image_to_power_of_two(&input_image_data, &width, &height, channels);
    printf("Image padded to: %dx%d\n", width, height);

    // allochiamo memoria per trasformata
    int image_size = width * height * channels;
    complex* complex_input_image_data = (complex*)malloc(sizeof(complex) * image_size);
    convert_to_complex(input_image_data, complex_input_image_data, image_size);
    complex* output_fft_2D_data = (complex*)malloc(sizeof(complex) * image_size);

    // calcolo la FFT 2D lato cpu
    double start = cpuSecond();
    fft_2D(complex_input_image_data, output_fft_2D_data, image_size, width * channels, height);
    double elapsed_host = cpuSecond() - start;
    







    /* ESECUZIONE CON GPU */








    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    printf("CUDA System Information:\n");
    printf("CUDA Driver Version: %d.%d\n", CUDART_VERSION / 1000, CUDART_VERSION % 100);
    printf("CUDA Runtime Version: %d.%d\n", CUDART_VERSION / 1000, CUDART_VERSION % 100);
    printf("\t1. Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("\t2. Total Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("\t3. Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("\t\t4. Maximum number of blocks per MultiProcessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
    printf("\t\t5. Maximum number of threads per MultiProcessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("\t8. Shared Memory per Block: %lu KB\n", deviceProp.sharedMemPerBlock / 1024);
    printf("\t9. Maximum Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("\t10. Maximum Grid Dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("\t11. Maximum Block Dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("\t12. Maximum number of concurrent kernels: %d\n\n", deviceProp.concurrentKernels);

    
    complex* gpu_ref_output_fft_2D_data = (complex *)malloc(image_size*sizeof(complex));
    
    double elapsed_device = fft_2D_cuda(complex_input_image_data, gpu_ref_output_fft_2D_data, image_size, width*channels, height,
                                        threads_per_fft_block, righe_processate_alla_volta);
    
    checkResult(output_fft_2D_data, gpu_ref_output_fft_2D_data, image_size);
    printf("Host: %f ms\n", elapsed_host*1000);
    printf("Device: %f ms\n", elapsed_device*1000);
    printf("\tSPEEDUP: %f\n\n", elapsed_host/elapsed_device);










    /* --- PARTE IFFT-2D --- */

    





    /* ----- COMPRESSIONE ----- */

    char COMPRESSED_FILE_NAME[256];
    sprintf(COMPRESSED_FILE_NAME, "compressed_%s.myformat", FILE_NAME);
    float max_ampiezza = trova_max_ampiezza(gpu_ref_output_fft_2D_data, image_size);
    float soglia = max_ampiezza / FATTORE_DI_COMPRESSIONE; 
    printf("\tSoglia di filtro frequenze: %f\n", soglia);
    comprimi_in_file_binario(gpu_ref_output_fft_2D_data, image_size, width, height, soglia, COMPRESSED_FILE_NAME);

    /*  ----- DECOMPRESSIONE ----- */
    memset(gpu_ref_output_fft_2D_data, 0, sizeof(complex) * image_size);
    memset(complex_input_image_data, 0, sizeof(complex) * image_size);

    decomprimi_in_campioni_fft_2D(COMPRESSED_FILE_NAME, gpu_ref_output_fft_2D_data, width, height);
    ifft_2D(gpu_ref_output_fft_2D_data, complex_input_image_data, image_size, width * channels, height);

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
    free(complex_input_image_data);
    free(output_image_data);
    free(gpu_ref_output_fft_2D_data);
    free(output_fft_2D_data);
}