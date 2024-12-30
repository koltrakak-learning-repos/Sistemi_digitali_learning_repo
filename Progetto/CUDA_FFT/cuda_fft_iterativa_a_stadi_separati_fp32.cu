#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

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
            printf("Arrays do not match!\n");
            printf("host (%f; %f) gpu (%f; %f) at current %d\n", hostRef[i].real, hostRef[i].imag, gpuRef[i].real, gpuRef[i].imag, i);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
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

void convert_to_complex(short *input, complex *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i].real = (float)input[i];
        output[i].imag = 0.0;
    }
}

void convert_to_short(complex *input, short *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = (short)round(input[i].real); 
    }
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
    printf("\tcpu bit_reversal: %f\n", cpuSecond() - start);

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





__global__ void fft_bit_reversal(complex *input, complex *output, int N, int num_stadi) {
    uint32_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    // controllo se ci sono dei thread in eccesso
    if (thread_id >= N) {
        return;
    }

    // Copia input nell'output con bit-reversal (stadio 0)
    uint32_t rev = reverse_bits(thread_id) >> (32 - num_stadi);

    output[thread_id] = input[rev];    
}

// Kernel che calcola una farfalla 
__global__ void fft_stage(complex *output, int N, int N_stadio_corrente, int N_stadio_corrente_mezzi) {
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    // controllo se ci sono dei thread in eccesso
    if (thread_id >= N/2) {
        // printf("\tsono un thread in eccesso\n");
        return;
    }

    // Indice (denormalizzato) del blocco di farfalle considerato nell'array di output 
    int k = (thread_id / N_stadio_corrente_mezzi) * N_stadio_corrente;
    // Offset all'interno del blocco di farfalle considerato
    int j = thread_id % N_stadio_corrente_mezzi;
    // % evitare il modulo non cambia le performance
    // int j = thread_id - (thread_id / N_stadio_corrente_mezzi) * N_stadio_corrente_mezzi;

    /*
        TODO: ogni thread che produce lo stesso 'j' ripete questo calcolo inutilmente
        potrebbe essere utile precalcolare il vettore dei twiddle factor  
    */
    float phi = __fdividef(-2.0f*PI, N_stadio_corrente) * j;
    complex twiddle_factor = {
        __cosf(phi),
        __sinf(phi)
    };

    complex a = output[k + j];
    complex b = prodotto_tra_complessi(twiddle_factor, output[k + j + N_stadio_corrente_mezzi]);

    output[k + j].real = a.real + b.real;
    output[k + j].imag = a.imag + b.imag;
    // simmetria
    output[k + j + N_stadio_corrente_mezzi].real = a.real - b.real;
    output[k + j + N_stadio_corrente_mezzi].imag = a.imag - b.imag;
}

double fft_iterativa_cuda(complex *input, complex *output, int N) {
    // Controllo che N sia una potenza di 2
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u deve essere una potenza di due\n", N);
        return 0;
    }

    int num_stadi = (int)log2f((double)N);

    // Alloca memoria sulla GPU
    complex *d_input;
    complex *d_output;
    cudaMalloc(&d_output, N*sizeof(complex));
    cudaMalloc(&d_input, N*sizeof(complex));
    cudaMemcpy(d_input, input, N*sizeof(complex), cudaMemcpyHostToDevice);

    // Configurazione dei blocchi e dei thread per il bit reversal
    int threads_per_block = 256;
    int num_threads = N;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    double start = cpuSecond();
    // stadio 0
    fft_bit_reversal<<<num_blocks, threads_per_block>>>(d_input, d_output, N, num_stadi);
    // cudaDeviceSynchronize(); non necessario
    printf("\tgpu bit_reversal: %f\n", cpuSecond() - start);

    // Configurazione dei blocchi e dei thread per gli stadi (in generale diversa da quella per il bit reversal)
    threads_per_block = 256;
    num_threads = N/2;  // per calcolare N campioni della trasformata, ho bisogno di soli N/2 thread data la simmetria
    num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    // Lancia i kernel per ogni stadio
    for (int stadio = 1; stadio <= num_stadi; stadio++) {
        int N_stadio_corrente = 1 << stadio;
        int N_stadio_corrente_mezzi = N_stadio_corrente/2;

        fft_stage<<<num_blocks, threads_per_block>>>(d_output, N, N_stadio_corrente, N_stadio_corrente_mezzi);
        // cudaDeviceSynchronize(); non necessario
    }

    cudaMemcpy(output, d_output, N*sizeof(complex), cudaMemcpyDeviceToHost);
    double elapsed_gpu = cpuSecond() - start;

    cudaFree(d_input);
    cudaFree(d_output);

    return elapsed_gpu;
}



int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <file_name>\n", argv[0]);
        return 1;
    }

    const char* FILE_NAME = argv[1];
    drwav wav_in;
    
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file %s.wav.\n", FILE_NAME);
        return 1;
    }

    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;
    printf("NUMERO DI CAMPIONI NEL FILE AUDIO SCELTO: %ld; -> %0.2f secondi\n", num_samples, (double)num_samples/SAMPLE_RATE);
    printf("dopo il padding: 2^%d\n\n", (int)ceil(log2(num_samples)));
    // importante avere una potenza di 2
    int padded_samples = 1 << (int)ceil(log2(num_samples));
    if (padded_samples > num_samples) {
        num_samples = padded_samples;
    }

    /*
        Alloco memoria per:
            - campioni PCM a 16 bit del file di ingresso
            - campioni PCM a 16 bit del file di ingresso convertiti in numeri complessi
            - campioni della trasformata ottenuti con FFT
    */
    short* signal_samples = (short*)malloc(num_samples * sizeof(short));
    if (signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    complex* complex_signal_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (complex_signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }
    complex* fft_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (fft_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        return 1;
    }

    // Lettura dei dati audio dal file di input
    size_t frames_read = drwav_read_pcm_frames_s16(&wav_in, wav_in.totalPCMFrameCount, signal_samples);
    if (frames_read != wav_in.totalPCMFrameCount) {
        fprintf(stderr, "Errore durante la lettura dei dati audio.\n");
        return 1;
    }
    drwav_uninit(&wav_in); 

    // calcolo la FFT
    convert_to_complex(signal_samples, complex_signal_samples, num_samples);
    double start = cpuSecond();
    fft_iterativa(complex_signal_samples, fft_samples, num_samples);
    double elapsed_host = cpuSecond() - start;









    /* ESECUZIONE CON GPU */











    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    
    complex* gpu_ref_fft_samples = (complex *)malloc(num_samples*sizeof(complex));
    
    double elapsed_device = fft_iterativa_cuda(complex_signal_samples, gpu_ref_fft_samples, num_samples); 
    
    checkResult(fft_samples, gpu_ref_fft_samples, num_samples);
    printf("Host: %f ms\n", elapsed_host*1000);
    printf("Device: %f ms\n", elapsed_device*1000);
    printf("SPEEDUP: %f\n", elapsed_host/elapsed_device);










    /* --- PARTE IFFT --- */

    

    // inizializzazione dati
    char generated_filename[100];   //dimensione arbitraria perchè non ho voglia
    sprintf(generated_filename, "GPU-IFFT-generated-%s", FILE_NAME);
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
    
    ifft_iterativa(gpu_ref_fft_samples, complex_signal_samples, num_samples);
    convert_to_short(complex_signal_samples, signal_samples, num_samples);

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    drwav_uninit(&wav_out); // Chiusura del file di output

    printf("File WAV %s creato con successo\n", generated_filename);

    free(signal_samples);
    free(complex_signal_samples);
    free(fft_samples);
    free(gpu_ref_fft_samples);
}