#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SAMPLE_RATE 44100
#define PI 3.14159265358979323846

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

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
    double real;
    double imag;
} complex;

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void checkResult(complex *hostRef, complex *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i].real - gpuRef[i].real) > epsilon || abs(hostRef[i].imag - gpuRef[i].imag) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host (%0.2f; %0.2f) gpu (%0.2f; %0.2f) at current %d\n", hostRef[i].real, hostRef[i].imag, gpuRef[i].real, gpuRef[i].imag, i);
            break;
        }
    }
    if (match)
        printf("Arrays match.\n\n");
}



/* FUNZIONI HOST */




/*
    ...
    - "step" indica la distanza tra due campioni SUCCESSIVI che stiamo considerando ad un determinato stadio.
*/
void fft_inplace_recursive(complex *input, complex *output, int step, int N) {
    if (N == 1) {
        // Caso base: La DFT di un solo campione
        // è uguale al campione stesso.
        //  -> copia direttamente l'input nell'output
        output[0] = input[0];
        return;
    }

    // Calcola la IFFT sui sotto-array pari e dispari
    /*
        NB: Occhio all'input e all'output della parte dispari
        - la parte dispari inizia dopo step celle
        - le componenti della trasformata dispari iniziano 
          nella seconda metà dell'output (guarda diagramma a farfalla) 

        A vederlo ad occhio è difficile ma se fai i passaggi su un foglio
        salta fuori il bit reversal order
    */ 
    fft_inplace_recursive(input, output, step*2, N/2);                 // campioni pari
    fft_inplace_recursive(input + step, output + N/2, step*2, N/2);    // campioni dispari

    // Combina i risultati delle due FFT dello stadio precedente per
    // ottenere quella dello stadio corrente
    for (int k = 0; k < N/2; k++) {
        double phi = (-2*PI/N) * k; // Segno negativo per la FFT
        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        complex even = output[k];

        /*
            temp = prodotto algebrico tra campioni della trasformata odd e twiddle factor 
            ho usato una variabile d'appoggio per rendere più leggibile sotto

            Da quanto detto sopra, abbiamo che output[k...N/2-1] sono le componenti della 
            trasformata pari, mentre output[N/2...N-1] sono le componenti della trasformata
            dispari. Di nuovo guarda diagramma a farfalla. 
        */
        complex temp = {
            twiddle.real * output[k + N/2].real - twiddle.imag * output[k + N/2].imag,
            twiddle.real * output[k + N/2].imag + twiddle.imag * output[k + N/2].real
        };

        output[k].real = even.real + temp.real;
        output[k].imag = even.imag + temp.imag;
        // seconda metà delle frequenza ottenuta per simmetria
        // (temp con segno meno dato che il twiddle della seconda metà ha segno opposto)
        output[k + N/2].real = even.real - temp.real;
        output[k + N/2].imag = even.imag - temp.imag;
    }
}

void fft_host_side(complex *input, complex *output, int N) {
    // Controllo se N è una potenza di 2
    if ((N & (N - 1)) != 0) {
        fprintf(stderr, "Errore: N (%d) non è una potenza di 2.\n", N);
        exit(1);
    }

    fft_inplace_recursive(input, output, 1, N);
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

void ifft_host_side(complex *input, complex *output, int N) {
    ifft_inplace_recursive(input, output, 1, N);

    // Non scordarti di normalizzare
    // NB: a quanto pare è importante che la normalizzazione avvenga soltanto alla fine di tutto
    for (int i = 0; i < N; i++) {
        output[i].real /= N;
        output[i].imag /= N;
    }
}

void convert_to_complex(short *input, complex *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i].real = (double)input[i];
        output[i].imag = 0.0;
    }
}

void convert_to_short(complex *input, short *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = (short)round(input[i].real); 
    }
}




/* KERNEL GPU */



 
__global__ void fft_device_side(complex *input, complex *output, int step, int N) {
    /*
        Quando lancio un'altra griglia questo si ripete... è giusto???
        Penso di si, è un sostituto di k
    */
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grazie alla simmetria della DFT, per calcolare N campioni della trasformata 
    // ho bisogno di soli N/2 thread
    int num_threads = N/2;  

    // controllo superfluo ma ce lo metto per sicurezza
    if(global_idx < num_threads) {
        // printf("\t -------------------- hello i'm %d\n", global_idx);
        if (N == 1) {
            output[0] = input[0];

            return;
        }

        if(global_idx == 0) {
            /*
                Lancio due kernel da N/2 thread

                Ogni campione k della trasformata necessita
                di TUTTI i campioni dello stadio precedente, divisi
                in campioni pari e dispari. 
            */

            // OCCHIO: quando i thread diventano meno di 1024
            int threads_per_block = (num_threads >= 1024) ? 1024 : num_threads;
            int num_blocks = (num_threads + threads_per_block-1) / threads_per_block;

            fft_device_side<<<num_blocks, threads_per_block>>>(input, output, step*2, N/2);
            fft_device_side<<<num_blocks, threads_per_block>>>(input + step, output + N/2, step*2, N/2);
            
            // Occhio che a quanto pare è deprecata, usare: -D CUDA_FORCE_CDP1_IF_SUPPORTED 
            cudaDeviceSynchronize();
        }
        // __syncthreads();


        // Combina i risultati
        
        // NB: sto ripetendo il calcolo del twiddle factor molte volte
        // forse posso risparmiarmi queste ripetizioni con la memoria condivisa?
        double phi = (-2*PI/N) * global_idx;  // Segno negativo per la FFT     
        complex twiddle = {
            cos(phi),
            sin(phi)
        };

        complex even = output[global_idx];

        complex temp = {
            twiddle.real * output[global_idx + N/2].real - twiddle.imag * output[global_idx + N/2].imag,
            twiddle.real * output[global_idx + N/2].imag + twiddle.imag * output[global_idx + N/2].real
        };

        output[global_idx].real = even.real + temp.real;
        output[global_idx].imag = even.imag + temp.imag;
        //relazione simmetrica
        output[global_idx + N/2].real = even.real - temp.real;
        output[global_idx + N/2].imag = even.imag - temp.imag;
    }
}







int main(int argc, char **argv) {
    // Configuration parameters
    const char* FILE_NAME = "test_voce.wav";
    
    /* ESECUZIONE SOLO LATO HOST */

    drwav wav_in;
    if (!drwav_init_file(&wav_in, FILE_NAME, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file %s.wav.\n", FILE_NAME);
        return 1;
    }

    size_t num_samples = wav_in.totalPCMFrameCount * wav_in.channels;
    printf("NUMERO DI CAMPIONI NEL FILE AUDIO SCELTO: %ld; -> %0.2f secondi\n\n", num_samples, (float)num_samples/SAMPLE_RATE);
    printf("\nSchiaccia un tasto per avviare...\n");
    getchar();

    // importante avere una potenza di 2, se non ce l'ho faccio del padding
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

    // calcolo la FFT sull'host
    convert_to_complex(signal_samples, complex_signal_samples, num_samples);
    double start = cpuSecond();
    fft_host_side(complex_signal_samples, fft_samples, num_samples);
    double elapsed_host = cpuSecond() - start;
    // printf("sumArrayOnHost Time elapsed %f sec\n", elapsed_host);



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
            // printf("Frequenza: %lf sembra essere un componente utile del segnale\n", frequency);
        }
    }

    printf("I dati dello spettro sono stati scritti in 'amplitude_spectrum.txt'.\n");
    fclose(output_file);



    







    /* ESECUZIONE CON GPU */










    printf("\nSchiaccia qualsiasi tasto per rifare con la GPU...\n");
    getchar();

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Alloca memoria per i campioni sul device
    complex* device_complex_signal_samples;
    complex* device_fft_samples;
    
    CHECK(cudaMalloc(&device_complex_signal_samples, num_samples*sizeof(complex)));
    CHECK(cudaMalloc(&device_fft_samples, num_samples*sizeof(complex)));
    
    CHECK(cudaMemcpy(device_complex_signal_samples, complex_signal_samples, num_samples*sizeof(complex), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(device_fft_samples, 0, num_samples*sizeof(complex)));


    // Invoke kernel

    /*
        Grazie alla simmetria della DFT, per calcolare N campioni
        della trasformata ho bisogno di soli N/2 thread

        Fai un po' di trial and error con threads_per_block
    */
    int num_threads = num_samples/2;  
    int threads_per_block = 512;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block; // dato che num_threads è una potenza di due, questo è superfluo

    start = cpuSecond();
    fft_device_side<<<num_blocks, threads_per_block>>>(device_complex_signal_samples, device_fft_samples, 1, num_samples);
    CHECK(cudaDeviceSynchronize());
    double elapsed_device  = cpuSecond() - start;

    // Copy kernel result back to host side
    complex* gpu_ref_fft_samples = (complex *)malloc(num_samples*sizeof(complex));
    CHECK(cudaMemcpy(gpu_ref_fft_samples, device_fft_samples, num_samples, cudaMemcpyDeviceToHost));
    
    checkResult(fft_samples, gpu_ref_fft_samples, num_samples);
    printf("Host: %f\n", elapsed_host);
    printf("Device: %f\n", elapsed_device);
    printf("SPEEDUP: %f\n", elapsed_host/elapsed_device);
    getchar();


    /* --- PARTE IFFT --- */



    

    // Preparazione del formato del file di output
    drwav_data_format format_out;
    format_out.container = drwav_container_riff;
    format_out.format = DR_WAVE_FORMAT_PCM;
    format_out.channels = 1;              // Mono
    format_out.sampleRate = SAMPLE_RATE;  // Frequenza di campionamento
    format_out.bitsPerSample = 16;        // 16 bit per campione

    // inizializzazione dati
    char generated_filename[100];  
    sprintf(generated_filename, "GPU-IFFT-generated-%s", FILE_NAME);

    // Inizializzazione del file di output
    drwav wav_out;
    if (!drwav_init_file_write(&wav_out, generated_filename, &format_out, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file di output %s.\n", generated_filename);
        return 1;
    }

    // mi assicuro di non imbrogliare riciclando i dati di prima
    memset(signal_samples, 0, num_samples*sizeof(short));
    memset(complex_signal_samples, 0, num_samples);
    
    ifft_host_side(gpu_ref_fft_samples, complex_signal_samples, num_samples);
    convert_to_short(complex_signal_samples, signal_samples, num_samples);

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    drwav_uninit(&wav_out); // Chiusura del file di output

    printf("File WAV %s creato con successo\n", generated_filename);





    // Free device global memory
    CHECK(cudaFree(device_complex_signal_samples));
    CHECK(cudaFree(device_fft_samples));
    // Free host memory
    free(signal_samples);
    free(complex_signal_samples);
    free(fft_samples);
    free(gpu_ref_fft_samples);

    return 0;
}
