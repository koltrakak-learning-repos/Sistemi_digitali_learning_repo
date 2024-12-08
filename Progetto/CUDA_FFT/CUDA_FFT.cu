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

void do_fft_stuff_on_host(const char* filename) {
    drwav wav_in;
    if (!drwav_init_file(&wav_in, filename, NULL)) {
        fprintf(stderr, "Errore nell'aprire il file %s.wav.\n", filename);
        exit(1);
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

    // Alloco memoria per:
    //  - dati audio (PCM a 16 bit)
    //  - dati audio (PCM a 16 bit) convertiti in numeri complessi
    //  - dati delle sinusoidi prodotte dalla fft
    short* signal_samples = (short*)malloc(num_samples * sizeof(short));
    if (signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        exit(1);
    }
    complex* complex_signal_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (complex_signal_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        exit(1);
    }
    complex* fft_samples = (complex*)malloc(num_samples * sizeof(complex));
    if (fft_samples == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        exit(1);
    }

    // Lettura dei dati audio dal file di input
    size_t frames_read = drwav_read_pcm_frames_s16(&wav_in, wav_in.totalPCMFrameCount, signal_samples);
    if (frames_read != wav_in.totalPCMFrameCount) {
        fprintf(stderr, "Errore durante la lettura dei dati audio.\n");
        exit(1);
    }
    drwav_uninit(&wav_in); 




    // calcolo la FFT sull'host
    convert_to_complex(signal_samples, complex_signal_samples, num_samples);
    double start = cpuSecond();
    fft_host_side(complex_signal_samples, fft_samples, num_samples);
    double elapsed = cpuSecond() - start;
    // printf("sumArrayOnHost Time elapsed %f sec\n", elapsed);




    // Calcola e salvo l'ampiezza per ciascuna frequenza
    FILE *output_file = fopen("amplitude_spectrum.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file di output.\n");
        exit(1);
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
    sprintf(generated_filename, "IFFT-generated-%s", filename);
    // mi assicuro di non imbrogliare riciclando i dati di prima
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
        exit(1);
    }
    
    ifft_host_side(fft_samples, complex_signal_samples, num_samples);
    convert_to_short(complex_signal_samples, signal_samples, num_samples);

    // Scrittura dei dati audio nel file di output
    drwav_write_pcm_frames(&wav_out, num_samples, signal_samples);
    drwav_uninit(&wav_out); // Chiusura del file di output

    printf("File WAV %s creato con successo\n", generated_filename);

    free(signal_samples);
    free(complex_signal_samples);
    free(fft_samples);
}








/* KERNEL GPU */





// __global__ void fft_device_side(complex *x, complex *X, int N) {
//     /*
//         Quando lancio un'altra griglia questo si ripete... è giusto???
//     */
//     int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

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

//     /*
//         OCCHIO QUESTO è SBAGLIATO!
//         La memoria locale è privata per ogni thread

//         Mi sa che mi tocca allocare dinamicamente con CudaMalloc...
//         (e anche deallocare con CudaFree)

//         Posso usare device?
//     */
    
//     complex signal_even[N/2];
//     complex signal_odd[N/2];
//     complex trasformata_even[N/2];
//     complex trasformata_odd[N/2];

//     cudaMalloc((void**)&signal_even, N/2*sizeof(complex));
//     cudaMalloc((void**)&signal_odd, N/2*sizeof(complex));
//     cudaMalloc((void**)&trasformata_even, N/2*sizeof(complex));
//     cudaMalloc((void**)&trasformata_odd, N/2*sizeof(complex));

//     /*
//         Solamente il primo blocco di ogni stadio lancia un kernel ricorsivo?
//         Sono abbastanza sicuro di si!
//     */
//     if(global_idx == 0) {
//         /*
//         Ogni frequenza k della trasformata necessita
//         di TUTTI i campioni dello stadio precedente, divisi
//         in campioni pari e dispari. Come faccio a selezionarli 
//         con ogni thread.

//         Forse questo lo posso preparare lato CPU 
//         passando alla funzione un array di array
//         indicizzato con il tid???
//         */

//         // Separazione dei campioni pari e dispari
//         for(int i = 0; i < N/2; i++) {
//             signal_even[i] = x[2*i];
//             signal_odd[i] = x[2*i + 1];
//         }

//         int threads_per_block = 1024;
//         int num_blocks = (N + threads_per_block-1) / threads_per_block;

//         // Ricorsivamente calcola la FFT per pari e dispari
//         fft_device_side<<<num_blocks, threads_per_block>>>(signal_even, trasformata_even, N/2);
//         fft_device_side<<<num_blocks, threads_per_block>>>(signal_odd, trasformata_odd, N/2);

//         /*
//             Qua c'è da sincronizzare per forza dato?
//             La componente k-esima della trasformata ha delle dipendenze 
//             con quelle calcolate dagli altri thread? 

//             cuda_device_synchronize();
//         */
//     }


//     /*
//         Qua al posto di un ciclo avrà N/2 thred che mi calcolano
//         le trasformate dei vari stadi 

//         k === tid; o un qualcosa del genere
//     */

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



__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}




int main(int argc, char **argv) {
    // Configuration parameters
    const char* FILE_NAME = "test_voce.wav";
    
    printf("%s Starting...\n", argv[0]);


    do_fft_stuff_on_host(FILE_NAME);






    /* ESECUZIONE SOLO LATO HOST */
    /* TODO: refactor in una funzione a parte */










    







    /* ESECUZIONE CON GPU */
    /* TODO: refactor in una funzione a parte */






    printf("Vuoi rifare con la GPU?\n");
    getchar();

    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    
    /*
        Numero di frequenze da calcolare: N/2
    */
    int numThreads = 1 << 20;  
    /*
        Fai un po' di trial and error, direi di partire con 1024
    */
    int threadsPerBlock = 1024;
    int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    // // Alloca memoria per i campioni sul device
    // short* device_signal_samples;
    // complex* device_dft_samples
    
    // CHECK(cudaMalloc((short**)&device_signal_samples, num_samples*sizeof(short)));
    // CHECK(cudaMalloc((complex**)&device_dft_samples, (num_samples/2)*sizeof(complex)));
    
    // CHECK(cudaMemcpy(device_signal_samples, signal_samples, num_samples*sizeof(short), cudaMemcpyHostToDevice));





    // // TODO: continualo...




    // // Invoke kernel
    // int gridSize = (nElem + blockSize - 1) / blockSize;

    // iStart = cpuSecond();
    // sumArrayOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, nElem);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
    // printf("sumArrayOnGPU <<<%d, %d>>> Time elapsed %f sec\n", gridSize, blockSize, iElaps);

    // // Copy kernel result back to host side
    // CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // // Check device results
    // checkResult(hostRef, gpuRef, nElem);

    // // Free device global memory
    // CHECK(cudaFree(d_A));
    // CHECK(cudaFree(d_B));
    // CHECK(cudaFree(d_C));

    // // Free host memory
    // free(h_A);
    // free(h_B);
    // free(hostRef);
    // free(gpuRef);

    // return 0;
}
