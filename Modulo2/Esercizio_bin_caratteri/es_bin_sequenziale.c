#include <stdio.h>
#include <stdlib.h>

long get_length_in_bytes(FILE* input_file) {
    fseek(input_file, 0, SEEK_END);
    long dimensione_file = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    return dimensione_file;
}

/*
    7 bin:
        - (a; d)
        - (e; h)
        - (i; l)
        - (m; p)
        - (q; t)
        - (u; x)
        - (y; z)
*/
void categorize_in_chuncks(FILE* input_file, int chunk_dim, int* output) {
    char buffer;
    int cur_letti = 0;

    while ( cur_letti < chunk_dim ) {
        int nread = fread(&buffer, sizeof(char), sizeof(buffer), input_file);
        if(nread <= 0) {
            fprintf(stderr, "questo non dovrebbe essere successo\n");
            exit(1);
        }

        cur_letti++;

        if( 'a' <= buffer <= 'd' ) {
            output[0]++;
        }
        else if( 'e' <= buffer <= 'h' ) {
            output[1]++;
        }
        else if( 'i' <= buffer <= 'l' ) {
            output[2]++;
        }
        else if( 'm' <= buffer <= 'p' ) {
            output[3]++;
        }
        else if( 'q' <= buffer <= 't' ) {
            output[4]++;
        }
        else if( 'u' <= buffer <= 'x' ) {
            output[5]++;
        }
        else if( 'y' <= buffer <= 'z' ) {
            output[6]++;
        }
        else{
            // carattere da non considerare, fai niente
        }
    }
}

void print_bins(int* bins) {
    printf("{%d, %d, %d, %d, %d, %d, %d}\n", bins[0], bins[1], bins[2], bins[3], bins[4], bins[5], bins[6]);
}

int main() {
    const char* FILENAME = "testo_lungo.txt";
    int bins[7] = {0, 0, 0, 0, 0, 0, 0};

    FILE *input_file = fopen(FILENAME, "rb");
    if (input_file == NULL) {
        fprintf(stderr, "Errore nell'aprire il file: %s.\n", FILENAME);
        return 1;
    }

    int dimensione_file = get_length_in_bytes(input_file);

    categorize_in_chuncks(input_file, dimensione_file, bins);

    fclose(input_file);


    // Stampare la dimensione del file
    printf("La dimensione del file %s è: %ld byte\n", FILENAME, dimensione_file);
    print_bins(bins);

    return 0;
}