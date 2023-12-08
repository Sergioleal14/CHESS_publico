%%writefile /content/CHESS_publico/parallel.cu

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "definiciones.h"
#include "movimientos.cuh"
#include "ataque.cuh"
#include "tablero.cuh"


#define NUM_HILOS 1
#define MAX_2MOVES 4000
#define MAX_MOVES 60

__host__ __device__ MOVE **Generador_Movimientos2(TABLERO *t, int *count){
	MOVE **m;
	if(!t) return NULL;

	m = (MOVE**) malloc(sizeof(MOVE*));
	//m[0] = insert_move(0,A1,A1,0,0,0,0);
	*count = 1;
  return m;
}

void cuda_create_tablero(TABLERO *tab, TABLERO **d_tab){
	int *pieces;//tablero en si
	int *KingSq;
	int side;
	int AlPaso;
	int fiftyMove;
	int histcont; //numero de elementos de history
	int j_real;//numero de jugadas
	int enroque;
	int *pceNum;//numero de piezas de cada tipo
	int *material;
	S_UNDO **history;
	int **pList;//lista por cada tipo de pieza guarda su casilla
	int *listas[13];
	cudaError_t error;

	cudaMalloc((void**)d_tab, sizeof(TABLERO));
	cudaMalloc((void**)&(KingSq), 2 * sizeof(int));

	cudaMalloc((void **)&(pceNum), 13*sizeof(int));
	cudaMalloc((void **)&(material), 2*sizeof(int));
	cudaMalloc((void **)&(pieces), NUM_CASILLAS*sizeof(int));
	cudaMalloc((void **)&(pList), 13*sizeof(int*));

	//printf("El malloc normal no petó\n");
	for(int i=0;i<13;i++){
		error = cudaMalloc((void **)&(listas[i]), 10*sizeof(int));
		//printf("%s\n",cudaGetErrorString(error));
	}
	//printf("El malloc no petó\n");
	cudaMalloc((void **)&(history),MAXGAMEMOVES*sizeof(S_UNDO*));
	//printf("El malloc no petó, %d\n", tab->pList[0][0]);

	//printf("hola\n");

	cudaMemcpy(&((*d_tab)->KingSq), &(KingSq), sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->pceNum),&(pceNum), sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->material),&(material),sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->pieces), &(pieces) ,sizeof(int *), cudaMemcpyHostToDevice);

	cudaMemcpy(&((*d_tab)->pList), &(pList), sizeof(int **), cudaMemcpyHostToDevice);
	//printf("pre copia valores for\n");
	for(int i=0;i<13;i++){
		cudaMemcpy(&(pList[i]), &(listas[i]), sizeof(int *), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(&((*d_tab)->history), &(history) ,sizeof(S_UNDO **), cudaMemcpyHostToDevice);

	printf("pre copia valores\n");
	cudaMemcpy(KingSq, tab->KingSq, 2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->side), &(tab->side), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->AlPaso), &(tab->AlPaso), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->fiftyMove), &(tab->fiftyMove), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->histcont), &(tab->histcont), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->j_real), &(tab->j_real), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->enroque), &(tab->enroque), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pceNum, tab->pceNum, 13*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(material, tab->material, 2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pieces, tab->pieces, NUM_CASILLAS*sizeof(int), cudaMemcpyHostToDevice);

	for(int i=0;i<13;i++){
		cudaMemcpy(listas[i], tab->pList[i], 10*sizeof(int), cudaMemcpyHostToDevice);
	}

	printf("fin de la copia\n");
}

void cuda_copy_tablero(TABLERO *tab, TABLERO *d_tab){

	cudaMemcpy(d_tab->KingSq,tab->KingSq, 2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tab->pceNum,tab->pceNum, 13*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tab->material,tab->material, 2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tab->pieces, tab->pieces ,NUM_CASILLAS*sizeof(int), cudaMemcpyHostToDevice);

	for(int i=0;i<13;i++){
		cudaMemcpy(d_tab->pList[i],tab->pList[i], 10*sizeof(int), cudaMemcpyHostToDevice);
	}

	//Problema con UNDOS
}

__global__ void generar_GPU(TABLERO *t, MOVE **jugada1,int *count1,MOVE **jugada2, int *count2, int *acc_counts ) {

  MOVE **mi_jugada2;
  MOVE **jugada1_local;
	TABLERO mi_tablero[1];
	int acc = 0;
	int i;
	int pos_in,pos_fin;

	__shared__ int counts[NUM_HILOS + 1];

  printf("hola, %d\n", threadIdx.x);
	//memcpy(mi_tablero,t,sizeof(TABLERO));
	*count1 = 3;
	printf("le toca a %d\n", t->side);

  if(threadIdx.x == 0){
			printf("soy el 0\n");
			printf("t kg sq es %d, %d\n",t->KingSq[0],t->KingSq[1]);
			PrintBoard(t);
			printf("piece list es: ");
			for(i=0;i < 13; i++){
				for (int j = 0; j < 10; j ++){
					printf("%d, ",t->pList[i][j]);
				}
				printf("\n");
			}
		  jugada1_local = Generador_Movimientos(t,count1);
      for (i=0; i < *count1; i++){
				
        jugada1[i] = jugada1_local[i];
      }
			counts[0] = 0;

			int legal = HacerJugada(t, jugada1_local[1]);
			printf("JUGADA LEGAL: %d\n", legal);
			PrintBoard(t);

	}

	__syncthreads();
	return;
}

int main(void) {
		TABLERO *tab=NULL, *d_tab;
		int *count1, *count2,*acc_counts;
		int *d_count1, *d_count2, *d_acc_counts;
		MOVE **jugada1, **jugada2;
		MOVE **d_jugada1, **d_jugada2;
		cudaError_t error;
		int *KingSq;

    count1 = (int *)malloc(sizeof(int));
		count2 = (int *)malloc(sizeof(int));
		acc_counts = (int *)malloc(MAX_2MOVES*sizeof(int));

		jugada1 = (MOVE**) malloc(MAX_MOVES*sizeof(MOVE*));
		jugada2 = (MOVE**) malloc(MAX_2MOVES*sizeof(MOVE*));

	 	error = cudaMalloc((void **)&d_count1, sizeof(int));

		if(error != cudaSuccess){
        printf("%s\n",cudaGetErrorString(error));
    }

		cudaMalloc((void **)&d_count2, sizeof(int));
		cudaMalloc((void **)&d_acc_counts, MAX_2MOVES*sizeof(int));
	 	cudaMalloc((void **)&d_jugada1, MAX_MOVES*sizeof(MOVE*));
		cudaMalloc((void **)&d_jugada2, MAX_2MOVES*sizeof(MOVE*));

		printf("Antes del malloc\n");


		tab = Create_tablero();
		LeerFen(START_FEN, tab);
		cuda_create_tablero(tab,&d_tab);

		/*cudaMalloc((void**)&d_tab, sizeof(TABLERO));
		cudaMalloc((void**)&(KingSq), 2 * sizeof(int));
		cudaMemcpy(&(d_tab->KingSq), &(KingSq), sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(KingSq, tab->KingSq, 2*sizeof(int), cudaMemcpyHostToDevice);*/


    *count1 = 2;
    cudaMemcpy(d_count1, &count1, sizeof(int), cudaMemcpyHostToDevice);


		printf("fin del copy\n");
		//cudaMemcpyToSymbol(d_tab,tab,sizeof(TABLERO),cudaMemcpyHostToDevice);


    generar_GPU<<<1,NUM_HILOS>>>(d_tab,d_jugada1,d_count1,d_jugada2,d_count2,d_acc_counts);

    error = cudaMemcpy(count1, d_count1, sizeof(int), cudaMemcpyDeviceToHost);
    printf("fin del kernel\n");


    //if(error != cudaSuccess){
    printf("%s\n",cudaGetErrorString(error));
    //}

    printf("Count es %d\n",*count1);
		return 0;
}