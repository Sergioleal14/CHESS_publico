#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "definiciones.h"

#define NUM_HILOS 512
#define MAX_2MOVES 4000
#define MAX_MOVES 60

extern TABLERO* Create_tablero();
//__global__ MOVE **Generador_Movimientos(TABLERO *t, int *count);
__host__ __device__ MOVE **Generador_Movimientos2(TABLERO *t, int *count){
	MOVE **m;
	if(!t) return NULL;

	m = (MOVE**) malloc(sizeof(MOVE*));
	//m[0] = insert_move(0,A1,A1,0,0,0,0);
	*count = 1;
	return m;
}

__host__ __device__ int HacerJugada2(TABLERO *t,MOVE *m){
	return 0;
}

__global__ void generar_GPU(TABLERO *t, MOVE ***jugada1,int *count1,MOVE ***jugada2, int *count2, int *acc_counts ) {

  MOVE **mi_jugada2;
	TABLERO mi_tablero[1];
	int acc = 0;
	int i;
	int pos_in,pos_fin;

	extern __shared__ int counts[NUM_HILOS + 1];
	//extern __shared__ int acc_counts[NUM_HILOS + 1];

	memcpy(mi_tablero,t,sizeof(TABLERO));

	//El primer hilo genera los primeros movimientos
	if(blockIdx.x == 0){
			*jugada1 = Generador_Movimientos2(mi_tablero,count1);
			counts[0] = 0;
	}

	__syncthreads();
	/*
	 //Si no hay juagadas posibles es mate
	 // En el código de Pprog siempre generábamos al menos un movimiento para evitar NULLs.
	 // Por lo tanto, es primer movimiento es nulo.
	if (*count1 <= 1) return;

	if(blockIdx.x < *count1 - 1){
			HacerJugada2(mi_tablero,(*jugada1)[blockIdx.x + 1]);
			mi_jugada2 = Generador_Movimientos2(t,&counts[blockIdx.x +1]);
	}
	else{
			counts[blockIdx.x + 1] = -1;
			acc_counts[blockIdx.x + 1] = 0;
	}

	__syncthreads();
	//¡Pensar en mates!

	if(blockIdx.x == 0){
			acc_counts[0] = 0;
			for(i = 1, acc = 0; (i< NUM_HILOS + 1) & (counts[i] != -1) ;i++){
					acc += counts[i] - 1;
					acc_counts[i] = acc;
			}
			*count2 = acc;
	}

	__syncthreads();

	 //Hace falta guardar las jugadas en una array plano para que podamos recorrerlas.
	 //Para saber de qué índice a qué indice se deben guardar creamos el array acc_counts
	 // que en la posición i guarda el comienzo para el hilo i, y en la posición i+1 su final

	pos_in = acc_counts[blockIdx.x];
	pos_fin = acc_counts[blockIdx.x + 1];
	for(int i = pos_in; i< pos_fin; i++){
			*jugada2[i] = mi_jugada2[i - pos_in + 1];
	}*/
	return;
}

int main(void) {
		TABLERO *tab=NULL, *d_tab;
		int *count1, *count2,*acc_counts;
		int *d_count1, *d_count2, *d_acc_counts;
		MOVE **jugada1, **jugada2;
		MOVE **d_jugada1, **d_jugada2;

		count1 = (int *)malloc(sizeof(int));
		count2 = (int *)malloc(sizeof(int));
		acc_counts = (int *)malloc(MAX_2MOVES*sizeof(int));

		jugada1 = (MOVE**) malloc(MAX_MOVES*sizeof(MOVE*));
		jugada2 = (MOVE**) malloc(MAX_2MOVES*sizeof(MOVE*));

	 	cudaMalloc((void **)&d_count1, sizeof(int));
		cudaMalloc((void **)&d_count2, sizeof(int));
		cudaMalloc((void **)&d_acc_counts, MAX_2MOVES*sizeof(int));
	 	cudaMalloc((void **)&d_jugada1, MAX_MOVES*sizeof(MOVE*));
		cudaMalloc((void **)&d_jugada2, MAX_2MOVES*sizeof(MOVE*));
		cudaMalloc((void **)&d_tab, sizeof(TABLERO));

		tab = Create_tablero();
		LeerFen(START_FEN, tab);
		cudaMemcpy(d_tab, &tab, sizeof(TABLERO), cudaMemcpyHostToDevice);

		printf("comienzo de kernel\n");
		//generar_GPU<<<1,NUM_HILOS>>>(d_tab,&d_jugada1,d_count1,&d_jugada2,d_count2,d_acc_counts);

		printf("Fin de kernel\n");

		//cudaMemcpy(&count2, d_count2, sizeof(int), cudaMemcpyDeviceToHost);

		//printf("Hay %d jugadas\n",*count2);

		return 0;
}
