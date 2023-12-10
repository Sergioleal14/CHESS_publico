#ifndef PARALLEL_CUH
#define PARALLEL_CUH


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "definiciones.h"
#include "movimientos.cuh"
#include "ataque.cuh"
#include "tablero.cuh"

#define NUM_HILOS 64
#define MAX_2MOVES 4000
#define MAX_MOVES 150


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

	////printf("El malloc normal no petó\n");
	for(int i=0;i<13;i++){
		error = cudaMalloc((void **)&(listas[i]), 10*sizeof(int));
		////printf("%s\n",cudaGetErrorString(error));
	}
	////printf("El malloc no petó\n");
	cudaMalloc((void **)&(history),MAXGAMEMOVES*sizeof(S_UNDO*));
	////printf("El malloc no petó, %d\n", tab->pList[0][0]);

	////printf("hola\n");

	cudaMemcpy(&((*d_tab)->KingSq), &(KingSq), sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->pceNum),&(pceNum), sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->material),&(material),sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&((*d_tab)->pieces), &(pieces) ,sizeof(int *), cudaMemcpyHostToDevice);

	cudaMemcpy(&((*d_tab)->pList), &(pList), sizeof(int **), cudaMemcpyHostToDevice);
	////printf("pre copia valores for\n");
	for(int i=0;i<13;i++){
		cudaMemcpy(&(pList[i]), &(listas[i]), sizeof(int *), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(&((*d_tab)->history), &(history) ,sizeof(S_UNDO **), cudaMemcpyHostToDevice);

	////printf("pre copia valores\n");
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

	////printf("fin de la copia\n");
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

__device__ void copy_tablero(TABLERO *dest, TABLERO *src){
	int i,j;
	dest->side = src->side;
	dest->AlPaso = src->AlPaso;
	dest->fiftyMove = src->fiftyMove;
	dest->histcont= src->histcont;
	dest->j_real = src->j_real;
	dest->enroque = src->enroque;

	for(i=0;i<2;i++){
		dest->KingSq[i] = src->KingSq[i];
		dest->material[i] = src->material[i];
	}

	for(i=0;i<13;i++){
		dest->pceNum[i] = src->pceNum[i];
	}


	for(i=0;i<NUM_CASILLAS;i++){
		dest->pieces[i] = src->pieces[i];
	}


	for(i=0;i<13;i++){
		for(j=0;j<10;j++){
			dest->pList[i][j] = src->pList[i][j];
		}
	}

	//Ver que hacer con los UNDOS

}

void cuda_retrieve_jugadas(MOVE ***host, MOVE **device,int count){
	int i;
	cudaError_t error;
	error = cudaMemcpy(*host, device, MAX_MOVES*sizeof(MOVE*), cudaMemcpyDeviceToHost);
	//printf("%s\n",error);
	for(i=0;i<count;i++){
		//int *piezas = (int*) malloc(3*sizeof(int));
		MOVE *m = (MOVE*) malloc(sizeof(MOVE));
		//error = cudaMemcpy(piezas, device[i]->piezas, 3*sizeof(int), cudaMemcpyDeviceToHost);
		////printf("1.- %s\n",cudaGetErrorString(error));
		////printf("hola\n");
		error = cudaMemcpy(m,(*host)[i],sizeof(MOVE), cudaMemcpyDeviceToHost);
		//printf("2.- %s\n",cudaGetErrorString(error));
		(*host)[i] = m;
		//host[i]->piezas = piezas;
	}
}

__global__ void generar_GPU(TABLERO *t, MOVE *jugada1,int *count1,MOVE *jugada2, int *count2, int *acc_counts ) {

  MOVE **mi_jugada2;
  MOVE **jugada1_local;
	TABLERO *mi_tablero;
	int acc = 0;
	int i;
	int pos_in,pos_fin;

	__shared__ int counts[NUM_HILOS + 1];

  ////printf("hola, %d\n", threadIdx.x);
	mi_tablero = Create_tablero();
	copy_tablero(mi_tablero,t);
	//memcpy(mi_tablero,t,sizeof(TABLERO));
	//*count1 = 3;
	////printf("le toca a %d\n", t->side);

  if(threadIdx.x == 0){
			//PrintBoard(mi_tablero);

		  jugada1_local = Generador_Movimientos(mi_tablero,count1);
      for (i=1; i < *count1; i++){

        jugada1[i-1] = *(jugada1_local[i]);
      }
			*count1 = *count1-1;
			counts[0] = 0;
	}

	__syncthreads();

	//Si no hay juagadas posibles es mate
	// En el código de Pprog siempre generábamos al menos un movimiento para evitar NULLs.
	// Por lo tanto, es primer movimiento es nulo.
	if (*count1 <= 1) return;

	if(threadIdx.x < *count1){
		HacerJugada(mi_tablero, &jugada1[threadIdx.x]);
		mi_jugada2 = Generador_Movimientos(mi_tablero,&counts[threadIdx.x +1]);
	}
	else{
			counts[threadIdx.x + 1] = -1;
			acc_counts[threadIdx.x + 1] = 0;
	}

	__syncthreads();
	//¡Pensar en mates!

	if(threadIdx.x == 0){
			//printf("acumulado\n");
			acc_counts[0] = 0;
			for(i = 1, acc = 0; (i< NUM_HILOS + 1) & (counts[i] != -1) ;i++){
					acc += counts[i] - 1;
					acc_counts[i] = acc;
					//printf("%d, ",acc_counts[i]);
			}
			*count2 = acc;
	}

	__syncthreads();

	 //Hace falta guardar las jugadas en una array plano para que podamos recorrerlas.
	 //Para saber de qué índice a qué indice se deben guardar creamos el array acc_counts
	 // que en la posición i guarda el comienzo para el hilo i, y en la posición i+1 su final

	pos_in = acc_counts[threadIdx.x];
	pos_fin = acc_counts[threadIdx.x + 1];
	for(int i = pos_in; i< pos_fin; i++){
			jugada2[i] = *(mi_jugada2[i - pos_in + 1]);
	}

	return;
}



MOVE* Generador_Movimientos_GPU(TABLERO* tab, int* count1, int** acc_counts, MOVE **jugadas){
	TABLERO *d_tab;
	int *count2;
	int *d_count1, *d_count2, *d_acc_counts;
	MOVE *jugada1, *jugada2;
	MOVE *d_jugada1, *d_jugada2;
	cudaError_t error;

	count2 = (int *)malloc(sizeof(int));
	*acc_counts = (int *)malloc(MAX_2MOVES*sizeof(int));

	jugada1 = (MOVE*) malloc(MAX_MOVES*sizeof(MOVE));
	jugada2 = (MOVE*) malloc(MAX_2MOVES*sizeof(MOVE));

	error = cudaMalloc((void **)&d_count1, sizeof(int));

	if(error != cudaSuccess){
			//printf("%s\n",cudaGetErrorString(error));
	}

	cudaMalloc((void **)&d_count2, sizeof(int));
	cudaMalloc((void **)&d_acc_counts, MAX_MOVES*sizeof(int));
	cudaMalloc((void **)&d_jugada1, MAX_MOVES*sizeof(MOVE));
	cudaMalloc((void **)&d_jugada2, MAX_2MOVES*sizeof(MOVE));

	cuda_create_tablero(tab,&d_tab);

	generar_GPU<<<1,NUM_HILOS>>>(d_tab,d_jugada1,d_count1,d_jugada2,d_count2,d_acc_counts);

	error = cudaMemcpy(count1, d_count1, sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(count2, d_count2, sizeof(int), cudaMemcpyDeviceToHost);

	//printf("%s\n",cudaGetErrorString(error));

	error = cudaMemcpy(jugada1, d_jugada1, MAX_MOVES*sizeof(MOVE), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(jugada2, d_jugada2, MAX_2MOVES*sizeof(MOVE), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(*acc_counts, d_acc_counts, MAX_MOVES*sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%s\n",cudaGetErrorString(error));

	//printf("Count1 es %d\n",*count1);
	//printf("Count2 es %d\n",*count2);

	cudaFree(d_count1);
	cudaFree(d_count2);
	cudaFree(d_acc_counts);
	cudaFree(d_jugada1);
	cudaFree(d_jugada2);

	free(count2);
	*jugadas = jugada2;
	//printf("JUGADAAAAA, from: %d, to %d, piece %d\n", jugada1[2].from, jugada1[2].to, jugada1[2].piezas[0]);
	MOVE *m = &(jugada1[2]);
	funcion_prueba(m);

	//PrintMove2(&(jugada1[2]));
	return jugada1;
}

#endif