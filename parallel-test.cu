#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include "definiciones.h"
//#include "tablero.cuh"

#include "busqueda.cuh"

#define FENP "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P5/RNBQKBNR w KQkq e6 0 1"

int main(void) {
  TABLERO *tab=NULL;
  int count1, count2,*acc_counts;
  MOVE *jugada1, *jugada2;
  int bestScore;
  MOVE **Best;
  INFO info;
  info.tiempo = 0;
  info.maxtemp = 1000000000;
	info.visited=0;
	info.stop=FALSE;
  info.depth = 2;

  Best=(MOVE**)malloc(sizeof(MOVE*));

  tab = Create_tablero();
  LeerFen(START_FEN,tab);

  //jugada1 = Generador_Movimientos_GPU(tab, &count1, &acc_counts, &jugada2);
  //PrintBoard(tab);
  printf("ANTES DE LLAMAR A ALPHA BETA");
  bestScore = AlphaBeta(-50000,50000,4,tab,&info,Best,NULL,0,0);
  printf("bestScore: %d\n",bestScore);
  printf("visited: %d\n",info.visited);
  printf("fin del programa\n");
  return 0;
}