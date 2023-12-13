#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <time.h>
#include "definiciones.h"
#define N 10

int AlphaBeta(int alpha, int beta, int depth, TABLERO *pos, INFO *info,MOVE** Best);

#define FENP "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P5/RNBQKBNR w KQkq e6 0 1"

int main(void) {
  struct timeval inicio, fin;
  TABLERO *tab=NULL;
  int count1, count2,*acc_counts;
  MOVE *jugada1, *jugada2;
  int bestScore;
  long time;
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
  for(int i = 0; i < N;i++){
    gettimeofday(&inicio, NULL);
    bestScore = AlphaBeta(-50000,50000,2,tab,&info,Best);
    gettimeofday(&fin, NULL);
    time= (fin.tv_sec - inicio.tv_sec) * 1000000L + (fin.tv_usec - inicio.tv_usec);
    printf("%ld, ",time);
  }
  return 0;
}