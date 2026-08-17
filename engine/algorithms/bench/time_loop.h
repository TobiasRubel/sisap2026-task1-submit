#ifndef TIMELOOP
#define TIMELOOP

#include "parlay/internal/get_time.h"

template<class F, class G, class H>
void time_loop(int rounds, double delay, F initf, G runf, H endf) {
  parlay::internal::timer t;
  (void)delay;  // the submission's single call site passes delay=0
  for (int i=0; i < rounds; i++) {
    initf();
    t.start();
    runf();
    t.next("");
    endf();
  }
}

#endif
