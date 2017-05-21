# Optimizing Matrix-Matrix multiple with C instrinsics and vectorization

This code applied multiple approaches to optimizing matrix-matrix multiply with vectorization, blocking at both the cache and register level. These optimizations are measured using the PAPI library for counting cache misses, cache hits, loads & stores, etc. 
