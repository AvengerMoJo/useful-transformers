#ifndef _LIB_FP16_H_
#define _LIB_FP16_H_

// __fp16 is a native keyword on ARM GCC/Clang and on x86 Clang, but was
// removed on x86 GCC 13+. Map it to the ISO C23 _Float16 type there, which is
// also IEEE binary16 and is supported as a first-class type on x86.
#if defined(__GNUC__) && !defined(__clang__) && \
    (defined(__x86_64__) || defined(__i386__) || defined(__i686__))
typedef _Float16 __fp16;
#endif

#endif  // _LIB_FP16_H_