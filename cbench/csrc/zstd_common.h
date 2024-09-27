/*
 * Copyright (c) Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/*
 * This header file has common utility functions used in examples.
 */
#ifndef ZSTD_COMMON_H
#define ZSTD_COMMON_H

#include <stdlib.h>    // malloc, free, exit
#include <stdio.h>     // fprintf, perror, fopen, etc.
#include <string.h>    // strerror
#include <errno.h>     // errno
#include <sys/stat.h>  // stat
#include <3rdparty/zstd-cbench/lib/zstd.h>


/*
 * Define the returned error code from utility functions.
 */
typedef enum {
    ERROR_fsize = 1,
    ERROR_fopen = 2,
    ERROR_fclose = 3,
    ERROR_fread = 4,
    ERROR_fwrite = 5,
    ERROR_loadFile = 6,
    ERROR_saveFile = 7,
    ERROR_malloc = 8,
    ERROR_largeFile = 9,
} COMMON_ErrorCode;

/*! CHECK
 * Check that the condition holds. If it doesn't print a message and die.
 */
#define CHECK(cond, ...)                        \
    do {                                        \
        if (!(cond)) {                          \
            fprintf(stderr,                     \
                    "%s:%d CHECK(%s) failed: ", \
                    __FILE__,                   \
                    __LINE__,                   \
                    #cond);                     \
            fprintf(stderr, "" __VA_ARGS__);    \
            fprintf(stderr, "\n");              \
            exit(1);                            \
        }                                       \
    } while (0)

/*! CHECK_ZSTD
 * Check the zstd error code and die if an error occurred after printing a
 * message.
 */
#define CHECK_ZSTD(fn, ...)                                      \
    do {                                                         \
        size_t const err = (fn);                                 \
        CHECK(!ZSTD_isError(err), "%s", ZSTD_getErrorName(err)); \
    } while (0)



// from tests/fuzzer.c

// static void FUZ_decodeSequences(BYTE* dst, ZSTD_Sequence* seqs, size_t seqsSize,
//                                 BYTE* src, size_t size, ZSTD_sequenceFormat_e format)
// {
//     size_t i;
//     size_t j;
//     for(i = 0; i < seqsSize; ++i) {
//         assert(dst + seqs[i].litLength + seqs[i].matchLength <= dst + size);
//         assert(src + seqs[i].litLength + seqs[i].matchLength <= src + size);
//         if (format == ZSTD_sf_noBlockDelimiters) {
//             assert(seqs[i].matchLength != 0 || seqs[i].offset != 0);
//         }

//         memcpy(dst, src, seqs[i].litLength);
//         dst += seqs[i].litLength;
//         src += seqs[i].litLength;
//         size -= seqs[i].litLength;

//         if (seqs[i].offset != 0) {
//             for (j = 0; j < seqs[i].matchLength; ++j)
//                 dst[j] = dst[j - seqs[i].offset];
//             dst += seqs[i].matchLength;
//             src += seqs[i].matchLength;
//             size -= seqs[i].matchLength;
//         }
//     }
//     if (format == ZSTD_sf_noBlockDelimiters) {
//         memcpy(dst, src, size);
//     }
// }


#endif
