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
#ifndef ZSTD_WRAPPER_H
#define ZSTD_WRAPPER_H

// #include <zstd.h>
# include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// to use advanced functions
#define ZSTD_STATIC_LINKING_ONLY 
#define HUF_STATIC_LINKING_ONLY
#define FSE_STATIC_LINKING_ONLY
#define ZDICT_STATIC_LINKING_ONLY

// zstd headers
#include <3rdparty/zstd-cbench/lib/zstd.h>
// #include <zstd_ext.h>
#include <3rdparty/zstd-cbench/lib/common/fse.h>
// #include <3rdparty/zstd-cbench/lib/common/fse_tans.h>
#include <3rdparty/zstd-cbench/lib/common/huf.h>
#include <3rdparty/zstd-cbench/lib/common/zstd_internal.h>
// zdict headers
#include <3rdparty/zstd-cbench/lib/dictBuilder/zdict.h>

#include "zstd_common.h"

// pybind for ZSTD_Sequence (offset, litLength, matchLength, rep)
#define PYBIND_ZSTD_Sequence std::tuple<unsigned int, unsigned int, unsigned int, unsigned int>

// LZ77 data representation, including literals and ZSTD_Sequences
#define ZSTD_LZ77_Data std::tuple<py::bytes, std::vector<PYBIND_ZSTD_Sequence>>

// LZ77 phrase format
struct PYBIND_ZSTD_Sequence_Ext{
    unsigned int offset;
    unsigned int matchLength;
    py::bytes literals;
    unsigned int rep;

    PYBIND_ZSTD_Sequence_Ext()=default;
    PYBIND_ZSTD_Sequence_Ext(unsigned int offset, unsigned int matchLength, py::bytes literals, unsigned int rep) :
        offset(offset),
        matchLength(matchLength),
        literals(literals),
        rep(rep)
    {}
    PYBIND_ZSTD_Sequence_Ext(ZSTD_Sequence_Ext seq) :
        offset(seq.offset),
        matchLength(seq.matchLength),
        literals(py::bytes((char*)seq.lit, seq.litLength)),
        rep(seq.rep)
    {}
};
#define PYBIND_Tuple_ZSTD_Sequence_Ext std::tuple<unsigned int, unsigned int, py::bytes, unsigned int>

// parameter config (maybe use py::object for config)?
#define ZSTD_cParameter_Configs std::map<ZSTD_cParameter, int>


#endif
