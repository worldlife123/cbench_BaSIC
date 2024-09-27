/* Copyright (c) 2021-2022, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "../search_tree.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// to use advanced functions
#define FSE_STATIC_LINKING_ONLY

// zstd headers
#include <3rdparty/zstd-cbench/lib/zstd.h>
#include <3rdparty/zstd-cbench/lib/common/fse.h>
#include <3rdparty/zstd-cbench/lib/common/error_private.h>

namespace py = pybind11;

struct TansSymbol {
  uint16_t value;
  uint16_t index;
  bool bypass; // bypass flag to write raw bits to the stream
};

typedef struct {
    // ptrdiff_t   value;
    uint64_t value;
    // const uint16_t* stateTable;
    // const FSE_symbolCompressionTransform* symbolTT;
    unsigned stateLog;
} TansState;

typedef struct {
    const void* stateTable;
    const void* symbolTT;
    // unsigned stateLog;
} TansCTable_t;
typedef uint32_t TansCTable;

typedef struct {
    const FSE_decode_t* table;
} TansDTable_t;
typedef uint32_t TansDTable;

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class BufferedTansEncoder {
public:
  BufferedTansEncoder() = default;

  BufferedTansEncoder(const BufferedTansEncoder &) = delete;
  BufferedTansEncoder(BufferedTansEncoder &&) = delete;
  BufferedTansEncoder &operator=(const BufferedTansEncoder &) = delete;
  BufferedTansEncoder &operator=(BufferedTansEncoder &&) = delete;

  

  void encode_with_indexes(const std::vector<int32_t> &symbols,
                           const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<TansCTable>> &ctables,
                           const std::vector<int32_t> &offsets);
  void encode_with_indexes_np(const py::array_t<int32_t> &symbols,
                              const py::array_t<int32_t> &indexes,
                              const py::array_t<TansCTable, py::array::c_style | py::array::forcecast> &ctables,
                              const py::array_t<int32_t> &offsets);
  py::bytes flush();

private:
  std::vector<TansSymbol> _syms;
  std::vector<TansCTable_t> _ctables_t;
  std::vector<std::vector<TansCTable>> _ctables;
};

class TansEncoder {
public:
  TansEncoder() = default;

  TansEncoder(const TansEncoder &) = delete;
  TansEncoder(TansEncoder &&) = delete;
  TansEncoder &operator=(const TansEncoder &) = delete;
  TansEncoder &operator=(TansEncoder &&) = delete;

  py::bytes encode_with_indexes(const std::vector<int32_t> &symbols,
                                const std::vector<int32_t> &indexes,
                                const std::vector<std::vector<TansCTable>> &ctables,
                                const std::vector<int32_t> &offsets);
  py::bytes encode_with_indexes_np(const py::array_t<int32_t> &symbols,
                              const py::array_t<int32_t> &indexes,
                              const std::optional<py::array_t<TansCTable, py::array::c_style | py::array::forcecast>> &ctables,
                              const std::optional<py::array_t<int32_t>> &offsets);
  py::bytes encode_autoregressive_np(const py::array_t<int32_t> &symbols,
                              const py::array_t<int32_t> &indexes,
                              const py::array_t<int32_t> &ar_offsets,
                              const std::optional<py::array_t<TansCTable, py::array::c_style | py::array::forcecast>> &ctables,
                              const std::optional<py::array_t<int32_t>> &offsets,
                              const std::optional<uint32_t> max_symbol_value);

  void create_ctable_using_cnt(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> count, 
                                 unsigned maxSymbolValue, 
                                 unsigned tableLog);
  
  void set_ctables(const py::array_t<TansCTable, py::array::c_style> &ctables);

private:
  py::array_t<TansCTable, py::array::c_style> _ctables;
  std::vector<TansCTable_t> _ctables_t;
  std::vector<std::vector<std::vector<TansCTable_t>>> _ctables_t_3d;
  std::vector<std::vector<TansCTable_t>> _ctables_t_2d;
  SearchTreeNode<TansCTable_t> _ctables_t_st;
  unsigned _maxSymbolValue;
  unsigned _tableLog;

};

class TansDecoder {
public:
  TansDecoder() = default;

  TansDecoder(const TansDecoder &) = delete;
  TansDecoder(TansDecoder &&) = delete;
  TansDecoder &operator=(const TansDecoder &) = delete;
  TansDecoder &operator=(TansDecoder &&) = delete;

  std::vector<int32_t>
  decode_with_indexes(const std::string &encoded,
                      const std::vector<int32_t> &indexes,
                      const std::vector<std::vector<TansDTable>> &dtables,
                      const std::vector<int32_t> &offsets);
  py::array_t<int32_t>
  decode_with_indexes_np(const std::string &encoded,
                         const py::array_t<int32_t> &indexes,
                         const std::optional<py::array_t<TansDTable, py::array::c_style | py::array::forcecast>> &dtables,
                         const std::optional<py::array_t<int32_t>> &offsets);
  py::array_t<int32_t>
  decode_autoregressive_np(const std::string &encoded,
                         const py::array_t<int32_t> &indexes,
                         const py::array_t<int32_t> &ar_offsets,
                         const std::optional<py::array_t<TansDTable, py::array::c_style | py::array::forcecast>> &dtables,
                         const std::optional<py::array_t<int32_t>> &offsets,
                         const std::optional<uint32_t> max_symbol_value);

  void create_dtable_using_cnt(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> count, 
                                 unsigned maxSymbolValue, 
                                 unsigned tableLog);

  void set_stream(const std::string &stream);

  std::vector<int32_t>
  decode_stream(const std::vector<int32_t> &indexes,
                const std::vector<std::vector<TansDTable>> &dtables,
                const std::vector<int32_t> &offsets);
  
  void set_dtables(const py::array_t<TansDTable, py::array::c_style> &dtables);

private:
  TansState _state;
  BIT_DStream_t _bitD;
  unsigned _maxSymbolValue;
  unsigned _tableLog;
  py::array_t<TansDTable, py::array::c_style> _dtables;
  std::vector<TansDTable_t> _dtables_t;
  std::vector<std::vector<std::vector<TansDTable_t>>> _dtables_t_3d;
  std::vector<std::vector<TansDTable_t>> _dtables_t_2d;
  SearchTreeNode<TansDTable_t> _dtables_t_st;
  // std::vector<TansDTable_t> _dtables;
  // std::string _stream;
  // void *_ptr;
};
