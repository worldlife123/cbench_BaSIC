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

#include "tans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

constexpr bool enable_debug_print = false;
#define TANS_DEBUG_PRINT(...) if(enable_debug_print) fprintf(stderr, __VA_ARGS__)
constexpr bool fast = false; // fast mode?

namespace py = pybind11;

/* probability range, this could be a parameter... */
// constexpr int precision = 16;

// constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
// constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

constexpr int default_tableLog = 10;

namespace {

// FSE helper functions
MEM_STATIC void initCState(TansState* statePtr, const U32 tableLog)
{
    statePtr->value = (1<<tableLog);
    statePtr->stateLog = tableLog;
}

MEM_STATIC void initCTable(TansCTable_t* tablePtr, const TansCTable* ct)
{
    const void* ptr = ct;
    const U16* u16ptr = (const U16*) ptr;
    const U32 tableLog = MEM_read16(ptr);
    tablePtr->stateTable = u16ptr+2;
    tablePtr->symbolTT = (const FSE_symbolCompressionTransform*) (ct + 1 + (tableLog ? (1<<(tableLog-1)) : 1));
}

MEM_STATIC void initCState2(TansState* statePtr, const U32 tableLog, const TansCTable_t* cTable, U32 symbol)
{
    initCState(statePtr, tableLog);
    {   const FSE_symbolCompressionTransform symbolTT = ((const FSE_symbolCompressionTransform*)(cTable->symbolTT))[symbol];
        const U16* stateTable = (const U16*)(cTable->stateTable);
        U32 nbBitsOut  = (U32)((symbolTT.deltaNbBits + (1<<15)) >> 16);
        statePtr->value = (nbBitsOut << 16) - symbolTT.deltaNbBits;
        statePtr->value = stateTable[(statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
        TANS_DEBUG_PRINT("Init Cstate is %llu, nbBitsOut=%u, symbolTT.deltaFindState=%i\n", statePtr->value, nbBitsOut, symbolTT.deltaFindState);
    }
}

MEM_STATIC void encodeSymbol(BIT_CStream_t* bitC, TansState* statePtr, const TansCTable_t* cTable, U32 symbol)
{
    FSE_symbolCompressionTransform const symbolTT = ((const FSE_symbolCompressionTransform*)(cTable->symbolTT))[symbol];
    const U16* const stateTable = (const U16*)(cTable->stateTable);
    U32 const nbBitsOut  = (U32)((statePtr->value + symbolTT.deltaNbBits) >> 16);
    BIT_addBits(bitC, statePtr->value, nbBitsOut);
    statePtr->value = stateTable[ (statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
    TANS_DEBUG_PRINT("Added %u bits, current state is %llu, symbolTT.deltaFindState=%i\n", nbBitsOut, statePtr->value, symbolTT.deltaFindState);
}

MEM_STATIC void flushCState(BIT_CStream_t* bitC, const TansState* statePtr)
{
    BIT_addBits(bitC, statePtr->value, statePtr->stateLog);
    BIT_flushBits(bitC);
    TANS_DEBUG_PRINT("Flushed state is %llu, stateLog=%i\n", statePtr->value, statePtr->stateLog);
}

MEM_STATIC void initDState(TansState* DStatePtr, BIT_DStream_t* bitD, const U32 tableLog)
{
    DStatePtr->value = BIT_readBits(bitD, tableLog);
    DStatePtr->stateLog = tableLog;
    TANS_DEBUG_PRINT("Init Dstate is %llu, tableLog=%i\n", DStatePtr->value, tableLog);
}

MEM_STATIC void initDTable(TansDTable_t* DTablePtr, const TansDTable* dt)
{
    // const void* ptr = dt;
    // const FSE_DTableHeader* const DTableH = (const FSE_DTableHeader*)ptr;
    DTablePtr->table = (const FSE_decode_t*) (dt + 1);
}

MEM_STATIC BYTE peekSymbol(TansState* DStatePtr, const TansDTable_t* DTablePtr)
{
    FSE_decode_t const DInfo = ((const FSE_decode_t*)(DTablePtr->table))[DStatePtr->value];
    return DInfo.symbol;
}

MEM_STATIC void updateState(TansState* DStatePtr, const TansDTable_t* DTablePtr, BIT_DStream_t* bitD)
{
    FSE_decode_t const DInfo = ((const FSE_decode_t*)(DTablePtr->table))[DStatePtr->value];
    U32 const nbBits = DInfo.nbBits;
    size_t const lowBits = BIT_readBits(bitD, nbBits);
    DStatePtr->value = DInfo.newState + lowBits;
}

MEM_STATIC BYTE decodeSymbol(TansState* DStatePtr, const TansDTable_t* DTablePtr, BIT_DStream_t* bitD)
{
    FSE_decode_t const DInfo = ((const FSE_decode_t*)(DTablePtr->table))[DStatePtr->value];
    U32 const nbBits = DInfo.nbBits;
    BYTE const symbol = DInfo.symbol;
    size_t const lowBits = BIT_readBits(bitD, nbBits);

    DStatePtr->value = DInfo.newState + lowBits;
    TANS_DEBUG_PRINT("Read %u bits, current state is %llu\n", nbBits, DStatePtr->value);
    return symbol;
}

/*! FSE_decodeSymbolFast() :
    unsafe, only works if no symbol has a probability > 50% */
MEM_STATIC BYTE decodeSymbolFast(TansState* DStatePtr, const TansDTable_t* DTablePtr, BIT_DStream_t* bitD)
{
    FSE_decode_t const DInfo = ((const FSE_decode_t*)(DTablePtr->table))[DStatePtr->value];
    U32 const nbBits = DInfo.nbBits;
    BYTE const symbol = DInfo.symbol;
    size_t const lowBits = BIT_readBitsFast(bitD, nbBits);

    DStatePtr->value = DInfo.newState + lowBits;
    return symbol;
}

MEM_STATIC unsigned endOfDState(const TansState* DStatePtr)
{
    return DStatePtr->value == 0;
}

MEM_STATIC size_t create_single_ctable_using_cnt(FSE_CTable* table, const uint32_t* count, unsigned maxSymbolValue, unsigned tableLog)
{
  size_t total_cnt = 0;
  size_t sym=0;
  for (sym=0; sym <= maxSymbolValue; sym++) total_cnt += count[sym];
  short norm[maxSymbolValue+1];
  FSE_normalizeCount(norm, tableLog, count, total_cnt, maxSymbolValue, 0);

  return FSE_buildCTable(table, norm, maxSymbolValue, tableLog);

}

MEM_STATIC size_t create_single_dtable_using_cnt(FSE_DTable* table, const uint32_t* count, unsigned maxSymbolValue, unsigned tableLog)
{
  size_t total_cnt = 0;
  size_t sym=0;
  for (sym=0; sym <= maxSymbolValue; sym++) total_cnt += count[sym];
  short norm[maxSymbolValue+1];
  FSE_normalizeCount(norm, tableLog, count, total_cnt, maxSymbolValue, 0);

  return FSE_buildDTable(table, norm, maxSymbolValue, tableLog);
}

uint32_t uint32_pow (uint32_t x, uint32_t p) {
  uint32_t i = 1;
  for (uint32_t j = 0; j < p; j++)  i *= x;
  return i;
}

std::vector<ssize_t> create_ar_ptr_offsets(py::array_t<uint32_t> indexes, py::array_t<uint32_t> ar_offsets) {
    // initialize ar offsets as ptr offset
  if (ar_offsets.ndim() != 2 || ar_offsets.shape(1) > indexes.ndim()-1) {
    throw pybind11::value_error("ar_offset should be 2-dimensional with shape (*, <=data_dims)");
  }
  // if (uint32_pow(max_symbol_value, ar_offsets.shape(0)) != _ctables.shape(1)) {
  //   throw pybind11::value_error("ctables size incorrect for a lookup table!");
  // }

  std::vector<ssize_t> ar_ptr_offsets(ar_offsets.shape(0));
  for (ssize_t i = 0; i < ar_offsets.shape(0); ++i) {
    // std::vector<ssize_t> cur_offsets(ar_offsets.shape(1));
    ssize_t ar_offset = (indexes.shape(0) - 1) * indexes.strides(0) / sizeof(int32_t);
    for (ssize_t j = 0; j < indexes.ndim()-1; ++j) {
      const int32_t cur_offset = j < ar_offsets.shape(1) ? ar_offsets.at(i, j) : 0;
      
      if (cur_offset > 0) {
        throw pybind11::value_error("ar_offset should be non-positive!");
      }
      // NOTE: indexes has a batch dim, so j+1
      ar_offset += (indexes.shape(j+1) - 1 + cur_offset) * indexes.strides(j+1) / sizeof(int32_t);
      // printf("indexes.shape(j+1)=%u, ar_offset=%i, indexes.strides(j+1)=%i\n", indexes.shape(j+1), cur_offset, indexes.strides(j+1));
    }
    ar_ptr_offsets[i] = indexes.size() - 1 - ar_offset;
    // printf("indexes.size()=%u, ar_offset=%i\n", indexes.size(), ar_offset);
  }

  return ar_ptr_offsets;

}

} // namespace

void BufferedTansEncoder::encode_with_indexes(
  const std::vector<int32_t> &symbols,
  const std::vector<int32_t> &indexes,
  const std::vector<std::vector<TansCTable>> &ctables,
  const std::vector<int32_t> &offsets) {
  
  // initialize ctables
  _ctables = std::vector<std::vector<TansCTable>>(ctables);
  // for (size_t i = 0; i < ctables.size(); ++i) {
  //   _ctables[i] = std::vector<TansCTable>(ctables[i]);
  // }
  _ctables_t.resize(ctables.size());
  for (size_t i = 0; i < ctables.size(); ++i) {
    initCTable(&_ctables_t[i], _ctables[i].data());
  }

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t table_idx = indexes[i];
    assert(table_idx >= 0);
    assert(table_idx < ctables.size());

    // const auto &cdf = cdfs[table_idx];

    // const int32_t max_value = ctables[table_idx].size() - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    int32_t value = symbols[i] - offsets[table_idx];

    // uint32_t raw_val = 0;
    // if (value < 0) {
    //   raw_val = -2 * value - 1;
    //   value = max_value;
    // } else if (value >= max_value) {
    //   raw_val = 2 * (value - max_value);
    //   value = max_value;
    // }

    assert(value >= 0);
    assert(value < ctables[table_idx].size() - 1);

    _syms.push_back({static_cast<uint16_t>(value),
                     static_cast<uint16_t>(table_idx),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    // if (value == max_value) {
    //   /* Determine the number of bypasses (in bypass_precision size) needed to
    //    * encode the raw value. */
    //   int32_t n_bypass = 0;
    //   while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
    //     ++n_bypass;
    //   }

    //   /* Encode number of bypasses */
    //   int32_t val = n_bypass;
    //   while (val >= max_bypass_val) {
    //     _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
    //     val -= max_bypass_val;
    //   }
    //   _syms.push_back(
    //       {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

    //   /* Encode raw value */
    //   for (int32_t j = 0; j < n_bypass; ++j) {
    //     const int32_t val =
    //         (raw_val >> (j * bypass_precision)) & max_bypass_val;
    //     _syms.push_back(
    //         {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
    //   }
    // }
  }
}

void BufferedTansEncoder::encode_with_indexes_np(const py::array_t<int32_t> &symbols,
                              const py::array_t<int32_t> &indexes,
                              const py::array_t<TansCTable, py::array::c_style | py::array::forcecast> &ctables,
                              const py::array_t<int32_t> &offsets) {
  // TODO: this copys memory! Is there a way to avoid this?
  std::vector<int32_t> symbols_vec(symbols.data(), symbols.data() + symbols.size());
  std::vector<int32_t> indexes_vec(indexes.data(), indexes.data() + indexes.size());

  std::vector<std::vector<TansCTable>> ctables_vec;
  if (ctables.ndim() != 2) {
    throw pybind11::value_error("cdfs should be 2-dimensional with shape (cdfs_sizes.size, cdfs_sizes)");
  }
  for (int32_t idx=0; idx < ctables.shape(0); idx++){
    ctables_vec.emplace_back(ctables.data(idx), ctables.data(idx) + ctables.shape(1));
  }
  std::vector<int32_t> offsets_vec(offsets.data(), offsets.data() + offsets.size());

  encode_with_indexes(symbols_vec, indexes_vec, ctables_vec, offsets_vec);
}

py::bytes BufferedTansEncoder::flush() {
  TansState state;
  BIT_CStream_t bitC;

  std::vector<uint8_t> output(_syms.size() * default_tableLog / 8, 0xFF); // too much space ?
  // uint8_t *ptr = output.data() + output.size();
  // assert(ptr != nullptr);

  { 
    size_t const initError = BIT_initCStream(&bitC, (void*) output.data(), output.size());
    if (ERR_isError(initError)) {
      printf("BIT_initCStream error: %s !!!\n", ERR_getErrorName(initError));
      return ""; /* not enough space available to write a bitstream */ 
    }
  }

  size_t idx = 0;
  while (!_syms.empty()) {
    const TansSymbol sym = _syms.back();

#define FSE_FLUSHBITS(s)  (fast ? BIT_flushBitsFast(s) : BIT_flushBits(s))

    if (idx == 0) {
      initCState2(&state, default_tableLog, &_ctables_t[sym.index], sym.value);
    }
    else {
      encodeSymbol(&bitC, &state, &_ctables_t[sym.index], sym.value);
    }
    // TODO: reduce flush bits
    FSE_FLUSHBITS(&bitC);
    TANS_DEBUG_PRINT("Encoded %u using table %u, bitstring length is %u\n", (unsigned) sym.value, (unsigned) sym.index, (unsigned) (bitC.ptr-bitC.startPtr));

    // if (!sym.bypass) {
    //   Tans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    // } else {
    //   // unlikely...
    //   Tans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    // }
    _syms.pop_back();
    idx++;
  }

  // Tans64EncFlush(&rans, &ptr);
  flushCState(&bitC, &state);

  const int nbytes = BIT_closeCStream(&bitC);
  if (nbytes == 0) {
    printf("BIT_closeCStream error: overflow detected !!!\n");
    return "";
  }

  return std::string(reinterpret_cast<char *>(output.data()), nbytes);
}

py::bytes
TansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols,
                           const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<TansCTable>> &ctables,
                           const std::vector<int32_t> &offsets) {

  BufferedTansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes(symbols, indexes, ctables,
                                        offsets);
  return buffered_rans_enc.flush();
}

py::bytes TansEncoder::encode_with_indexes_np(const py::array_t<int32_t> &symbols,
                              const py::array_t<int32_t> &indexes,
                              const std::optional<py::array_t<TansCTable, py::array::c_style | py::array::forcecast>> &ctables,
                              const std::optional<py::array_t<int32_t>> &offsets){
  TansState state;
  BIT_CStream_t bitC;

  std::vector<uint8_t> output(symbols.size() * default_tableLog / 8, 0xFF); // too much space ?
  // uint8_t *ptr = output.data() + output.size();
  // assert(ptr != nullptr);

  // initialize ctables
  if (ctables.has_value()) {
    set_ctables(ctables.value());
  }

  // initialize offset
  std::vector<int32_t> offset_vec;
  if (offsets.has_value()) {
    offset_vec = std::vector<int32_t>(offsets.value().data(), offsets.value().data() + offsets.value().size());
  }
  // else {
  //   offset_vec = std::vector<int32_t>(*std::max_element(indexes.data(), indexes.data()+indexes.size()), 0);
  // }

  // initialize ctables
  // if (ctables.ndim() != 2 || ctables.shape(0) != offsets.size()) {
  //   throw pybind11::value_error("ctables should be 2-dimensional with shape (offset.size, *)");
  // }
  // std::vector<TansCTable_t> ctables_t(ctables.shape(0));
  // for (ssize_t i = 0; i < ctables.shape(0); ++i) {
  //   initCTable(&ctables_t[i], ctables.data(i));
  // }

  { 
    size_t const initError = BIT_initCStream(&bitC, (void*) output.data(), output.size());
    if (ERR_isError(initError)) {
      printf("BIT_initCStream error: %s !!!\n", ERR_getErrorName(initError));
      return ""; /* not enough space available to write a bitstream */ 
    }
  }

  const int32_t* symbols_ptr = symbols.data();
  const int32_t* indexes_ptr = indexes.data();
  // iterate symbols reversed
  for (ssize_t i=symbols.size()-1; i>=0; i--) {

    const int32_t table_idx = indexes_ptr[i];
    assert(table_idx >= 0);
    assert(table_idx < _ctables_t.size());
    const int32_t offset = offset_vec.size() > 0 ? offset_vec[table_idx] : 0;

    // const auto &cdf = cdfs[table_idx];

    // const int32_t max_value = ctables[table_idx].size() - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    const auto& table = _ctables_t[table_idx];
    int32_t value = symbols_ptr[i] - offset;

    // uint32_t raw_val = 0;
    // if (value < 0) {
    //   raw_val = -2 * value - 1;
    //   value = max_value;
    // } else if (value >= max_value) {
    //   raw_val = 2 * (value - max_value);
    //   value = max_value;
    // }

    assert(value >= 0);
    assert(value <= _maxSymbolValue);

#define FSE_FLUSHBITS(s)  (fast ? BIT_flushBitsFast(s) : BIT_flushBits(s))

    if (i == symbols.size()-1) {
      initCState2(&state, default_tableLog, &table, value);
    }
    else {
      encodeSymbol(&bitC, &state, &table, value);
    }
    // TODO: reduce flush bits
    FSE_FLUSHBITS(&bitC);
    TANS_DEBUG_PRINT("Encoded %u using table %u, bitstring length is %u\n", (unsigned) value, (unsigned) table_idx, (unsigned) (bitC.ptr-bitC.startPtr));

    // if (!sym.bypass) {
    //   Tans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    // } else {
    //   // unlikely...
    //   Tans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    // }
  }

  // Tans64EncFlush(&rans, &ptr);
  flushCState(&bitC, &state);

  const int nbytes = BIT_closeCStream(&bitC);
  if (nbytes == 0) {
    printf("BIT_closeCStream error: overflow detected !!!\n");
    return "";
  }

  // TODO: maybe we should encode array shape?
  return std::string(reinterpret_cast<char *>(output.data()), nbytes);
}

py::bytes TansEncoder::encode_autoregressive_np(const py::array_t<int32_t> &symbols,
                              const py::array_t<int32_t> &indexes,
                              const py::array_t<int32_t> &ar_offsets,
                              const std::optional<py::array_t<TansCTable, py::array::c_style | py::array::forcecast>> &ctables,
                              const std::optional<py::array_t<int32_t>> &offsets,
                              const std::optional<uint32_t> max_symbol_value) {
  TansState state;
  BIT_CStream_t bitC;

  std::vector<uint8_t> output(symbols.size() * default_tableLog / 8, 0xFF); // too much space ?
  // uint8_t *ptr = output.data() + output.size();
  // assert(ptr != nullptr);

  if (max_symbol_value.has_value()) {
    _maxSymbolValue = max_symbol_value.value();
  }

  // initialize ctables
  if (ctables.has_value()) {
    set_ctables(ctables.value());
  }

  // initialize offset
  std::vector<int32_t> offset_vec;
  if (offsets.has_value()) {
    offset_vec = std::vector<int32_t>(offsets.value().data(), offsets.value().data() + offsets.value().size());
  }

  // initialize ctables
  // if (ctables.ndim() != 3 || ctables.shape(0) != offsets.size()) {
  //   throw pybind11::value_error("ctables should be 3-dimensional with shape (offset.size, lookup_table.size, *)");
  // }
  // std::vector<std::vector<TansCTable_t>> ctables_t(ctables.shape(0));
  // for (ssize_t i = 0; i < ctables.shape(0); ++i) {
  //   ctables_t[i].resize(ctables.shape(1));
  //   for (ssize_t j = 0; j < ctables.shape(1); ++j) {
  //     initCTable(&ctables_t[i][j], ctables.data(i, j));
  //   }
  // }

  // initialize ar offsets as ptr offset
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(indexes, ar_offsets);

  // initialize output stream
  { 
    size_t const initError = BIT_initCStream(&bitC, (void*) output.data(), output.size());
    if (ERR_isError(initError)) {
      printf("BIT_initCStream error: %s !!!\n", ERR_getErrorName(initError));
      return ""; /* not enough space available to write a bitstream */ 
    }
  }

  const int32_t* symbols_ptr = symbols.data();
  const int32_t* indexes_ptr = indexes.data();
  // iterate symbols reversed
  for (ssize_t i=symbols.size()-1; i>=0; i--) {

    int32_t table_idx = indexes_ptr[i];
    assert(table_idx >= 0);
    assert(table_idx < _ctables_t.size());
    const int32_t offset = offset_vec.size() > 0 ? offset_vec[table_idx] : 0;

    // add ar_idx
    // for (ssize_t offset : ar_ptr_offsets ) {
    //   table_idx *= (_maxSymbolValue+1);
    //   table_idx += (i >= offset) ? symbols_ptr[i - offset] : 0;
    // }
    // const auto& table = _ctables_t[table_idx];

    // std::vector<size_t> ar_values(ar_ptr_offsets.size());
    // size_t ar_idx=0;
    // for (ssize_t offset : ar_ptr_offsets ) {
    //   ar_values[ar_idx++] = ((i >= offset) ? symbols_ptr[i - offset] : 0);
    // }
    // printf("ar_values.size()=%u\n", ar_values.size());    
    // const auto& table = _ctables_t_st.index_node(table_idx)->index_data(ar_values);

    // TansCTable_t table;
    // auto node = _ctables_t_st.index_node(table_idx);
    // if (ar_ptr_offsets.size() == 1) {
    //   table = node->index_data(ar_values[0]);
    // }
    // else if (ar_ptr_offsets.size() == 2) {
    //   table = node->index_node(ar_values[0])->index_data(ar_values[1]);
    // }
    // else {
    //   throw py::value_error("Too many dimensions!");
    // } 

#define GET_AR_VALUE(off) (i >= off) ? symbols_ptr[i - off] : 0
#define GET_AR_VALUE_DEFAULT(off) (i >= off) ? symbols_ptr[i - off]+1 : 0
    TansCTable_t table;
    if (ar_ptr_offsets.size() == 1) {
      const auto ar_values_0 = (i >= ar_ptr_offsets[0]) ? symbols_ptr[i - ar_ptr_offsets[0]] : 0;
      table = _ctables_t_2d[table_idx][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])];
    }
    else if (ar_ptr_offsets.size() == 2) {
      const auto ar_values_0 = (i >= ar_ptr_offsets[0]) ? symbols_ptr[i - ar_ptr_offsets[0]] : 0;
      const auto ar_values_1 = (i >= ar_ptr_offsets[1]) ? symbols_ptr[i - ar_ptr_offsets[1]] : 0;
      table = _ctables_t_3d[table_idx][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])][GET_AR_VALUE_DEFAULT(ar_ptr_offsets[1])];
    }
    else {
      throw py::value_error("Too many dimensions!");
    } 

    // printf("Choose prior table=%i ar=%i for symbol %i\n", table_idx, ar_idx, i);
    
    int32_t value = symbols_ptr[i] - offset;

    // uint32_t raw_val = 0;
    // if (value < 0) {
    //   raw_val = -2 * value - 1;
    //   value = max_value;
    // } else if (value >= max_value) {
    //   raw_val = 2 * (value - max_value);
    //   value = max_value;
    // }

    assert(value >= 0);
    assert(value <= _maxSymbolValue);

#define FSE_FLUSHBITS(s)  (fast ? BIT_flushBitsFast(s) : BIT_flushBits(s))

    if (i == symbols.size()-1) {
      initCState2(&state, default_tableLog, &table, value);
    }
    else {
      encodeSymbol(&bitC, &state, &table, value);
    }
    // TODO: reduce flush bits
    FSE_FLUSHBITS(&bitC);
    TANS_DEBUG_PRINT("Encoded %u using table %u, bitstring length is %u\n", (unsigned) value, (unsigned) table_idx, (unsigned) (bitC.ptr-bitC.startPtr));

    // if (!sym.bypass) {
    //   Tans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    // } else {
    //   // unlikely...
    //   Tans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    // }
  }

  // Tans64EncFlush(&rans, &ptr);
  flushCState(&bitC, &state);

  const int nbytes = BIT_closeCStream(&bitC);
  if (nbytes == 0) {
    printf("BIT_closeCStream error: overflow detected !!!\n");
    return "";
  }

  // TODO: maybe we should encode array shape?
  return std::string(reinterpret_cast<char *>(output.data()), nbytes);
}

void TansEncoder::set_ctables(const py::array_t<TansCTable, py::array::c_style> &ctables) {
  _ctables = py::array_t<TansCTable, py::array::c_style>(ctables);
  const ssize_t ndim = _ctables.ndim();
  ssize_t batch_size = 1;
  if (ndim <= 0) {
    throw py::value_error("Empty count array!");
  }
  if (ndim == 1) {
    batch_size = 1;
  }
  else {
    batch_size = _ctables.size() / _ctables.shape(ndim-1);
  }
  std::vector<size_t> dims_vec(ndim-1);
  for (ssize_t i=0; i<ndim-1; i++) {
    dims_vec[i] = ctables.shape(i);
  }

  // _tableLog = tableLog; // TODO: infer tableLog?
  _ctables.resize({batch_size, _ctables.shape(ndim-1)});
  _ctables_t.resize(batch_size);

  // iterate through batch
  for (ssize_t i=0; i<batch_size; i++)
  {
    initCTable(&_ctables_t[i], _ctables.data(i));
  }

  // initialize search tree
  if (ndim == 1)
    _ctables_t_st = SearchTreeNode<TansCTable_t>(_ctables_t[0]);
  else {
    _ctables_t_st = SearchTreeNode<TansCTable_t>(_ctables_t, dims_vec);
  }

}

std::vector<int32_t>
TansDecoder::decode_with_indexes(const std::string &encoded,
                      const std::vector<int32_t> &indexes,
                      const std::vector<std::vector<TansDTable>> &dtables,
                      const std::vector<int32_t> &offsets) {
  std::vector<int32_t> output(indexes.size());

  TansState state;
  BIT_DStream_t bitD;
  std::vector<TansDTable_t> tans_dtables(dtables.size());
  /* Init */
  for (size_t i = 0; i < dtables.size(); ++i) {
    initDTable(&tans_dtables[i], dtables[i].data());
  }

  { 
    size_t const initError = BIT_initDStream(&bitD, (void*) encoded.data(), encoded.size());
    if (ERR_isError(initError)) {
      printf("BIT_initDStream error: %s !!!\n", ERR_getErrorName(initError));
      return std::vector<int32_t>();
    }
  }

  initDState(&state, &bitD, default_tableLog);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t table_idx = indexes[i];
    assert(table_idx >= 0);
    assert(table_idx < dtables.size());

    const auto &dtable = tans_dtables[table_idx];

    // const int32_t max_value = cdfs_sizes[table_idx] - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[table_idx];

#define FSE_GETSYMBOL(statePtr) fast ? decodeSymbolFast(statePtr, &dtable, &bitD) : decodeSymbol(statePtr, &dtable, &bitD)

    BIT_reloadDStream(&bitD); // TODO: reduce BIT_reloadDStream call
    const uint8_t symbol = FSE_GETSYMBOL(&state);
    TANS_DEBUG_PRINT("Decoded %u using table %u, bitstring length is %u\n", (unsigned) symbol, (unsigned) table_idx, (unsigned) (bitD.ptr-bitD.start));

    int32_t value = static_cast<int32_t>(symbol);
    

    // if (value == max_value) {
    //   /* Bypass decoding mode */
    //   int32_t val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //   int32_t n_bypass = val;

    //   while (val == max_bypass_val) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     n_bypass += val;
    //   }

    //   int32_t raw_val = 0;
    //   for (int j = 0; j < n_bypass; ++j) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     assert(val <= max_bypass_val);
    //     raw_val |= val << (j * bypass_precision);
    //   }
    //   value = raw_val >> 1;
    //   if (raw_val & 1) {
    //     value = -value - 1;
    //   } else {
    //     value += max_value;
    //   }
    // }

    output[i] = value + offset;
  }

  return output;
}

py::array_t<int32_t>
TansDecoder::decode_with_indexes_np(const std::string &encoded,
                         const py::array_t<int32_t> &indexes,
                         const std::optional<py::array_t<TansDTable, py::array::c_style | py::array::forcecast>> &dtables,
                         const std::optional<py::array_t<int32_t>> &offsets) {

  py::array_t<int32_t> output(indexes.request(true)); // is this a deep copy?
  
  TansState state;
  BIT_DStream_t bitD;

  // initialize dtables
  if (dtables.has_value()) {
    set_dtables(dtables.value());
  }

  // initialize offset
  std::vector<int32_t> offset_vec;
  if (offsets.has_value()) {
    offset_vec = std::vector<int32_t>(offsets.value().data(), offsets.value().data() + offsets.value().size());
  }

  // std::vector<TansDTable_t> tans_dtables(dtables.shape(0));
  // if (dtables.ndim() != 2 || dtables.shape(0) != offsets.size()) {
  //   throw pybind11::value_error("dtables should be 2-dimensional with shape (offset.size, *)");
  // }

  /* Init */
  // for (size_t i = 0; i < tans_dtables.size(); ++i) {
  //   initDTable(&tans_dtables[i], dtables.data(i));
  // }

  { 
    size_t const initError = BIT_initDStream(&bitD, (void*) encoded.data(), encoded.size());
    if (ERR_isError(initError)) {
      printf("BIT_initDStream error: %s !!!\n", ERR_getErrorName(initError));
      return py::array_t<int32_t>();
    }  }

  initDState(&state, &bitD, default_tableLog);

  int32_t* symbols_ptr = output.mutable_data();
  const int32_t* indexes_ptr = indexes.data();
  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t table_idx = indexes_ptr[i];
    assert(table_idx >= 0);
    assert(table_idx < _dtables_t.size());

    const auto &dtable = _dtables_t[table_idx];

    // const int32_t max_value = cdfs_sizes[table_idx] - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    const int32_t offset = offset_vec.size() > 0 ? offset_vec[table_idx] : 0;

#define FSE_GETSYMBOL(statePtr) fast ? decodeSymbolFast(statePtr, &dtable, &bitD) : decodeSymbol(statePtr, &dtable, &bitD)

    BIT_reloadDStream(&bitD); // TODO: reduce BIT_reloadDStream call
    const uint8_t symbol = FSE_GETSYMBOL(&state);
    TANS_DEBUG_PRINT("Decoded %u using table %u, bitstring length is %u\n", (unsigned) symbol, (unsigned) table_idx, (unsigned) (bitD.ptr-bitD.start));

    int32_t value = static_cast<int32_t>(symbol);

    // if (value == max_value) {
    //   /* Bypass decoding mode */
    //   int32_t val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //   int32_t n_bypass = val;

    //   while (val == max_bypass_val) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     n_bypass += val;
    //   }

    //   int32_t raw_val = 0;
    //   for (int j = 0; j < n_bypass; ++j) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     assert(val <= max_bypass_val);
    //     raw_val |= val << (j * bypass_precision);
    //   }
    //   value = raw_val >> 1;
    //   if (raw_val & 1) {
    //     value = -value - 1;
    //   } else {
    //     value += max_value;
    //   }
    // }

    symbols_ptr[i] = value + offset;
  }

  return output;
}

py::array_t<int32_t>
TansDecoder::decode_autoregressive_np(const std::string &encoded,
                         const py::array_t<int32_t> &indexes,
                         const py::array_t<int32_t> &ar_offsets,
                         const std::optional<py::array_t<TansDTable, py::array::c_style | py::array::forcecast>> &dtables,
                         const std::optional<py::array_t<int32_t>> &offsets,
                         const std::optional<uint32_t> max_symbol_value) {

  py::array_t<int32_t> output(indexes.request(true)); // is this a deep copy?
  
  TansState state;
  BIT_DStream_t bitD;

  if (max_symbol_value.has_value()) {
    _maxSymbolValue = max_symbol_value.value();
  }

  // initialize dtables
  if (dtables.has_value()) {
    set_dtables(dtables.value());
  }

  // initialize offset
  std::vector<int32_t> offset_vec;
  if (offsets.has_value()) {
    offset_vec = std::vector<int32_t>(offsets.value().data(), offsets.value().data() + offsets.value().size());
  }

  // std::vector<std::vector<TansDTable_t>> tans_dtables(dtables.shape(0));
  // // initialize dtables
  // if (dtables.ndim() != 3 || dtables.shape(0) != offsets.size()) {
  //   throw pybind11::value_error("dtables should be 3-dimensional with shape (offset.size, lookup_table.size, *)");
  // }
  // for (ssize_t i = 0; i < dtables.shape(0); ++i) {
  //   tans_dtables[i].resize(dtables.shape(1));
  //   for (ssize_t j = 0; j < dtables.shape(1); ++j) {
  //     initDTable(&tans_dtables[i][j], dtables.data(i, j));
  //   }
  // }

  // initialize ar offsets as ptr offset
  std::vector<ssize_t> ar_ptr_offsets = create_ar_ptr_offsets(indexes, ar_offsets);

  { 
    size_t const initError = BIT_initDStream(&bitD, (void*) encoded.data(), encoded.size());
    if (ERR_isError(initError)) {
      printf("BIT_initDStream error: %s !!!\n", ERR_getErrorName(initError));
      return py::array_t<int32_t>();
    }  
  }

  initDState(&state, &bitD, default_tableLog);

  int32_t* symbols_ptr = output.mutable_data();
  const int32_t* indexes_ptr = indexes.data();
  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    int32_t table_idx = indexes_ptr[i];
    assert(table_idx >= 0);
    assert(table_idx < _dtables_t.size());
    const int32_t offset = offset_vec.size() > 0 ? offset_vec[table_idx] : 0;

    // add ar_idx
    // for (ssize_t offset : ar_ptr_offsets ) {
    //   table_idx *= (_maxSymbolValue+1);
    //   table_idx += (i >= offset) ? symbols_ptr[i - offset] : 0;
    // }
    // const auto &table = _dtables_t[table_idx];

    // std::vector<size_t> ar_values;
    // for (ssize_t offset : ar_ptr_offsets ) {
    //   ar_values.push_back((i >= offset) ? symbols_ptr[i - offset] : 0);
    // }

    // const auto& table = _dtables_t_st.index_node(table_idx)->index_data(ar_values);

    TansDTable_t table;
    // if (ar_ptr_offsets.size() == 1) {
    //   const auto ar_values_0 = (i >= ar_ptr_offsets[0]) ? symbols_ptr[i - ar_ptr_offsets[0]] : 0;
    //   table = _dtables_t_2d[table_idx][ar_values_0];
    // }
    // else if (ar_ptr_offsets.size() == 2) {
    //   const auto ar_values_0 = (i >= ar_ptr_offsets[0]) ? symbols_ptr[i - ar_ptr_offsets[0]] : 0;
    //   const auto ar_values_1 = (i >= ar_ptr_offsets[1]) ? symbols_ptr[i - ar_ptr_offsets[1]] : 0;
    //   table = _dtables_t_3d[table_idx][ar_values_0][ar_values_1];
    // }
    // else {
    //   throw py::value_error("Too many dimensions!");
    // } 
#define DEC_GET_AR_VALUE(off) (i >= off) ? symbols_ptr[i - off] : 0
#define DEC_GET_AR_VALUE_DEFAULT(off) (i >= off) ? symbols_ptr[i - off]+1 : 0
    if (ar_ptr_offsets.size() == 1) {
      const auto ar_values_0 = (i >= ar_ptr_offsets[0]) ? symbols_ptr[i - ar_ptr_offsets[0]] : 0;
      table = _dtables_t_2d[table_idx][DEC_GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])];
    }
    else if (ar_ptr_offsets.size() == 2) {
      const auto ar_values_0 = (i >= ar_ptr_offsets[0]) ? symbols_ptr[i - ar_ptr_offsets[0]] : 0;
      const auto ar_values_1 = (i >= ar_ptr_offsets[1]) ? symbols_ptr[i - ar_ptr_offsets[1]] : 0;
      table = _dtables_t_3d[table_idx][DEC_GET_AR_VALUE_DEFAULT(ar_ptr_offsets[0])][DEC_GET_AR_VALUE_DEFAULT(ar_ptr_offsets[1])];
    }
    else {
      throw py::value_error("Too many dimensions!");
    } 


    // const int32_t max_value = cdfs_sizes[table_idx] - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());


#define FSE_GETSYMBOL(statePtr) fast ? decodeSymbolFast(statePtr, &table, &bitD) : decodeSymbol(statePtr, &table, &bitD)

    BIT_reloadDStream(&bitD); // TODO: reduce BIT_reloadDStream call
    const uint8_t symbol = FSE_GETSYMBOL(&state);
    TANS_DEBUG_PRINT("Decoded %u using table %u, bitstring length is %u\n", (unsigned) symbol, (unsigned) table_idx, (unsigned) (bitD.ptr-bitD.start));

    int32_t value = static_cast<int32_t>(symbol);

    // if (value == max_value) {
    //   /* Bypass decoding mode */
    //   int32_t val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //   int32_t n_bypass = val;

    //   while (val == max_bypass_val) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     n_bypass += val;
    //   }

    //   int32_t raw_val = 0;
    //   for (int j = 0; j < n_bypass; ++j) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     assert(val <= max_bypass_val);
    //     raw_val |= val << (j * bypass_precision);
    //   }
    //   value = raw_val >> 1;
    //   if (raw_val & 1) {
    //     value = -value - 1;
    //   } else {
    //     value += max_value;
    //   }
    // }

    symbols_ptr[i] = value + offset;
  }

  return output;
}

void TansDecoder::set_dtables(const py::array_t<TansDTable, py::array::c_style> &dtables) {
  _dtables = py::array_t<TansDTable, py::array::c_style>(dtables);
  const ssize_t ndim = _dtables.ndim();
  ssize_t batch_size = 1;
  if (ndim <= 0) {
    throw py::value_error("Empty count array!");
  }
  if (ndim == 1) {
    batch_size = 1;
  }
  else {
    batch_size = _dtables.size() / _dtables.shape(ndim-1);
  }
  std::vector<size_t> dims_vec(ndim-1);
  for (ssize_t i=0; i<ndim-1; i++) dims_vec[i] = dtables.shape(i);

  // _tableLog = tableLog; // TODO: infer tableLog?
  _dtables.resize({batch_size, _dtables.shape(ndim-1)});
  _dtables_t.resize(batch_size);
  
  // iterate through batch
  for (ssize_t i=0; i<batch_size; i++)
  {
    initDTable(&_dtables_t[i], _dtables.data(i));
  }

  // initialize search tree
  if (ndim == 1)
    _dtables_t_st = SearchTreeNode<TansDTable_t>(_dtables_t[0]);
  else {
    _dtables_t_st = SearchTreeNode<TansDTable_t>(_dtables_t, dims_vec);
  }

}

void TansDecoder::set_stream(const std::string &encoded) {
  { 
    size_t const initError = BIT_initDStream(&_bitD, (void*) encoded.data(), encoded.size());
    if (ERR_isError(initError)) return; /* TODO: deal with error */ 
  }

  initDState(&_state, &_bitD, default_tableLog);

}

std::vector<int32_t>
TansDecoder::decode_stream(const std::vector<int32_t> &indexes,
                const std::vector<std::vector<TansDTable>> &dtables,
                const std::vector<int32_t> &offsets) {

  std::vector<int32_t> output(indexes.size());
  std::vector<TansDTable_t> tans_dtables(dtables.size());
  /* Init */
  for (size_t i = 0; i < dtables.size(); ++i) {
    initDTable(&tans_dtables[i], dtables[i].data());
  }

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t table_idx = indexes[i];
    assert(table_idx >= 0);
    assert(table_idx < dtables.size());

    const auto &dtable = tans_dtables[table_idx];

    // const int32_t max_value = cdfs_sizes[table_idx] - 2;
    // assert(max_value >= 0);
    // assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[table_idx];

#define FSE_GETSYMBOL_STREAM(statePtr) fast ? decodeSymbolFast(statePtr, &dtable, &_bitD) : decodeSymbol(statePtr, &dtable, &_bitD)

    BIT_reloadDStream(&_bitD); // TODO: reduce BIT_reloadDStream call
    const uint8_t symbol = FSE_GETSYMBOL_STREAM(&_state);

    int32_t value = static_cast<int32_t>(symbol);
    

    // if (value == max_value) {
    //   /* Bypass decoding mode */
    //   int32_t val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //   int32_t n_bypass = val;

    //   while (val == max_bypass_val) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     n_bypass += val;
    //   }

    //   int32_t raw_val = 0;
    //   for (int j = 0; j < n_bypass; ++j) {
    //     val = Tans64DecGetBits(&rans, &ptr, bypass_precision);
    //     assert(val <= max_bypass_val);
    //     raw_val |= val << (j * bypass_precision);
    //   }
    //   value = raw_val >> 1;
    //   if (raw_val & 1) {
    //     value = -value - 1;
    //   } else {
    //     value += max_value;
    //   }
    // }

    output[i] = value + offset;
  }

  return output;
}

void TansEncoder::create_ctable_using_cnt(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> count, unsigned maxSymbolValue, unsigned tableLog)
{
  const ssize_t ndim = count.ndim();
  const ssize_t out_table_size = (ssize_t) FSE_CTABLE_SIZE_U32 (tableLog, maxSymbolValue);
  ssize_t batch_size = 1;
  if (ndim <= 0) {
    throw py::value_error("Empty count array!");
  }
  if (ndim == 1) {
    batch_size = 1;
  }
  else {
    batch_size = count.size() / count.shape(ndim-1);
  }

  _maxSymbolValue = maxSymbolValue;
  _tableLog = tableLog;
  _ctables.resize({batch_size, out_table_size});
  _ctables_t.resize(batch_size);
  // resize as batched
  py::array_t<uint32_t> batched_count(count);
  batched_count = batched_count.reshape({batch_size, count.shape(ndim-1)});

  // iterate through batch
  for (ssize_t i=0; i<batch_size; i++)
  {
    size_t cs = create_single_ctable_using_cnt(_ctables.mutable_data(i), batched_count.data(i), maxSymbolValue, tableLog);
    if (ERR_isError(cs)) {
        printf("FSE create ctable error: %s !!!\n", ERR_getErrorName(cs));
    }
    initCTable(&_ctables_t[i], _ctables.data(i));
  }

  // initialize search tree
  if (ndim == 1)
    _ctables_t_st = SearchTreeNode<TansCTable_t>(_ctables_t[0]);
  else {
    std::vector<size_t> dims_vec(ndim-1);
    for (ssize_t i=0; i<ndim-1; i++) dims_vec[i] = count.shape(i);
    _ctables_t_st = SearchTreeNode<TansCTable_t>(_ctables_t, dims_vec);
  }

  // initialize corresponding lookup tables
  switch (ndim)
  {
  case 2:
    break;
  case 3:
    printf("Initializing 2d ctables\n");
    _ctables_t_2d.resize(count.shape(0));
    for (ssize_t i = 0; i < count.shape(0); ++i) {
      _ctables_t_2d[i].resize(count.shape(1));
      for (ssize_t j = 0; j < count.shape(1); ++j) {
        ssize_t ctable_idx = i * count.shape(1) + j;
        initCTable(&_ctables_t_2d[i][j], _ctables.data(ctable_idx));
      }
    }
    break;
  case 4:
    printf("Initializing 3d ctables\n");
    _ctables_t_3d.resize(count.shape(0));
    for (ssize_t i = 0; i < count.shape(0); ++i) {
      _ctables_t_3d[i].resize(count.shape(1));
      for (ssize_t j = 0; j < count.shape(1); ++j) {
        _ctables_t_3d[i][j].resize(count.shape(2));
        for (ssize_t k = 0; k < count.shape(2); ++k) {
          ssize_t ctable_idx = i * count.shape(1) * count.shape(2) + j * count.shape(2) + k;
          initCTable(&_ctables_t_3d[i][j][k], _ctables.data(ctable_idx));
        }
      }
    }
    break;
  default:
    throw py::value_error("Too many dimensions!");
  }
}

void TansDecoder::create_dtable_using_cnt(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> count, unsigned maxSymbolValue, unsigned tableLog)
{
  const ssize_t ndim = count.ndim();
  const ssize_t out_table_size = (ssize_t) FSE_DTABLE_SIZE_U32 (tableLog);
  ssize_t batch_size = 1;
  if (ndim <= 0) {
    throw py::value_error("Empty count array!");
  }
  if (ndim == 1) {
    batch_size = 1;
  }
  else {
    batch_size = count.size() / count.shape(ndim-1);
  }

  // Let python manage the memory of dtable
  _maxSymbolValue = maxSymbolValue;
  _tableLog = tableLog;
  _dtables.resize({batch_size, out_table_size});
  _dtables_t.resize(batch_size);
  // resize as batched
  py::array_t<uint32_t> batched_count(count);
  batched_count = batched_count.reshape({batch_size, count.shape(ndim-1)});

  // iterate through batch
  for (ssize_t i=0; i<batch_size; i++)
  {
    size_t cs = create_single_dtable_using_cnt(_dtables.mutable_data(i), batched_count.data(i), maxSymbolValue, tableLog);
    if (ERR_isError(cs)) {
        printf("FSE create dtable error: %s !!!\n", ERR_getErrorName(cs));
    }
    initDTable(&_dtables_t[i], _dtables.data(i));
  }

  // initialize search tree
  if (ndim == 1)
    _dtables_t_st = SearchTreeNode<TansDTable_t>(_dtables_t[0]);
  else {
    std::vector<size_t> dims_vec(ndim-1);
    for (ssize_t i=0; i<ndim-1; i++) dims_vec[i] = count.shape(i);
    _dtables_t_st = SearchTreeNode<TansDTable_t>(_dtables_t, dims_vec);
  }

  // initialize corresponding lookup tables
  switch (ndim)
  {
  case 2:
    break;
  case 3:
    printf("Initializing 2d dtables\n");
    _dtables_t_2d.resize(count.shape(0));
    for (ssize_t i = 0; i < count.shape(0); ++i) {
      _dtables_t_2d[i].resize(count.shape(1));
      for (ssize_t j = 0; j < count.shape(1); ++j) {
        ssize_t dtable_idx = i * count.shape(1) + j;
        initDTable(&_dtables_t_2d[i][j], _dtables.data(dtable_idx));
      }
    }
    break;
  case 4:
    printf("Initializing 3d dtables\n");
    _dtables_t_3d.resize(count.shape(0));
    for (ssize_t i = 0; i < count.shape(0); ++i) {
      _dtables_t_3d[i].resize(count.shape(1));
      for (ssize_t j = 0; j < count.shape(1); ++j) {
        _dtables_t_3d[i][j].resize(count.shape(2));
        for (ssize_t k = 0; k < count.shape(2); ++k) {
          ssize_t dtable_idx = i * count.shape(1) * count.shape(2) + j * count.shape(2) + k;
          initDTable(&_dtables_t_3d[i][j][k], _dtables.data(dtable_idx));
        }
      }
    }
    break;
  default:
    throw py::value_error("Too many dimensions!");
  }

}


py::array_t<TansCTable, py::array::c_style> create_ctable_using_cnt(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> count, unsigned maxSymbolValue, unsigned tableLog)
{
  const ssize_t ndim = count.ndim();
  const ssize_t out_table_size = (ssize_t) FSE_CTABLE_SIZE_U32 (tableLog, maxSymbolValue);
  ssize_t batch_size = 1;
  if (ndim <= 0) {
    throw py::value_error("Empty count array!");
  }
  if (ndim == 1) {
    batch_size = 1;
  }
  else {
    batch_size = count.size() / count.shape(ndim-1);
  }

  // Let python manage the memory of ctable
  py::array_t<FSE_CTable, py::array::c_style> out;
  out.resize({batch_size, out_table_size});
  // resize as batched
  py::array_t<uint32_t> batched_count(count);
  batched_count = batched_count.reshape({batch_size, count.shape(ndim-1)});

  // iterate through batch
  for (ssize_t i=0; i<batch_size; i++)
  {
    size_t cs = create_single_ctable_using_cnt(out.mutable_data(i), batched_count.data(i), maxSymbolValue, tableLog);
    if (ERR_isError(cs)) {
        printf("FSE create ctable error: %s !!!\n", ERR_getErrorName(cs));
    }
  }

  // reshape as original (TODO: a var length code to reshape?)
  switch (ndim)
  {
  case 1:
    out = out.reshape({out_table_size});
    break;
  case 2:
    out = out.reshape({count.shape(0), out_table_size});
    break;
  case 3:
    out = out.reshape({count.shape(0), count.shape(1), out_table_size});
    break;
  case 4:
    out = out.reshape({count.shape(0), count.shape(1), count.shape(2), out_table_size});
    break;
  default:
    throw py::value_error("Too many dimensions!");
  }

  return out;

}

py::array_t<TansDTable, py::array::c_style> create_dtable_using_cnt(const py::array_t<uint32_t, py::array::c_style | py::array::forcecast> count, unsigned maxSymbolValue, unsigned tableLog)
{
  const ssize_t ndim = count.ndim();
  const ssize_t out_table_size = (ssize_t) FSE_DTABLE_SIZE_U32 (tableLog);
  ssize_t batch_size = 1;
  if (ndim <= 0) {
    throw py::value_error("Empty count array!");
  }
  if (ndim == 1) {
    batch_size = 1;
  }
  else {
    batch_size = count.size() / count.shape(ndim-1);
  }

  // Let python manage the memory of dtable
  py::array_t<FSE_DTable, py::array::c_style> out;
  out.resize({batch_size, out_table_size});
  // resize as batched
  py::array_t<uint32_t> batched_count(count);
  batched_count = batched_count.reshape({batch_size, count.shape(ndim-1)});

  // iterate through batch
  for (ssize_t i=0; i<batch_size; i++)
  {
    size_t cs = create_single_dtable_using_cnt(out.mutable_data(i), batched_count.data(i), maxSymbolValue, tableLog);
    if (ERR_isError(cs)) {
        printf("FSE create dtable error: %s !!!\n", ERR_getErrorName(cs));
    }
  }

  // reshape as original (TODO: a var length code to reshape?)
  switch (ndim)
  {
  case 1:
    out = out.reshape({out_table_size});
    break;
  case 2:
    out = out.reshape({count.shape(0), out_table_size});
    break;
  case 3:
    out = out.reshape({count.shape(0), count.shape(1), out_table_size});
    break;
  case 4:
    out = out.reshape({count.shape(0), count.shape(1), count.shape(2), out_table_size});
    break;
  default:
    throw py::value_error("Too many dimensions!");
  }

  return out;

}

PYBIND11_MODULE(tans, m) {
  m.doc() = "table Asymmetric Numeral System python bindings";

  py::class_<BufferedTansEncoder>(m, "BufferedTansEncoder", py::module_local())
      .def(py::init<>())
      .def("encode_with_indexes", &BufferedTansEncoder::encode_with_indexes)
      .def("encode_with_indexes_np", &BufferedTansEncoder::encode_with_indexes_np)
      .def("flush", &BufferedTansEncoder::flush)
      ;

  py::class_<TansEncoder>(m, "TansEncoder", py::module_local())
      .def(py::init<>())
      .def("encode_with_indexes", &TansEncoder::encode_with_indexes)
      .def("encode_with_indexes_np", &TansEncoder::encode_with_indexes_np,
        py::arg("symbols"), py::arg("indexes"), py::arg("ctables")=py::none(), py::arg("offsets")=py::none())
      .def("encode_autoregressive_np", &TansEncoder::encode_autoregressive_np,
        py::arg("symbols"), py::arg("indexes"), py::arg("ar_offsets"), py::arg("ctables")=py::none(), py::arg("offsets")=py::none(), py::arg("max_symbol_value")=py::none())
      .def("create_ctable_using_cnt", &TansEncoder::create_ctable_using_cnt, "", py::arg("count"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=default_tableLog)
      ;

  py::class_<TansDecoder>(m, "TansDecoder", py::module_local())
      .def(py::init<>())
      .def("set_stream", &TansDecoder::set_stream)
      .def("decode_stream", &TansDecoder::decode_stream)
      .def("decode_with_indexes", &TansDecoder::decode_with_indexes,
           "Decode a string to a list of symbols")
      .def("decode_with_indexes_np", &TansDecoder::decode_with_indexes_np,
           "Decode a string to a list of symbols",
        py::arg("encoded"), py::arg("indexes"), py::arg("dtables")=py::none(), py::arg("offsets")=py::none())
      .def("decode_autoregressive_np", &TansDecoder::decode_autoregressive_np,
        py::arg("encoded"), py::arg("indexes"), py::arg("ar_offsets"), py::arg("dtables")=py::none(), py::arg("offsets")=py::none(), py::arg("max_symbol_value")=py::none())
      .def("create_dtable_using_cnt", &TansDecoder::create_dtable_using_cnt, "", py::arg("count"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=default_tableLog)
      ;
  
  // utils
  m.def("create_ctable_using_cnt", &create_ctable_using_cnt, "", py::arg("count"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=default_tableLog);
  m.def("create_dtable_using_cnt", &create_dtable_using_cnt, "", py::arg("count"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=default_tableLog);

}
