#include "zstd_wrapper.h"

/***************
* ZSTD bindings
****************/

py::bytes zstd_compress(const std::string& src_string, const std::string& dict_string, int level=0)
{
    std::vector<char> out;
    // std::string src_string = src();
    size_t out_capacity = ZSTD_compressBound(src_string.size());
    out.reserve(out_capacity);
    // size_t cs = ZSTD_compress(&out[0], out_capacity, src_string.data(), src_string.size(), level);
    
    /* Create the context. */
    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
    CHECK(cctx != NULL, "ZSTD_createCCtx() failed!");
    CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level) );
    
    size_t cs = 0;
    if (!dict_string.empty())
    {
        ZSTD_CCtx_loadDictionary(cctx, (void*) dict_string.data(), dict_string.size());
        // cs = ZSTD_compress_usingDict(cctx, &out[0], out_capacity, src_string.data(), src_string.size(), (void*) dict_string.data(), dict_string.size(), level);
    }
    // else
    // {
    //     cs = ZSTD_compressCCtx(cctx, &out[0], out_capacity, src_string.data(), src_string.size(), level);
    // }

    cs = ZSTD_compress2(cctx, &out[0], out_capacity, src_string.data(), src_string.size());
    if (ZSTD_isError(cs)) {
        printf("ZSTD enc error: %s !!!\n", ZSTD_getErrorName(cs));
        return src_string;
    }
    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    
    ZSTD_freeCCtx(cctx);
    return py::bytes(&out[0], cs);
}

py::bytes zstd_decompress(const std::string& src_string, size_t max_decode_size, const std::string& dict_string)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    // size_t out_capacity = ZSTD_compressBound(src_string.size());
    out.reserve(max_decode_size);
    // size_t cs = ZSTD_decompress(&out[0], max_decode_size, src_string.data(), src_string.size());
    
    ZSTD_DCtx* const dctx = ZSTD_createDCtx();
    CHECK(dctx != NULL, "ZSTD_createDCtx() failed!");
    
    size_t cs = 0;
    if (!dict_string.empty())
    {
        ZSTD_DCtx_loadDictionary(dctx, (void*) dict_string.data(), dict_string.size());
        // cs = ZSTD_decompress_usingDict(dctx, &out[0], max_decode_size, src_string.data(), src_string.size(), dict_string.data(), dict_string.size());
    }
    // else
    // {
    //     cs = ZSTD_decompressDCtx(dctx, &out[0], max_decode_size, src_string.data(), src_string.size());
    // }

    cs = ZSTD_decompressDCtx(dctx, &out[0], max_decode_size, src_string.data(), src_string.size());
    if (ZSTD_isError(cs)) {
        printf("ZSTD dec error: %s !!!\n", ZSTD_getErrorName(cs));
        return py::bytes("");;
    }

    ZSTD_freeDCtx(dctx);

    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs);
}

py::bytes fse_compress(const std::string& src_string, unsigned maxSymbolValue=255, unsigned tableLog=12)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    size_t out_capacity = FSE_compressBound(src_string.size());
    out.reserve(out_capacity);
    size_t cs = FSE_compress2(&out[0], out_capacity, src_string.data(), src_string.size(), maxSymbolValue,
                            /*FSE_MAX_TABLELOG=*/tableLog);
    if (FSE_isError(cs)) {
        printf("FSE enc error: %s !!!\n", FSE_getErrorName(cs));
        return src_string;
    }
    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs);
}

py::bytes fse_decompress(const std::string& src_string, size_t max_decode_size)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    // size_t out_capacity = FSE_compressBound(src_string.size());
    out.reserve(max_decode_size);
    size_t cs = FSE_decompress(&out[0], max_decode_size, src_string.data(), src_string.size());
    if (FSE_isError(cs)) {
        printf("FSE dec error: %s !!!\n", FSE_getErrorName(cs));
        return py::bytes("");;
    }
    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs);
}

py::bytes fse_tans_compress(const std::string& base_codes, const py::array_t<TANS_EXTRA_CODE>& extra_codes, const py::array_t<unsigned int>& extra_num_symbols, 
                            unsigned maxSymbolValue, unsigned tableLog, std::vector<std::vector<unsigned>> predefined_counts)
{
    std::vector<char> out;
    // TODO: FSE_tans_compressBound
    size_t out_capacity = FSE_compressBound(base_codes.size() + extra_codes.size() * sizeof(TANS_EXTRA_CODE));
    out.reserve(out_capacity+1);

    void* op = static_cast<void*>(&out[1]);    
    size_t cs = FSE_tans_compress2(op, out_capacity, 
                                base_codes.data(), extra_codes.data(), base_codes.size(), 
                                extra_num_symbols.data(), extra_num_symbols.size(), 
                                maxSymbolValue, tableLog);
    // first byte indicate compress mode
    // Uncompressable / RLE / Error
    // TODO: special case for RLE
    if (cs == 0 || cs == 1 || FSE_isError(cs)) {
        printf("Uncompressable! Switching to direct store mode!\n");
        out[0] = (char) 0;
        cs = FSE_tans_compress_ds(op, out_capacity, 
                            base_codes.data(), extra_codes.data(), base_codes.size(), 
                            extra_num_symbols.data(), extra_num_symbols.size(), 
                            maxSymbolValue, tableLog);
    }
    else if (predefined_counts.size() > 0) {
        assert(predefined_counts.size() < 254);
        // TODO: a faster estimation for best predefined_counts
        size_t cnt_idx = 1;
        size_t cnt_best_idx = 255; // start from undefined
        size_t cnt_best_cs = cs;
        size_t cnt_cs = 0;
        std::vector<char> out_tmp;
        out_tmp.reserve(out_capacity);
        for (std::vector<unsigned> count : predefined_counts) {
            if (count.size() != maxSymbolValue+1) {
                printf("Predefined count %zu does not fit maxSymbolValue! Skip this one...\n", cnt_idx);
                continue;
            }
            cnt_cs = FSE_tans_compress_usingCnt(&out_tmp[0], out_capacity, 
                                base_codes.data(), extra_codes.data(), base_codes.size(), 
                                extra_num_symbols.data(), extra_num_symbols.size(), 
                                maxSymbolValue, tableLog, count.data());
            // copy better compression to out buffer
            if (cnt_cs > 1 && !FSE_isError(cnt_cs) && cnt_cs < cnt_best_cs) {
                cnt_best_idx = cnt_idx;
                cnt_best_cs = cnt_cs;
                std::copy(out_tmp.begin(), out_tmp.begin() + cnt_cs, out.begin()+1);
                // out.insert(out.begin()+1, out_tmp.begin(), out_tmp.begin() + cnt_cs);
            }
            cnt_idx++;
            if (cnt_idx >= 255) break;
        }
        cs = cnt_best_cs;
        out[0] = (char) cnt_best_idx;
        // if (cnt_best_idx < 255) {
        //     printf("Using predefined count %lu!\n", cnt_best_idx-1);
        // }
    }
    else {
        out[0] = (char) 255;
    }

    if (FSE_isError(cs)) {
        printf("FSE enc error: %s !!!\n", FSE_getErrorName(cs));
        return py::bytes("");
    }

    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs+1);
}

std::tuple<py::bytes, py::array_t<TANS_EXTRA_CODE>> fse_tans_decompress(const std::string& src_string, const py::array_t<unsigned int>& extra_num_symbols, 
    size_t max_decode_size, unsigned maxSymbolValue, unsigned tableLog,
    std::vector<std::vector<unsigned>> predefined_counts)
{
    std::vector<char> out_base;
    std::vector<TANS_EXTRA_CODE> out_extra;
    out_base.reserve(max_decode_size);
    out_extra.reserve(max_decode_size);

    char* ip = const_cast<char*>(src_string.data());
    const char* iend = src_string.data() + src_string.size();
    unsigned mode = (unsigned) *ip++;

    size_t cs = 0;
    if (mode == 0) {
        cs = FSE_tans_decompress_ds(&out_base[0], &out_extra[0], max_decode_size, 
                                    extra_num_symbols.data(), extra_num_symbols.size(), 
                                    maxSymbolValue, tableLog,
                                    ip, iend-ip);
    }
    else if (mode < 255) {
        cs = FSE_tans_decompress_usingCnt(&out_base[0], &out_extra[0], max_decode_size, 
                                    extra_num_symbols.data(), extra_num_symbols.size(), 
                                    maxSymbolValue, tableLog, predefined_counts[mode-1].data(),
                                    ip, iend-ip);
    }
    else {
        cs = FSE_tans_decompress(&out_base[0], &out_extra[0], max_decode_size, 
                                    extra_num_symbols.data(), extra_num_symbols.size(), 
                                    ip, iend-ip);
    }

    if (FSE_isError(cs)) {
        printf("FSE dec error: %s !!!\n", FSE_getErrorName(cs));
        return std::tuple<py::bytes, py::array_t<TANS_EXTRA_CODE>>(py::bytes(""), py::array_t<TANS_EXTRA_CODE>());
    }
    return std::tuple<py::bytes, py::array_t<TANS_EXTRA_CODE>>(py::bytes(&out_base[0], cs), py::array_t<TANS_EXTRA_CODE>(cs, &out_extra[0]));
}

py::bytes fse_tans_compress_advanced(const std::string& base_codes, const py::array_t<TANS_EXTRA_CODE>& extra_codes, const py::array_t<unsigned int>& extra_num_symbols, 
                            unsigned maxSymbolValue, unsigned tableLog, 
                            const std::optional<py::array_t<unsigned>> predefined_count)
{
    std::vector<char> out;
    // TODO: FSE_tans_compressBound
    size_t out_capacity = FSE_compressBound(base_codes.size() + extra_codes.size() * sizeof(TANS_EXTRA_CODE));
    out.reserve(out_capacity);

    size_t cs = 0;
    if (!predefined_count.has_value()) {
        cs = FSE_tans_compress2(&out[0], out_capacity, 
                                base_codes.data(), extra_codes.data(), base_codes.size(), 
                                extra_num_symbols.data(), extra_num_symbols.size(), 
                                maxSymbolValue, tableLog);
    }
    else {
        assert(predefined_count.value().size() == maxSymbolValue+1);
        cs = FSE_tans_compress_usingCnt(&out[0], out_capacity, 
                                base_codes.data(), extra_codes.data(), base_codes.size(), 
                                extra_num_symbols.data(), extra_num_symbols.size(), 
                                maxSymbolValue, tableLog, predefined_count.value().data());
    }

    if (FSE_isError(cs)) {
        printf("FSE enc error: %s !!!\n", FSE_getErrorName(cs));
        return py::bytes("");
    }

    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs);
}

std::tuple<py::bytes, py::array_t<TANS_EXTRA_CODE>> fse_tans_decompress_advanced(const std::string& src_string, const py::array_t<unsigned int>& extra_num_symbols, 
    size_t max_decode_size, unsigned maxSymbolValue, unsigned tableLog,
    const std::optional<py::array_t<unsigned>> predefined_count)
{
    std::vector<char> out_base;
    std::vector<TANS_EXTRA_CODE> out_extra;
    out_base.reserve(max_decode_size);
    out_extra.reserve(max_decode_size);

    size_t cs = 0;
    if (!predefined_count.has_value()) {
        cs = FSE_tans_decompress(&out_base[0], &out_extra[0], max_decode_size, 
                                    extra_num_symbols.data(), extra_num_symbols.size(), 
                                    src_string.data(), src_string.size());
    }
    else{
        assert(predefined_count.value().size() == maxSymbolValue+1);
        cs = FSE_tans_decompress_usingCnt(&out_base[0], &out_extra[0], max_decode_size, 
                                    extra_num_symbols.data(), extra_num_symbols.size(), 
                                    maxSymbolValue, tableLog, predefined_count.value().data(),
                                    src_string.data(), src_string.size());
    }
    if (FSE_isError(cs)) {
        printf("FSE dec error: %s !!!\n", FSE_getErrorName(cs));
        return std::tuple<py::bytes, py::array_t<TANS_EXTRA_CODE>>(py::bytes(""), py::array_t<TANS_EXTRA_CODE>());
    }
    return std::tuple<py::bytes, py::array_t<TANS_EXTRA_CODE>>(py::bytes(&out_base[0], cs), py::array_t<TANS_EXTRA_CODE>(cs, &out_extra[0]));
}


py::bytes huf_compress(const std::string& src_string, unsigned maxSymbolValue=255, unsigned tableLog=12)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    size_t out_capacity = HUF_compressBound(src_string.size());
    // write first 2 bytes as length of data
    out.reserve(out_capacity+sizeof(uint16_t));

    char* op = &out[0];
    assert(src_string.size() < std::numeric_limits<uint16_t>::max());
    *(reinterpret_cast<uint16_t*>(op)) = static_cast<uint16_t>(src_string.size());
    op += sizeof(uint16_t);

    size_t cs = 0;
    // zstd defined threshold for huff0 singlestream control
    // if (src_string.size() < 256) {
    //     std::cout << "single stream" << std::endl;
    //     cs = HUF_compress1X(&out[0], out_capacity, src_string.data(), src_string.size(), maxSymbolValue,
    //                         /*HUF_MAX_TABLELOG=*/tableLog);
    // }
    // else {
    //     std::cout << "4 stream" << std::endl;
        cs = HUF_compress2(op, out_capacity, src_string.data(), src_string.size(), maxSymbolValue,
                            /*HUF_MAX_TABLELOG=*/tableLog);
    // }

    if (HUF_isError(cs)) {
        printf("HUF enc error: %s !!!\n", HUF_getErrorName(cs));
        return src_string;
    }
    else if (cs == 0) {
        return py::bytes("");
    }
    else {
        return py::bytes(&out[0], cs+sizeof(uint16_t));
    }
}

py::bytes huf_decompress(const std::string& src_string, size_t max_decode_size)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    // size_t out_capacity = HUF_compressBound(src_string.size());
    out.reserve(max_decode_size);

    // first 2 bytes is src length
    const char* ip = src_string.data();
    size_t src_size = static_cast<size_t>(*(reinterpret_cast<const uint16_t*>(ip)));
    assert(src_size < std::numeric_limits<uint16_t>::max());
    ip += sizeof(uint16_t);

    size_t cs = HUF_decompress(&out[0], src_size, ip, src_string.size()-sizeof(uint16_t));
    if (HUF_isError(cs)) {
        printf("HUF dec error: %s !!!\n", HUF_getErrorName(cs));
        return py::bytes("");;
    }
    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs);
}

/***************
* FSE bindings
****************/
py::array_t<FSE_CTable> fse_create_ctable_using_cnt(const py::array_t<uint32_t> count, unsigned maxSymbolValue, unsigned tableLog)
{
    // TODO: check count 0? It seems 0 causes error!
    size_t total_cnt = 0;
    size_t sym=0;
    for (sym=0; sym <= maxSymbolValue; sym++) total_cnt += count.data()[sym];
    short norm[maxSymbolValue+1];
    FSE_normalizeCount(norm, tableLog, count.data(), total_cnt, maxSymbolValue, 0);

    // Let python manage the memory of ctable
    py::array_t<FSE_CTable> out;
    out.resize({FSE_CTABLE_SIZE_U32 (tableLog, maxSymbolValue)});
    size_t cs = FSE_buildCTable(out.mutable_data(), norm, maxSymbolValue, tableLog);
    if (FSE_isError(cs)) {
        printf("FSE create ctable error: %s !!!\n", FSE_getErrorName(cs));
    }

    return out;

}

py::array_t<FSE_DTable> fse_create_dtable_using_cnt(const py::array_t<uint32_t> count, unsigned maxSymbolValue, unsigned tableLog)
{
    // TODO: check count 0? It seems 0 causes error!
    size_t total_cnt = 0;
    size_t sym=0;
    for (sym=0; sym <= maxSymbolValue; sym++) total_cnt += count.data()[sym];
    short norm[maxSymbolValue+1];
    FSE_normalizeCount(norm, tableLog, count.data(), total_cnt, maxSymbolValue, 0);

    // Let python manage the memory of dtable
    py::array_t<FSE_DTable> out;
    out.resize({FSE_DTABLE_SIZE_U32 (tableLog)});
    size_t ds = FSE_buildDTable(out.mutable_data(), norm, maxSymbolValue, tableLog);
    if (FSE_isError(ds)) {
        printf("FSE create dtable error: %s !!!\n", FSE_getErrorName(ds));
    }

    return out;

}

py::bytes fse_compress_using_ctable(const std::string& src_string, const py::array_t<FSE_CTable> CTable)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    size_t out_capacity = FSE_compressBound(src_string.size());
    out.reserve(out_capacity);

    size_t cs = FSE_compress_usingCTable(&out[0], out_capacity, src_string.data(), src_string.size(), CTable.data());

    if (FSE_isError(cs)) {
        printf("FSE enc error: %s !!!\n", FSE_getErrorName(cs));
        return src_string;
    }
    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], cs);
}

py::bytes fse_decompress_using_dtable(const std::string& src_string, const py::array_t<FSE_DTable> DTable, size_t max_decode_size)
{
    // std::string out; // could only do this in c++17
    std::vector<char> out;
    // std::string src_string = src();
    // size_t out_capacity = FSE_compressBound(src_string.size());
    out.reserve(max_decode_size);
    
    size_t ds = FSE_decompress_usingDTable(&out[0], max_decode_size, src_string.data(), src_string.size(), DTable.data());
    if (FSE_isError(ds)) {
        printf("FSE dec error: %s !!!\n", FSE_getErrorName(ds));
        return py::bytes("");;
    }

    // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
    return py::bytes(&out[0], ds);
}


// py::bytes fse_compress_using_cnt(const std::string& src_string, const py::array_t<uint32_t> count, unsigned maxSymbolValue, unsigned tableLog)
// {
//     // std::string out; // could only do this in c++17
//     std::vector<char> out;
//     // std::string src_string = src();
//     size_t out_capacity = FSE_compressBound(src_string.size());
//     out.reserve(out_capacity);

//     short norm[maxSymbolValue];
//     CHECK_F( FSE_normalizeCount(norm, tableLog, count.data(), count.sum(), maxSymbolValue, 0) );
//     // for (size_t i=0; i< FSE_MAX_SYMBOL_VALUE+1; i++) norm[i]=-1; // set all symbols to lowprob

//     FSE_CTable* CTable = FSE_createCTable(maxSymbolValue, tableLog);
//     CHECK_F( FSE_buildCTable(CTable, norm, maxSymbolValue, tableLog) );
//     size_t cs = FSE_compress_usingCTable(&out[0], out_capacity, src_string.data(), src_string.size(), CTable);
//     FSE_freeCTable(CTable);

//     if (FSE_isError(cs)) {
//         printf("FSE enc error: %s !!!\n", FSE_getErrorName(cs));
//         return src_string;
//     }
//     // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
//     return py::bytes(&out[0], cs);
// }

// py::bytes fse_decompress_using_cnt(const std::string& src_string, py::array_t<uint32_t> count, unsigned maxSymbolValue, unsigned tableLog, size_t max_decode_size)
// {
//     // std::string out; // could only do this in c++17
//     std::vector<char> out;
//     // std::string src_string = src();
//     // size_t out_capacity = FSE_compressBound(src_string.size());
//     out.reserve(max_decode_size);
    
//     short norm[maxSymbolValue];
//     CHECK_F( FSE_normalizeCount(norm, tableLog, count.data(), count.sum(), maxSymbolValue, 0) );
//     // for (size_t i=0; i< FSE_MAX_SYMBOL_VALUE+1; i++) norm[i]=-1; // set all symbols to lowprob

//     FSE_DTable* DTable = FSE_createDTable(tableLog);
//     size_t cs = FSE_decompress_usingDTable(&out[0], max_decode_size, src_string.data(), src_string.size(), DTable);
//     if (FSE_isError(cs)) {
//         printf("FSE dec error: %s !!!\n", FSE_getErrorName(cs));
//         return py::bytes("");;
//     }
//     FSE_freeDTable(DTable);

//     // out.resize(cs); // resize vector to match compressed size, not needed as we'll create new bytes later
//     return py::bytes(&out[0], cs);
// }



/***************
* LZ77 functions
****************/
ZSTD_LZ77_Data zstd_lz77_forward(const std::string& src_string, const std::string& dict_string, ZSTD_cParameter_Configs configs)
{
    std::vector<ZSTD_Sequence> seq;
    size_t out_capacity = ZSTD_compressBound(src_string.size());
    assert(out_capacity < 0xFFFF); // TODO: cannot copy longer than 0xFFFF literals with this function!
    seq.reserve(out_capacity);

    /* Create the context. */
    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
    CHECK(cctx != NULL, "ZSTD_createCCtx() failed!");

    /* Set any parameters you want.
     * Here we set the compression level, and enable the checksum.
     */
    if (!configs.empty())
    {
        for (auto const& c : configs)
        {
            CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, c.first, c.second) );
        }
    }
    CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_blockDelimiters, ZSTD_sf_noBlockDelimiters) );

    if (!dict_string.empty())
    {
        ZSTD_CCtx_loadDictionary(cctx, (void*) dict_string.data(), dict_string.size());
    }

    // TODO: remove last seq for better compression
    size_t seq_size = ZSTD_generateSequences(cctx, reinterpret_cast<ZSTD_Sequence*>(&seq[0]), out_capacity, src_string.data(), src_string.size());
    // seq_size = ZSTDEXT_copyBlockSequencesTo(cctx, reinterpret_cast<ZSTD_Sequence*>(&seq[0]), out_capacity, 0);
    seq_size = ZSTD_mergeBlockDelimiters(reinterpret_cast<ZSTD_Sequence*>(&seq[0]), seq_size);
    
    // seq = std::vector<PYBIND_ZSTD_Sequence>(seq.begin(), seq.begin()+seq_size); // resize without initalizing
    // for(size_t i=0; i< seq_size; i++)
    // {
    //     std::cout << std::get<0>(seq[i]) << " " << std::get<1>(seq[i]) << " " << std::get<2>(seq[i]) << " " << std::get<3>(seq[i]) << std::endl;
    // }

    // get literals
    std::vector<char> literals;
    literals.reserve(out_capacity);
    size_t lit_size = ZSTDEXT_copyLiteralsTo(cctx, &literals[0], out_capacity);

    ZSTD_freeCCtx(cctx);

    // finally return data
    // NOTE: std::tuple implementation is dependent on platform
    // so we need to explicitly copy it, instead of directly using reinterpret_cast!
    // this may be slower, and could be improved in the future!
    std::vector<PYBIND_ZSTD_Sequence> seq_out;
    seq_out.reserve(seq_size);
    for(size_t i=0; i< seq_size; i++)
    {
        seq_out.push_back({seq[i].offset, seq[i].litLength, seq[i].matchLength, seq[i].rep});
    }
    ZSTD_LZ77_Data ret {py::bytes(literals.data(), lit_size), seq_out};
    return ret;

}

py::bytes zstd_lz77_reverse(ZSTD_LZ77_Data data, const std::string& dict_string)
{
    std::vector<char> out;

    std::string literal_string = static_cast<std::string>(std::get<0>(data));
    literal_string.reserve(literal_string.size() + WILDCOPY_OVERLENGTH);

    // estimate out_capacity
    size_t out_capacity = literal_string.size();
    for (auto seq: std::get<1>(data))
    {
        // out_capacity += std::get<1>(seq); // add litLength
        out_capacity += std::get<2>(seq); // add matchLength
    }
    out.reserve(out_capacity);


    // ZSTD_DCtx* dctx = ZSTD_createDCtx();
    // if (!dict_string.empty())
    // {
    //     ZSTD_DCtx_loadDictionary(dctx, (void*) dict_string.data(), dict_string.size());
    // }

    // sequence execution

    char* out_start = &out[0];
    const void* prefix_start = static_cast<const void*>(out.data());
    // We need to copy dict_string as a prefix in order for ZSTDEXT_execSequenceExtUsingDict to work
    // if (!dict_string.empty())
    // {
    //     // vector::assign or std::copy reallocate the vector, should reserve with new size later!
    //     out.assign(dict_string.begin(), dict_string.end());
    //     // std::copy(dict_string.begin(), dict_string.end(), out.begin());
    //     out_start += dict_string.size();
    //     out.reserve(out_capacity + dict_string.size());
    // }

    char* out_ptr = out_start;
    char* lit_ptr = &literal_string[0];
// #ifdef assert
    const char* out_end = out_ptr + out_capacity;
    const char* lit_end = literal_string.data() + literal_string.size();
// #endif
    // char* lit_end_ptr = literal_string.data() + literal_string.size();
    size_t out_size;
    for (auto seq: std::get<1>(data))
    {
        size_t offset = std::get<0>(seq);
        size_t litLength = std::get<1>(seq);
        size_t matchLength = std::get<2>(seq);
        // std::cout << offset << " " << litLength << " " << matchLength << " " << std::endl;
        out_size = ZSTDEXT_execSequenceExtUsingDict(
            (void*) lit_ptr, litLength, 
            matchLength, offset,
            out_ptr, out_end-out_ptr,
            prefix_start, dict_string.data(), dict_string.size());
        out_ptr += out_size;
        lit_ptr += litLength;
        // TODO: check ptr valid
        assert(out_ptr < out_end);
        assert(lit_ptr < lit_end);
    }

    // copy last literals
    while(lit_ptr < lit_end) {
        *out_ptr++ = *lit_ptr++;
        // out_ptr++;lit_ptr++;
    }

    // ZSTD_freeDCtx(dctx);
    // for(size_t i=0; i< out_capacity + dict_string.size(); i++)
    // {
    //     std::cout << out.data()[i];
    // }
    // std::cout << std::endl;

    // for(size_t i=0; i< out_capacity; i++)
    // {
    //     std::cout << out_start[i];
    // }
    // std::cout << std::endl;

    return py::bytes(out_start, out_ptr - out_start);
}

std::vector<ZSTD_Sequence> zstd_extract_lz77_sequences(const std::string& src_string, const std::string& dict_string, int level=0)
{
    std::vector<ZSTD_Sequence> seq;
    size_t out_capacity = ZSTD_compressBound(src_string.size());
    seq.reserve(out_capacity);

    /* Create the context. */
    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
    CHECK(cctx != NULL, "ZSTD_createCCtx() failed!");

    /* Set any parameters you want.
     * Here we set the compression level, and enable the checksum.
     */
    CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level) );
    // CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1) );
    // ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 4);
    CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_blockDelimiters, ZSTD_sf_noBlockDelimiters) );

    if (!dict_string.empty())
    {
        ZSTD_CCtx_loadDictionary(cctx, (void*) dict_string.data(), dict_string.size());
    }
    
    // ZSTDEXT_buildSeqStore(cctx, src_string.data(), src_string.size());
    // size_t seq_size = ZSTDEXT_copyBlockSequencesTo(cctx, reinterpret_cast<ZSTD_Sequence*>(&seq[0]), out_capacity);

    // NOTE: using reinterpret_cast is not safe!!
    size_t seq_size = ZSTD_generateSequences(cctx, reinterpret_cast<ZSTD_Sequence*>(&seq[0]), out_capacity, src_string.data(), src_string.size());
    seq_size = ZSTD_mergeBlockDelimiters(reinterpret_cast<ZSTD_Sequence*>(&seq[0]), seq_size);
    // seq.resize(seq_size);
    // seq.shrink_to_fit();

    ZSTD_freeCCtx(cctx);

    return std::vector<ZSTD_Sequence>(seq.begin(), seq.begin() + seq_size);
}


py::bytes zstd_compress_with_lz77_sequences(const std::string& src_string, std::vector<ZSTD_Sequence> seqs, const std::string& dict_string)
{
    std::vector<char> out;
    // estimate out_capacity
    size_t out_capacity = ZSTD_compressBound(src_string.size());
    out.reserve(out_capacity);

    /* Create the context. */
    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
    CHECK(cctx != NULL, "ZSTD_createCCtx() failed!");
    if (!dict_string.empty())
    {
        ZSTD_CCtx_loadDictionary(cctx, (void*) dict_string.data(), dict_string.size());
    }
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_blockDelimiters, ZSTD_sf_noBlockDelimiters);
    
    size_t out_size = ZSTD_compressSequences(cctx, &out[0], out_capacity, 
        reinterpret_cast<ZSTD_Sequence*>(seqs.data()), seqs.size(), 
        src_string.data(), src_string.size()
    );

    ZSTD_freeCCtx(cctx);

    return py::bytes(out.data(), out_size);
}

py::bytes zstd_extract_and_compress_with_lz77_sequences(const std::string& src_string, const std::string& dict_string, int level=0)
{
    auto seq = zstd_extract_lz77_sequences(src_string, dict_string, level);
    return zstd_compress_with_lz77_sequences(src_string, seq, dict_string);
}
    


std::vector<PYBIND_Tuple_ZSTD_Sequence_Ext> zstd_extract_lz77_phrases(const std::string& src_string, const std::string& dict_string, int level=0)
{
    std::vector<ZSTD_Sequence> seq;
    size_t out_capacity = ZSTD_compressBound(src_string.size());
    assert(out_capacity < 0xFFFF); // TODO: cannot copy longer than 0xFFFF literals with this function!
    seq.reserve(out_capacity);

    /* Create the context. */
    ZSTD_CCtx* const cctx = ZSTD_createCCtx();
    CHECK(cctx != NULL, "ZSTD_createCCtx() failed!");

    /* Set any parameters you want.
     * Here we set the compression level, and enable the checksum.
     */
    CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level) );
    // CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1) );
    // ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, 4);
    CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_blockDelimiters, ZSTD_sf_noBlockDelimiters) );

    if (!dict_string.empty())
    {
        ZSTD_CCtx_loadDictionary(cctx, (void*) dict_string.data(), dict_string.size());
    }

    // ZSTDEXT_buildSeqStore(cctx, src_string.data(), src_string.size());
    // size_t seq_size = ZSTDEXT_copyBlockSequencesTo(cctx, reinterpret_cast<ZSTD_Sequence*>(&seq[0]), out_capacity);

    // NOTE: using reinterpret_cast is not safe!!
    size_t seq_size = ZSTD_generateSequences(cctx, reinterpret_cast<ZSTD_Sequence*>(&seq[0]), out_capacity, src_string.data(), src_string.size());
    // seq_size = ZSTD_mergeBlockDelimiters(reinterpret_cast<ZSTD_Sequence*>(&seq[0]), seq_size);
    // seq.resize(seq_size);
    // seq.shrink_to_fit();

    std::vector<ZSTD_Sequence_Ext> seq_ext;
    seq_ext.resize(seq_size);
    seq_size = ZSTDEXT_copySequencesExtTo(cctx, &seq_ext[0], out_capacity);
    
    // reinterpret copy
    std::vector<PYBIND_Tuple_ZSTD_Sequence_Ext> seq_ext_out;
    seq_ext_out.reserve(seq_size);
    for (auto exec: seq_ext) {
    // for (size_t i=0; i<seq_size ; i++) {
        // auto exec = seq_ext[i];
        seq_ext_out.emplace_back(
            exec.offset,
            exec.matchLength,
            py::bytes((char*)exec.lit, exec.litLength),
            exec.rep
        );
    }


    // std::cout << seq[0].offset << " " 
    //     << seq[0].litLength << " "
    //     << seq[0].matchLength << " "
    //     << seq[0].rep << " "
    //     << std::endl;
    ZSTD_freeCCtx(cctx);

    // return seq;
    // return std::vector<PYBIND_ZSTD_Sequence>(seq.begin(), seq.begin() + seq_size);
    return seq_ext_out;
}

py::bytes zstd_exec_lz77_phrases(const std::vector<PYBIND_Tuple_ZSTD_Sequence_Ext> sequences, const std::string& dict_string)
{
    std::vector<char> out;
    std::vector<ZSTD_Sequence_Ext> seq_ext;

    // estimate out_capacity
    size_t out_capacity = 0;
    for (auto seq: sequences)
    {
        out_capacity += std::get<1>(seq);
        out_capacity += static_cast<std::string>(std::get<2>(seq)).size();
    }
    out.reserve(out_capacity);

    // ZSTD_DCtx* dctx = ZSTD_createDCtx();
    // if (!dict_string.empty())
    // {
    //     ZSTD_DCtx_loadDictionary(dctx, (void*) dict_string.data(), dict_string.size());
    // }

    char* out_ptr = out.data();
    size_t out_size;
    for (auto seq: sequences)
    {
        std::string literals = static_cast<std::string>(std::get<2>(seq));
        size_t offset = std::get<0>(seq);
        size_t matchLength = std::get<1>(seq);
        out_size = ZSTDEXT_execSequenceExtUsingDict(
            (void*) literals.data(), literals.size(), 
            matchLength, offset,
            out_ptr, out_capacity,
            out.data(), (void*) dict_string.data(), dict_string.size());
        out_ptr += out_size;
        // TODO: check ptr valid
    }

    // ZSTD_DCtx* dctx = ZSTD_createDCtx();

    // size_t out_size = ZSTDEXT_execSequencesExt(dctx, seq_ext.data(), seq_ext.size(),
    //     out.data(), out_capacity);

    // ZSTD_freeDCtx(dctx);

    return py::bytes(out.data(), out_ptr - out.data());;
}

py::bytes zdict_train_from_buffer(size_t dict_size, std::vector<py::bytes> buffer, 
    const std::optional<ZDICT_fastCover_params_t> train_params)
{
    std::vector<char> out;
    out.reserve(dict_size);

    // prepare samples
    std::vector<char> buffer_concat;
    std::vector<size_t> sample_sizes;
    size_t total_size = 0;
    for (std::string sample : buffer)
    {
        sample_sizes.push_back(sample.size());
        total_size += sample.size();
    }
    buffer_concat.reserve(total_size);
    for (std::string sample : buffer)
    {
        buffer_concat.insert(buffer_concat.end(), sample.begin(), sample.end());
    }

    // size_t out_size = ZDICT_trainFromBuffer(&out[0], dict_size,
    //                                         buffer_concat.data(), sample_sizes.data(), sample_sizes.size());
    // support custom params (from ZDICT_trainFromBuffer)
    ZDICT_fastCover_params_t params;
    if (!train_params.has_value()) {
        memset(&params, 0, sizeof(params));
        params.d = 8;
        params.steps = 4;
        // params.countUniqueFreq = 1;
        // params.scoreFreqMean = 1;
        /* Use default level since no compression level information is available */
        params.zParams.compressionLevel = ZSTD_CLEVEL_DEFAULT;
    }
    else {
        params = train_params.value();
    }
    // params.zParams.notificationLevel = 4;
    size_t out_size = ZDICT_optimizeTrainFromBuffer_fastCover(
        &out[0], dict_size,
        buffer_concat.data(), sample_sizes.data(), sample_sizes.size(),
        &params);

    // ZDICT_params_t parameters;
    // out_size = ZDICT_finalizeDictionary(&out[0], dict_size, out.data(), dict_size,
    //                                         buffer_concat.data(), sample_sizes.data(), sample_sizes.size(),
    //                                         parameters);
    
    return py::bytes(out.data(), out_size);;

}

py::bytes zdict_finalize_dictionary(std::string dict, std::vector<py::bytes> buffer)
{
    std::vector<char> out;
    out.reserve(dict.size());

    // prepare samples
    std::vector<char> buffer_concat;
    std::vector<size_t> sample_sizes;
    size_t total_size = 0;
    for (std::string sample : buffer)
    {
        sample_sizes.push_back(sample.size());
        total_size += sample.size();
    }
    buffer_concat.reserve(total_size);
    for (std::string sample : buffer)
    {
        buffer_concat.insert(buffer_concat.end(), sample.begin(), sample.end());
    }

    /* Use default level since no compression level information is available */
    ZDICT_params_t params = {ZSTD_CLEVEL_DEFAULT, 0, 0};
    size_t out_size = ZDICT_finalizeDictionary(&out[0], dict.size(), dict.data(), dict.size(),
                                            buffer_concat.data(), sample_sizes.data(), sample_sizes.size(),
                                            params);
    
    return py::bytes(out.data(), out_size);;

}

class ZstdWrapper
{
public:
    ZstdWrapper() { 
        cctx = ZSTD_createCCtx();
        CHECK(cctx != NULL, "ZSTD_createCCtx() failed!");
        dctx = ZSTD_createDCtx();
        CHECK(dctx != NULL, "ZSTD_createDCtx() failed!");
    }

    void load_dictionary(const std::string &dict) { 
        if (!dict.empty())
        {
            ZSTD_CCtx_loadDictionary(cctx, (void*) dict.data(), dict.size());
            ZSTD_DCtx_loadDictionary(dctx, (void*) dict.data(), dict.size());
        }
    }
     
    py::bytes compress_once(const std::string& src_string, int level=0)
    {
        std::vector<char> out;
        // std::string src_string = src();
        size_t out_capacity = ZSTD_compressBound(src_string.size());
        out.reserve(out_capacity);
        // size_t cs = ZSTD_compress(&out[0], out_capacity, src_string.data(), src_string.size(), level);
        
        ZSTD_CCtx_reset(cctx, ZSTD_reset_session_only);
        CHECK_ZSTD( ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level) );
        
        size_t cs = 0;
        cs = ZSTD_compress2(cctx, &out[0], out_capacity, src_string.data(), src_string.size());
        if (ZSTD_isError(cs)) {
            printf("ZSTD enc error: %s !!!\n", ZSTD_getErrorName(cs));
            return src_string;
        }
        return py::bytes(&out[0], cs);
    }

    py::bytes decompress_once(const std::string& src_string, size_t max_decode_size)
    {
        std::vector<char> out;
        out.reserve(max_decode_size);

        ZSTD_DCtx_reset(dctx, ZSTD_reset_session_only);

        size_t cs = 0;
        cs = ZSTD_decompressDCtx(dctx, &out[0], max_decode_size, src_string.data(), src_string.size());
        if (ZSTD_isError(cs)) {
            printf("ZSTD dec error: %s !!!\n", ZSTD_getErrorName(cs));
            return py::bytes("");;
        }
        return py::bytes(&out[0], cs);
    }

protected:
    ZSTD_CCtx* cctx;
    ZSTD_DCtx* dctx;
};


PYBIND11_MAKE_OPAQUE( std::vector<ZSTD_Sequence> );
// PYBIND11_MAKE_OPAQUE( std::vector<PYBIND_ZSTD_Sequence> );
// PYBIND11_MAKE_OPAQUE( std::vector<PYBIND_ZSTD_Sequence_Ext> );

PYBIND11_MODULE(zstd_wrapper, m) {
    m.def("zstd_compress", &zstd_compress, "", py::arg("src_string"), py::arg("dict_string")="", py::arg("level")=0);
    m.def("zstd_decompress", &zstd_decompress, "", py::arg("src_string"), py::arg("max_decode_size"), py::arg("dict_string")="");

    m.def("fse_compress", &fse_compress, "", py::arg("src_string"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=12);
    m.def("fse_decompress", &fse_decompress, "");

    // tANS
    m.def("fse_tans_compress", &fse_tans_compress, "", 
        py::arg("base_codes"), 
        py::arg("extra_codes"), 
        py::arg("extra_num_symbols"), 
        py::arg("maxSymbolValue")=255, 
        py::arg("tableLog")=12,
        py::arg("predefined_counts")=py::list()
    );

    m.def("fse_tans_decompress", &fse_tans_decompress, "",
        py::arg("src_string"), 
        py::arg("extra_num_symbols"), 
        py::arg("max_decode_size"), 
        py::arg("maxSymbolValue")=255, 
        py::arg("tableLog")=12,
        py::arg("predefined_counts")=py::list()
    );

    m.def("fse_tans_compress_advanced", &fse_tans_compress_advanced, "", 
        py::arg("base_codes"), 
        py::arg("extra_codes"), 
        py::arg("extra_num_symbols"), 
        py::arg("maxSymbolValue")=255, 
        py::arg("tableLog")=12,
        py::arg("predefined_count")=py::none()
    );

    m.def("fse_tans_decompress_advanced", &fse_tans_decompress_advanced, "",
        py::arg("src_string"), 
        py::arg("extra_num_symbols"), 
        py::arg("max_decode_size"), 
        py::arg("maxSymbolValue")=255, 
        py::arg("tableLog")=12,
        py::arg("predefined_count")=py::none()
    );
    m.def("huf_compress", &huf_compress, "", py::arg("src_string"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=12);
    m.def("huf_decompress", &huf_decompress, "");

    // fse binding
    m.def("fse_create_ctable_using_cnt", &fse_create_ctable_using_cnt, "", py::arg("count"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=12);
    m.def("fse_create_dtable_using_cnt", &fse_create_dtable_using_cnt, "", py::arg("count"), py::arg("maxSymbolValue")=255, py::arg("tableLog")=12);
    m.def("fse_compress_using_ctable", &fse_compress_using_ctable, "");
    m.def("fse_decompress_using_dtable", &fse_decompress_using_dtable, "");

    py::class_<ZSTD_Sequence>(m, "ZSTD_Sequence")
        .def(py::init<unsigned int, unsigned int, unsigned int, unsigned int>())
        .def_readwrite("offset", &ZSTD_Sequence::offset)
        .def_readwrite("litLength", &ZSTD_Sequence::litLength)
        .def_readwrite("matchLength", &ZSTD_Sequence::matchLength)
        .def_readwrite("rep", &ZSTD_Sequence::rep)
        .def("__str__", [](){
            return "ZSTD_Sequence";
        })
        ;
    py::bind_vector< std::vector< ZSTD_Sequence > >( m, "ZSTD_Sequence_Vector" );
    // py::bind_vector< std::vector< PYBIND_ZSTD_Sequence > >( m, "PYBIND_ZSTD_Sequence_Vector" );

    // py::class_<PYBIND_ZSTD_Sequence_Ext>(m, "ZSTD_Sequence_Ext")
    //     .def(py::init<unsigned int, unsigned int, py::bytes, unsigned int>())
    //     .def_readwrite("offset", &PYBIND_ZSTD_Sequence_Ext::offset)
    //     .def_readwrite("matchLength", &PYBIND_ZSTD_Sequence_Ext::matchLength)
    //     .def_readwrite("literals", &PYBIND_ZSTD_Sequence_Ext::literals)
    //     .def_readwrite("rep", &PYBIND_ZSTD_Sequence_Ext::rep)
    //     ;        
    // py::bind_vector< std::vector< PYBIND_ZSTD_Sequence_Ext > >( m, "ZSTD_Sequence_Ext_Vector" );

    m.def("zstd_lz77_forward", &zstd_lz77_forward, "", py::arg("src_string"), py::arg("dict_string")="", py::arg("config")=py::dict());
    m.def("zstd_lz77_reverse", &zstd_lz77_reverse, "", py::arg("data"), py::arg("dict_string")="");

    m.def("zstd_extract_lz77_sequences", &zstd_extract_lz77_sequences, "", py::arg("src_string"), py::arg("dict_string")="", py::arg("level")=0);
    m.def("zstd_compress_with_lz77_sequences", &zstd_compress_with_lz77_sequences, "", py::arg("src_string"), py::arg("seqs"), py::arg("dict_string")="");
    m.def("zstd_extract_and_compress_with_lz77_sequences", &zstd_extract_and_compress_with_lz77_sequences, "", py::arg("src_string"), py::arg("dict_string")="", py::arg("level")=0);

    m.def("zstd_extract_lz77_phrases", &zstd_extract_lz77_phrases, "", py::arg("src_string"), py::arg("dict_string")="", py::arg("level")=0);
    m.def("zstd_exec_lz77_phrases", &zstd_exec_lz77_phrases, "", py::arg("sequences"), py::arg("dict_string")="");

    // parameter setting
    py::enum_<ZSTD_cParameter>(m, "ZSTD_cParameter")
        .value("ZSTD_c_compressionLevel", ZSTD_cParameter::ZSTD_c_compressionLevel)
        .value("ZSTD_c_windowLog", ZSTD_cParameter::ZSTD_c_windowLog)
        .value("ZSTD_c_hashLog", ZSTD_cParameter::ZSTD_c_hashLog)
        .value("ZSTD_c_chainLog", ZSTD_cParameter::ZSTD_c_chainLog)
        .value("ZSTD_c_searchLog", ZSTD_cParameter::ZSTD_c_searchLog)
        .value("ZSTD_c_minMatch", ZSTD_cParameter::ZSTD_c_minMatch)
        .value("ZSTD_c_targetLength", ZSTD_cParameter::ZSTD_c_targetLength)
        .export_values(); // TODO: all enums

    // zdict
    m.def("zdict_train_from_buffer", &zdict_train_from_buffer, "", 
        py::arg("dict_size"), 
        py::arg("buffer"), 
        py::arg("train_params")=py::none()
    );
    m.def("zdict_finalize_dictionary", &zdict_finalize_dictionary, "");
    // zdict parameter setting
    py::class_<ZDICT_fastCover_params_t>(m, "ZDICT_fastCover_params_t")
        .def(py::init([](){
            // auto params = std::unique_ptr<ZDICT_fastCover_params_t>(new ZDICT_fastCover_params_t());
            ZDICT_fastCover_params_t* params = new ZDICT_fastCover_params_t;
            memset(params, 0, sizeof(*params));
            return std::unique_ptr<ZDICT_fastCover_params_t>(params);
        }))
        // .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, double, unsigned, unsigned, unsigned, unsigned, unsigned, ZDICT_params_t>(),
        //     py::arg("k")=0,
        //     py::arg("d")=0,
        //     py::arg("f")=0,
        //     py::arg("steps")=0,
        //     py::arg("nbThreads")=0,
        //     py::arg("splitPoint")=0,
        //     py::arg("accel")=0,
        //     py::arg("shrinkDict")=0,
        //     py::arg("shrinkDictMaxReggression")=0,
        //     py::arg("countUniqueFreq")=0,
        //     py::arg("scoreFreqMean")=0,
        //     py::arg("zParams")=0
        // )
        .def_readwrite("k", &ZDICT_fastCover_params_t::k)
        .def_readwrite("d", &ZDICT_fastCover_params_t::d)
        .def_readwrite("f", &ZDICT_fastCover_params_t::f)
        .def_readwrite("steps", &ZDICT_fastCover_params_t::steps)
        .def_readwrite("nbThreads", &ZDICT_fastCover_params_t::nbThreads)
        .def_readwrite("splitPoint", &ZDICT_fastCover_params_t::splitPoint)
        .def_readwrite("accel", &ZDICT_fastCover_params_t::accel)
        .def_readwrite("shrinkDict", &ZDICT_fastCover_params_t::shrinkDict)
        .def_readwrite("shrinkDictMaxRegression", &ZDICT_fastCover_params_t::shrinkDictMaxRegression)
        .def_readwrite("countUniqueFreq", &ZDICT_fastCover_params_t::countUniqueFreq)
        .def_readwrite("scoreFreqMean", &ZDICT_fastCover_params_t::scoreFreqMean)
        .def_readwrite("zParams", &ZDICT_fastCover_params_t::zParams);

    py::class_<ZDICT_params_t>(m, "ZDICT_params_t")
        .def(py::init([](){
            // auto params = std::unique_ptr<ZDICT_fastCover_params_t>(new ZDICT_fastCover_params_t());
            ZDICT_params_t* params = new ZDICT_params_t;
            memset(params, 0, sizeof(*params));
            return std::unique_ptr<ZDICT_params_t>(params);
        }))
        .def_readwrite("compressionLevel", &ZDICT_params_t::compressionLevel)
        .def_readwrite("notificationLevel", &ZDICT_params_t::notificationLevel)
        .def_readwrite("dictID", &ZDICT_params_t::dictID);


    py::class_<ZstdWrapper>(m, "ZstdWrapper")
        .def(py::init<>())
        .def("load_dictionary", &ZstdWrapper::load_dictionary)
        .def("compress_once", &ZstdWrapper::compress_once, py::arg("src_string"), py::arg("level")=0)
        .def("decompress_once", &ZstdWrapper::decompress_once, py::arg("src_string"), py::arg("max_decode_size"));
        // TODO:
        // .def("compress_stream", &ZstdWrapper::compress_stream)
        // .def("compress_stream_end", &ZstdWrapper::compress_stream)
        // .def("decompress_stream", &ZstdWrapper::decompress_stream)
        // .def("decompress_stream_end", &ZstdWrapper::decompress_stream);
}