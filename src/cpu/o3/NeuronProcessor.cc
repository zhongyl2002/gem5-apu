#include "cpu/o3/NeuronProcessor.hh"

#include <algorithm>

namespace gem5
{
namespace o3
{

const std::string rvne_wvr::name = "wvr";
const std::string rvne_svr::name = "svr";
const std::string rvne_sor::name = "sor";
const std::string rvne_ncr::name = "ncr";
const std::string rvne_nvr::name = "nvr";
const std::string rvne_nsr::name = "nsr";
const std::string rvne_npr::name = "npr";
const std::string rvne_ntr::name = "ntr";
const std::string rvne_vtr::name = "vtr";
const std::string rvne_rpr::name = "rpr";
const std::string rvne_lpr::name = "lpr";
const std::string rvne_rvr::name = "rvr";
const std::string rvne_rcr::name = "rcr";
const std::string rvne_spm::name = "spm";
const int rvne_wvr::numRegs = 128;
const int rvne_svr::numRegs = 32;
const int rvne_sor::numRegs = 32;
const int rvne_ncr::numRegs = 1024;
const int rvne_nvr::numRegs = 1024;
const int rvne_nsr::numRegs = 1024;
const int rvne_npr::numRegs = 1024;
const int rvne_ntr::numRegs = 32;
const int rvne_vtr::numRegs = 2;
const int rvne_rpr::numRegs = 2;
const int rvne_lpr::numRegs = 2;
const int rvne_rvr::numRegs = 2;
const int rvne_rcr::numRegs = 2;
const int rvne_spm::len = 2048;

NeuronProcessor::NeuronProcessor()
{
    wvr = std::vector<rvne_wvr::valType>(rvne_wvr::numRegs);
    svr = std::vector<rvne_svr::valType>(rvne_svr::numRegs);
    sor = std::vector<rvne_sor::valType>(rvne_sor::numRegs);
    ncr = std::vector<rvne_ncr::valType>(rvne_ncr::numRegs);
    nvr = std::vector<rvne_nvr::valType>(rvne_nvr::numRegs);
    nsr = std::vector<rvne_nsr::valType>(rvne_nsr::numRegs);
    npr = std::vector<rvne_npr::valType>(rvne_npr::numRegs);
    ntr = std::vector<rvne_ntr::valType>(rvne_ntr::numRegs);
    vtr = std::vector<rvne_vtr::valType>(rvne_vtr::numRegs);
    rpr = std::vector<rvne_rpr::valType>(rvne_rpr::numRegs);
    lpr = std::vector<rvne_lpr::valType>(rvne_lpr::numRegs);
    rvr = std::vector<rvne_rvr::valType>(rvne_rvr::numRegs);
    rcr = std::vector<rvne_rcr::valType>(rvne_rcr::numRegs);
    spm = std::vector<rvne_spm::valType>(rvne_spm::len);
    for (int i = 0; i < rvne_spm::len; i++)
    {
        spm[i] = i % 256;
    }
}

template <typename regInfo>
void
NeuronProcessor::movToExtHelper(std::vector<typename regInfo::valType>& reg,
                             uint64_t data,
                             bool is32Bit,
                             uint32_t idx)
{
    int block_size = 0;
    if (is32Bit) {
        block_size = (sizeof(uint32_t)) / sizeof(typename regInfo::valType);
    } else {
        block_size = (sizeof(uint64_t)) / sizeof(typename regInfo::valType);
    }

    int right_bound = (int)(regInfo::numRegs / block_size) - 1;
    if (idx > right_bound) {
        panic("Invalid index %d for %s with block size %d, must be [0, %d]",
              idx, regInfo::name.c_str(), block_size, right_bound);
    }

    int elem_size = sizeof(typename regInfo::valType);
    int start_idx = idx * block_size;
    for (int i = 0; i < block_size; i++) {
        switch (elem_size)
        {
        case sizeof(uint8_t):
            reg[start_idx + i] =
                static_cast<typename regInfo::valType>((data >> (8 * i)) & 0xFFull);
            break;
        case sizeof(uint32_t):
            reg[start_idx + i] =
                static_cast<typename regInfo::valType>((data >> (32 * i)) & 0xFFFFFFFFull);
            break;
        default:
            panic("Unsupported element size %d in movHelper.", elem_size);
            break;
        }
    }
}

template <typename regInfo>
RegVal
NeuronProcessor::movToGPRHelper(std::vector<typename regInfo::valType>& reg,
                                uint32_t idx)
{
    int ret_size = sizeof(RegVal);
    int elem_size = sizeof(typename regInfo::valType);
    int block_size = (ret_size / elem_size);
    int right_bound =
        (int)(regInfo::numRegs * elem_size / ret_size) - 1;
    if (idx > right_bound) {
        panic("Invalid index %d for %s with GPR size(%d), must be [0, %d]",
              idx, regInfo::name.c_str(), ret_size, right_bound);
    }

    RegVal ret = 0;
    int start_idx = idx * block_size;
    for (int i = 0; i < (ret_size / elem_size); i++) {
        switch (elem_size)
        {
        case sizeof(uint8_t):
            ret |=
                (static_cast<RegVal>(reg[start_idx + i]) & 0xFFull) << (8 * i);
            break;
        case sizeof(uint32_t):
            ret |=
                (static_cast<RegVal>(reg[start_idx + i]) & 0xFFFFFFFFull) << (32 * i);
            break;
        default:
            panic("Unsupported element size %d in movHelper.", elem_size);
            break;
        }
    }

    return ret;
}

template <typename srcRegInfo, typename dstRegInfo>
void
NeuronProcessor::extRegCutHelper(std::vector<typename srcRegInfo::valType>& src,
                                 uint32_t srcIdx,
                                 std::vector<typename dstRegInfo::valType>& dst,
                                 uint32_t dstIdx)
{
    int srcElemSize = sizeof(typename srcRegInfo::valType);
    int dstElemSize = sizeof(typename dstRegInfo::valType);

    if (srcIdx >= srcRegInfo::numRegs) {
        panic("Invalid source index %d for %s, must be [0, %d]",
              srcIdx, srcRegInfo::name.c_str(), srcRegInfo::numRegs - 1);
    }
    if (dstIdx >= dstRegInfo::numRegs) {
        panic("Invalid source index %d for %s, must be [0, %d]",
              dstIdx, dstRegInfo::name.c_str(), dstRegInfo::numRegs - 1);
    }

    dst[dstIdx] = static_cast<typename srcRegInfo::valType>(src[srcIdx]);
    src[srcIdx] = static_cast<typename srcRegInfo::valType>(0);
}

int
NeuronProcessor::accCompute(uint32_t wvrBlockIdx,
                            uint32_t wvrBlockSize,
                            uint32_t svrBlockIdx,
                            uint32_t svrBlockSize)
{
    int wvrBoundIdx = (int)(rvne_wvr::numRegs / wvrBlockSize) - 1;
    if (wvrBlockIdx > wvrBoundIdx) {
        panic("Invalid wvrBlockIdx %d for WVR with block size %d, \
            must be [0, %d]", wvrBlockIdx, wvrBlockSize, wvrBoundIdx);
    }

    int svrBoundIdx = (int)(rvne_svr::numRegs / svrBlockSize) - 1;
    if (svrBlockIdx > svrBoundIdx) {
        panic("Invalid svrBlockIdx %d for SVR with block size %d, \
            must be [0, %d]", svrBlockIdx, svrBlockSize, svrBoundIdx);
    }

    int weightBits = 4;
    int wvrElemBits = sizeof(rvne_wvr::valType) * 8;
    int wvrIdxStart = wvrBlockIdx * wvrBlockSize;
    int spikeBits = 1;
    int svrElemBits = sizeof(rvne_svr::valType) * 8;
    int svrIdxStart = svrBlockIdx * svrBlockSize;
    int loops = svrElemBits * svrBlockSize;

    int acc = 0;
    for (int i = 0; i < loops; i++)
    {
        int _svrIdx = (i * spikeBits) / svrElemBits + svrIdxStart;
        int _svrOffset = (i * spikeBits) % svrElemBits;
        int _wvrIdx = (i * weightBits) / wvrElemBits + wvrIdxStart;
        int _wvrOffset = (i * weightBits) % wvrElemBits;
        int exist = (svr[_svrIdx] >> (_svrOffset)) & 0x1;
        if (exist) {
            int sdata =
                ((int32_t)(((wvr[_wvrIdx] >> (_wvrOffset)) & 0xf)) << 28) >> 28;
            acc += sdata;
        }
    }

    return acc;
}

void
NeuronProcessor::accHelper(int wvrBlockSize,
                           int svrBlockSize,
                           RegVal rs1_val,
                           RegVal rs2_val)
{
    if (wvr.empty() || svr.empty() || ncr.empty()) {
        panic("NeuronProcessor WVR or SVR register not initialized properly.");
    }

    int rightBound = rvne_ncr::numRegs - 1;
    if (rs2_val > rightBound) {
        panic("Invalid rs2_val (%d), must be [0, %d]", rs2_val, rightBound);
    }

    uint32_t wvrBlockIdx = (rs1_val >> 32) & 0xffffffff;
    uint32_t svrBlockIdx = rs1_val & 0xffffffff;
    int delta = accCompute(wvrBlockIdx,
                           wvrBlockSize,
                           svrBlockIdx,
                           svrBlockSize);
    int ntrIdx = rs2_val / (sizeof(ntr[0]) * 8);
    int ntrOffset = rs2_val % (sizeof(ntr[0]) * 8);
    int type = (ntr[ntrIdx] >> ntrOffset) & 0x1;
    int lprTmp = lpr[type];
    int weightShift = (lprTmp >> 25) & 0x1f;
    delta = delta << weightShift;
    ncr[rs2_val] = (uint32_t)((int)(ncr[rs2_val]) + delta);
}

int
NeuronProcessor::LIF(int &i, int &v, uint8_t &s, uint8_t &period, bool excitNeuron)
{
    int init_vthreshold = !excitNeuron ? vtr[0] : vtr[1];
    uint32_t init_trefractory_period = !excitNeuron ? rpr[0] : rpr[1];

    uint32_t lpr_elem = !excitNeuron ? lpr[0] : lpr[1];
    uint8_t weight_shift = lpr_elem >> 25 & 0x1f;
    uint8_t v_shift = lpr_elem >> 20 & 0x1f;
    uint8_t i_shift = lpr_elem >> 15 & 0x1f;
    uint8_t rnd_v_shift = lpr_elem >> 10 & 0x1f;
    uint8_t rnd_i_shift = lpr_elem >> 5 & 0x1f;
    uint8_t v_i_shift = lpr_elem & 0x1f;

    int reset_voltage = !excitNeuron ? rvr[0] : rvr[1];
    int reset_current = !excitNeuron ? rcr[0] : rcr[1];

    int rnd_i = i >> i_shift;
    // 此处修改需要调整位置，因为i被后续使用
    int prev_i = i;
    i = i - (rnd_i >> rnd_i_shift);
    if (period == 0) {
        int rnd_v = v >> v_shift;
        v = v - (rnd_v >> rnd_v_shift) + (prev_i >> v_i_shift);
        if (v > init_vthreshold) {
            s ++;
            v = reset_voltage;
            i = reset_current;
            rnd_v = 0;
            rnd_i = 0;
            period = init_trefractory_period;
            return 1;
        }
    }
    else {
        period --;
    }
    return 0;
}

void
NeuronProcessor::updateHelper(uint32_t flattenNeuronIdx, uint32_t flattenSORIdx)
{
    if (ncr.empty() || nvr.empty() || nsr.empty() || npr.empty()
        || sor.empty() || vtr.empty() || rpr.empty() || lpr.empty()
        || rvr.empty() || rcr.empty()) {
        panic("NeuronProcessor NCR/NVR/NSR/NPR/SOR/VTR/RPR/LPR/RVR/RCR register not initialized properly.");
    }

    if (flattenNeuronIdx > 1023)
    {
        panic("Invalid flattenNeuronIdx (%d), must be [0, 1023]", flattenNeuronIdx);
    }

    if (flattenSORIdx > 1023){
        panic("Invalid flattenSORIdx (%d), must be [0, 1023]", flattenSORIdx);
    }

    uint32_t ntrIdx = flattenNeuronIdx / 32;
    uint32_t ntrOffset = flattenNeuronIdx % 32;
    bool excitNeuron = (ntr[ntrIdx] >> ntrOffset) & 0x1;

    int spike = LIF(ncr[flattenNeuronIdx],
                    nvr[flattenNeuronIdx],
                    nsr[flattenNeuronIdx],
                    npr[flattenNeuronIdx],
                    excitNeuron);

    uint32_t sorIdx = flattenSORIdx / 32;
    uint32_t sorOffset = flattenSORIdx % 32;
    if (spike) {
        sor[sorIdx] |= (0x1 << sorOffset);
    } else {
        sor[sorIdx] &= ~(0x1 << sorOffset);
    }
}

void
NeuronProcessor::la_wv(RegVal rs1_val, RegVal rs2_val)
{
    if (wvr.empty()) {
        panic("NeuronProcessor WVR register not initialized properly.");
    }

    int memSize= 64;
    int wvrElemSize = sizeof(typename rvne_wvr::valType);
    int spmElemSize = sizeof(typename rvne_spm::valType);
    int memBlockSize = memSize / wvrElemSize;
    int wvrBlockSize = wvrElemSize / spmElemSize;

    int wvrBlockIdx = static_cast<int>(rs2_val & 0xffffffff);
    int rightBound = (int)(rvne_wvr::numRegs / memBlockSize) - 1;
    if (wvrBlockIdx > rightBound) {
        panic("Invalid wvrBlockIdx %d for WVR with block size %d, \
            must be [0, %d]", wvrBlockIdx, wvrBlockSize, rightBound);
    }

    if (rs1_val + memSize > rvne_spm::len) {
        panic("Load address out of bound, rs1_val = %d, memSize = %d, spm len = %d",
              rs1_val, memSize, rvne_spm::len);
    }

    int startWvrIdx = wvrBlockIdx * memBlockSize;
    for (int i = 0; i < memBlockSize; i++) {
        uint32_t wvrBlock = 0;
        for (int j = 0; j < wvrBlockSize; j++)
        {
            wvrBlock = wvrBlock +
                (spm[rs1_val + i * wvrBlockSize + j] << (j * spmElemSize * 8));
        }
        wvr[startWvrIdx + i] =
            static_cast<rvne_wvr::valType>(wvrBlock);
    }
}

void
NeuronProcessor::la_sv(RegVal rs1_val, RegVal rs2_val)
{
    if (svr.empty()) {
        panic("NeuronProcessor SVR register not initialized properly.");
    }

    int memSize= 64;
    int svrElemSize = sizeof(typename rvne_svr::valType);
    int spmElemSize = sizeof(typename rvne_spm::valType);
    int memBlockSize = memSize / svrElemSize;
    int svrBlockSize = svrElemSize / spmElemSize;

    int svrBlockIdx = static_cast<int>(rs2_val & 0xffffffff);
    int rightBound = (int)(rvne_svr::numRegs / memBlockSize) - 1;
    if (svrBlockIdx > rightBound) {
        panic("Invalid svrBlockIdx %d for SVR with block size %d, \
            must be [0, %d]", svrBlockIdx, svrBlockSize, rightBound);
    }

    if (rs1_val + memSize > rvne_spm::len) {
        panic("Load address out of bound, rs1_val = %d, memSize = %d, spm len = %d",
              rs1_val, memSize, rvne_spm::len);
    }

    int startWvrIdx = svrBlockIdx * memBlockSize;
    for (int i = 0; i < memBlockSize; i++) {
        uint32_t svrBlock = 0;
        for (int j = 0; j < svrBlockSize; j++)
        {
            svrBlock = svrBlock +
                (spm[rs1_val + i * svrBlockSize + j] << (j * spmElemSize * 8));
        }
        svr[startWvrIdx + i] =
            static_cast<rvne_svr::valType>(svrBlock);
    }
}

void
NeuronProcessor::movw_v(RegVal rs1_val, RegVal rs2_val)
{
    if (sor.empty() || ncr.empty() || nvr.empty() || nsr.empty() ||
        npr.empty() || ntr.empty() || vtr.empty() || rpr.empty() ||
        lpr.empty() || rvr.empty() || rcr.empty()) {
        panic("NeuronProcessor registers not initialized properly.");
    }

    movExtRegType type = static_cast<movExtRegType>(rs2_val >> 32 & 0xffffffff);
    uint32_t idx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    switch (type)
    {
    case WVRIdx:
        break;
    case SVRIdx:
        break;
    case SORIdx:
        movToExtHelper<rvne_sor>(sor, rs1_val, true, idx);
        break;
    case NCRIdx:
        movToExtHelper<rvne_ncr>(ncr, rs1_val, true, idx);
        break;
    case NVRIdx:
        movToExtHelper<rvne_nvr>(nvr, rs1_val, true, idx);
        break;
    case NSRIdx:
        movToExtHelper<rvne_nsr>(nsr, rs1_val, true, idx);
        break;
    case NPRIdx:
        movToExtHelper<rvne_npr>(npr, rs1_val, true, idx);
        break;
    case NTRIdx:
        movToExtHelper<rvne_ntr>(ntr, rs1_val, true, idx);
        break;
    case VTRIdx:
        movToExtHelper<rvne_vtr>(vtr, rs1_val, true, idx);
        break;
    case RPRIdx:
        movToExtHelper<rvne_rpr>(rpr, rs1_val, true, idx);
        break;
    case LPRIdx:
        movToExtHelper<rvne_lpr>(lpr, rs1_val, true, idx);
        break;
    case RVRIdx:
        movToExtHelper<rvne_rvr>(rvr, rs1_val, true, idx);
        break;
    case RCRIdx:
        movToExtHelper<rvne_rcr>(rcr, rs1_val, true, idx);
        break;
    }
}

void
NeuronProcessor::movd_v(RegVal rs1_val, RegVal rs2_val)
{
    if (sor.empty() || ncr.empty() || nvr.empty() || nsr.empty() ||
        npr.empty() || ntr.empty() || vtr.empty() || rpr.empty() ||
        lpr.empty() || rvr.empty() || rcr.empty()) {
        panic("NeuronProcessor registers not initialized properly.");
    }

    movExtRegType type = static_cast<movExtRegType>(rs2_val >> 32 & 0xffffffff);
    uint32_t idx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    switch (type)
    {
    case WVRIdx:
        break;
    case SVRIdx:
        break;
    case SORIdx:
        movToExtHelper<rvne_sor>(sor, rs1_val, false, idx);
        break;
    case NCRIdx:
        movToExtHelper<rvne_ncr>(ncr, rs1_val, false, idx);
        break;
    case NVRIdx:
        movToExtHelper<rvne_nvr>(nvr, rs1_val, false, idx);
        break;
    case NSRIdx:
        movToExtHelper<rvne_nsr>(nsr, rs1_val, false, idx);
        break;
    case NPRIdx:
        movToExtHelper<rvne_npr>(npr, rs1_val, false, idx);
        break;
    case NTRIdx:
        movToExtHelper<rvne_ntr>(ntr, rs1_val, false, idx);
        break;
    case VTRIdx:
        movToExtHelper<rvne_vtr>(vtr, rs1_val, false, idx);
        break;
    case RPRIdx:
        movToExtHelper<rvne_rpr>(rpr, rs1_val, false, idx);
        break;
    case LPRIdx:
        movToExtHelper<rvne_lpr>(lpr, rs1_val, false, idx);
        break;
    case RVRIdx:
        movToExtHelper<rvne_rvr>(rvr, rs1_val, false, idx);
        break;
    case RCRIdx:
        movToExtHelper<rvne_rcr>(rcr, rs1_val, false, idx);
        break;
    }
}

void
NeuronProcessor::movw_wv(RegVal rs1_val, RegVal rs2_val)
{
    if (wvr.empty()) {
        panic("NeuronProcessor WVR register not initialized properly.");
    }

    uint32_t idx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    movToExtHelper<rvne_wvr>(wvr, rs1_val, true, idx);
}

void
NeuronProcessor::movd_wv(RegVal rs1_val, RegVal rs2_val)
{
    if (wvr.empty()) {
        panic("NeuronProcessor WVR register not initialized properly.");
    }

    uint32_t idx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    movToExtHelper<rvne_wvr>(wvr, rs1_val, false, idx);
}

void
NeuronProcessor::movw_sv(RegVal rs1_val, RegVal rs2_val)
{
    if (svr.empty()) {
        panic("NeuronProcessor SVR register not initialized properly.");
    }

    uint32_t idx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    movToExtHelper<rvne_svr>(svr, rs1_val, true, idx);
}

void
NeuronProcessor::movd_sv(RegVal rs1_val, RegVal rs2_val)
{
    if (svr.empty()) {
        panic("NeuronProcessor SVR register not initialized properly.");
    }

    uint32_t idx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    movToExtHelper<rvne_svr>(svr, rs1_val, false, idx);
}

RegVal
NeuronProcessor::movd_g(RegVal rs1_val)
{
    if (sor.empty() || ncr.empty() || nvr.empty() || nsr.empty() ||
        npr.empty() || ntr.empty() || vtr.empty() || rpr.empty() ||
        lpr.empty() || rvr.empty() || rcr.empty()) {
        panic("NeuronProcessor registers not initialized properly.");
    }

    movExtRegType type = static_cast<movExtRegType>(rs1_val >> 32 & 0xffffffff);
    uint32_t idx = static_cast<uint32_t>(rs1_val & 0xffffffff);
    switch (type)
    {
    case WVRIdx:
        return movToGPRHelper<rvne_wvr>(wvr, idx);
        break;
    case SVRIdx:
        return movToGPRHelper<rvne_svr>(svr, idx);
        break;
    case SORIdx:
        return movToGPRHelper<rvne_sor>(sor, idx);
        break;
    case NCRIdx:
        return movToGPRHelper<rvne_ncr>(ncr, idx);
        break;
    case NVRIdx:
        return movToGPRHelper<rvne_nvr>(nvr, idx);
        break;
    case NSRIdx:
        return movToGPRHelper<rvne_nsr>(nsr, idx);
        break;
    case NPRIdx:
        return movToGPRHelper<rvne_npr>(npr, idx);
        break;
    case NTRIdx:
        return movToGPRHelper<rvne_ntr>(ntr, idx);
        break;
    case VTRIdx:
        return movToGPRHelper<rvne_vtr>(vtr, idx);
        break;
    case RPRIdx:
        return movToGPRHelper<rvne_rpr>(rpr, idx);
        break;
    case LPRIdx:
        return movToGPRHelper<rvne_lpr>(lpr, idx);
        break;
    case RVRIdx:
        return movToGPRHelper<rvne_rvr>(rvr, idx);
        break;
    case RCRIdx:
        return movToGPRHelper<rvne_rcr>(rcr, idx);
        break;
    }
    return static_cast<RegVal>(-1);
}

void
NeuronProcessor::movw_so(RegVal rs1_val, RegVal rs2_val)
{
    if (sor.empty() || svr.empty()) {
        panic("NeuronProcessor SOR or SVR register not initialized properly.");
    }

    uint32_t srcIdx = static_cast<uint32_t>(rs1_val & 0xffffffff);
    uint32_t dstIdx = static_cast<uint32_t>(rs2_val & 0xffffffff);
    extRegCutHelper<rvne_sor, rvne_svr>(sor, srcIdx, svr, dstIdx);
}

void
NeuronProcessor::mova_so()
{
    if (sor.empty() || svr.empty()) {
        panic("NeuronProcessor SOR or SVR register not initialized properly.");
    }

    if (rvne_sor::numRegs != rvne_svr::numRegs) {
        panic("NeuronProcessor SOR and SVR register size mismatch.");
    }

    for (int i = 0; i < rvne_svr::numRegs; i++)
    {
        extRegCutHelper<rvne_sor, rvne_svr>(sor, i, svr, i);
    }
}

void
NeuronProcessor::acch(RegVal rs1_val, RegVal rs2_val)
{
    int acchWvrBlockSize = 4;
    int acchSvrBlockSize = 1;

    accHelper(acchWvrBlockSize, acchSvrBlockSize, rs1_val, rs2_val);
}

void
NeuronProcessor::acca(RegVal rs1_val, RegVal rs2_val)
{
    int accaWvrBlockSize = 16;
    int accaSvrBlockSize = 4;

    accHelper(accaWvrBlockSize, accaSvrBlockSize, rs1_val, rs2_val);
}

void
NeuronProcessor::accm(RegVal rs2_val)
{
    int accmWvrBlockSize = 16;
    int accmSvrBlockSize = 4;
    int accmNcrBlockSize = 8;

    for (uint32_t i = 0; i < accmNcrBlockSize; i++)
    {
        RegVal rs1_val = ((RegVal)i << 32) | i;
        uint32_t ncrIdx = rs2_val * accmNcrBlockSize + i;
        accHelper(accmWvrBlockSize, accmSvrBlockSize, rs1_val, ncrIdx);
    }
}

void
NeuronProcessor::doth(RegVal rs1_val, RegVal rs2_val)
{
    if (wvr.empty() || svr.empty() || ncr.empty()) {
        panic("NeuronProcessor WVR or SVR or NCR register not initialized properly.");
    }

    int wvrBlockSize = 4;
    int ncrBlockSize = 32;
    int computeCnt = 32;
    int weightBits = 4;
    int wvrBlockIdx = static_cast<int>((rs1_val >> 32) & 0xffffffff);
    int wvrBlockBound = (rvne_wvr::numRegs / wvrBlockSize);
    int svrFlattenIdx= static_cast<int>(rs1_val & 0xffffffff);
    int svrFlattenBound = 1024;
    int ncrBlockIdx = static_cast<int>(rs2_val);
    int ncrBlockBound = (rvne_ncr::numRegs / ncrBlockSize);

    if (wvrBlockIdx > wvrBlockBound - 1) {
        panic("Invalid wvrBlockIdx (%d), must be [0, %d]", wvrBlockIdx, wvrBlockBound);
    }

    if (svrFlattenIdx > svrFlattenBound - 1) {
        panic("Invalid svrFlattenIdx (%d), must be [0, %d]", svrFlattenIdx, svrFlattenBound);
    }

    if (ncrBlockIdx > ncrBlockBound - 1) {
        panic("Invalid ncrBlockIdx (%d), must be [0, %d]", ncrBlockIdx, ncrBlockBound);
    }

    int svrIdx = svrFlattenIdx / (sizeof(typename rvne_svr::valType) * 8);
    int svrOffset = svrFlattenIdx / (sizeof(typename rvne_svr::valType) * 8);
    int inSpike = (svr[svrIdx] >> svrOffset) & 0x1;
    if (inSpike == 0) return;
    int neuronType = (ntr[svrIdx] >> svrOffset) & 0x1;
    int weightShift = (lpr[neuronType] >> 25) & 0x1f;
    int wvrIdxStart = wvrBlockIdx * wvrBlockSize;
    int ncrIdxStart = ncrBlockIdx * ncrBlockSize;
    for (int i = 0; i < computeCnt; i++)
    {
        int offset = i * weightBits;
        int deltaIdx = (offset) / (sizeof(typename rvne_wvr::valType) * 8);
        int deltaOffset = (offset) % (sizeof(typename rvne_wvr::valType) * 8);
        int uweight = (wvr[wvrIdxStart + deltaIdx] >> deltaOffset) & 0xf;
        int weight = ((int32_t)(uweight << 28)) >> 28;
        weight = weight << weightShift;
        ncr[ncrIdxStart + i] = (uint32_t)((int)ncr[ncrIdxStart + i] + weight);
    }
}

void
NeuronProcessor::dota(RegVal rs1_val, RegVal rs2_val)
{
    if (wvr.empty() || svr.empty() || ncr.empty()) {
        panic("NeuronProcessor WVR or SVR or NCR register not initialized properly.");
    }

    int wvrBlockSize = 16;
    int ncrBlockSize = 128;
    int wvrBlockIdx = static_cast<int>((rs1_val >> 32) & 0xffffffff);
    int wvrBlockBound = (rvne_wvr::numRegs / wvrBlockSize);
    int svrFlattenIdx= static_cast<int>(rs1_val & 0xffffffff);
    int svrFlattenBound = 1024;
    int ncrBlockIdx = static_cast<int>(rs2_val);
    int ncrBlockBound = (rvne_ncr::numRegs / ncrBlockSize);

    if (wvrBlockIdx > wvrBlockBound - 1) {
        panic("Invalid wvrBlockIdx (%d), must be [0, %d]", wvrBlockIdx, wvrBlockBound);
    }

    if (svrFlattenIdx > svrFlattenBound - 1) {
        panic("Invalid svrFlattenIdx (%d), must be [0, %d]", svrFlattenIdx, svrFlattenBound);
    }

    if (ncrBlockIdx > ncrBlockBound - 1) {
        panic("Invalid ncrBlockIdx (%d), must be [0, %d]", ncrBlockIdx, ncrBlockBound);
    }

    for (int i = 0; i < 4; i++)
    {
        RegVal _rs1_val = ((((uint64_t)(wvrBlockIdx * 4 + i)) << 32) + svrFlattenIdx);
        RegVal _rs2_val = (ncrBlockIdx * 4 + i);
        doth(_rs1_val, _rs2_val);
    }
}

void
NeuronProcessor::upds(RegVal rs1_val, RegVal rs2_val)
{
    uint32_t flattenNeuronIdx = static_cast<uint32_t>(rs1_val);
    uint32_t flattenSORIdx = static_cast<uint32_t>(rs2_val);
    updateHelper(flattenNeuronIdx, flattenSORIdx);
}


void
NeuronProcessor::updg(RegVal rs1_val, RegVal rs2_val)
{
    int blockSize = 32;
    uint32_t flattenNeuronIdxStart = static_cast<uint32_t>(rs1_val) * blockSize;
    uint32_t flattenSORIdxStart = static_cast<uint32_t>(rs2_val) * blockSize;
    for (int i = 0; i < blockSize; i++)
    {
        updateHelper(flattenNeuronIdxStart + i, flattenSORIdxStart + i);
    }
}

void
NeuronProcessor::upda()
{
    int blockSize = 1024;
    for (int i = 0; i < blockSize; i++)
    {
        updateHelper(i, i);
    }
}

RegVal
NeuronProcessor::mac_ns(RegVal rs1_val, RegVal rs2_val)
{
    if (nsr.empty()) {
        panic("NeuronProcessor NSR register not initialized properly.");
    }
    int block_size = 8;
    int right_bound = (rvne_nsr::numRegs / block_size) - 1;
    if (rs2_val > right_bound) {
        panic("Invalid rs2_val (%d), must be [0, %d]", rs2_val, right_bound);
    }

    int start_idx = rs2_val * block_size;

    int64_t acc = 0;
    uint64_t wpack = static_cast<uint64_t>(rs1_val);

    for (int i = 0; i < block_size; ++i) {
        int8_t weight = static_cast<int8_t>((wpack >> (8 * i)) & 0xFFull);
        uint8_t neuron = static_cast<uint8_t>(nsr[start_idx + i]);
        acc += static_cast<int64_t>(weight) * static_cast<int64_t>(neuron);
    }

    return static_cast<RegVal>(acc);
}

void
NeuronProcessor::clrs_ns(RegVal rs2_val)
{
    if (nsr.empty()) {
        panic("NeuronProcessor NSR register not initialized properly.");
    }
    int right_bound = rvne_nsr::numRegs - 1;
    if (rs2_val > right_bound) {
        panic("Invalid rs2_val (%d), must be [0, %d]", rs2_val, right_bound);
    }
    nsr[rs2_val] = static_cast<rvne_nsr::valType>(0);
}

void
NeuronProcessor::clrg_ns(RegVal rs2_val)
{
    if (nsr.empty()) {
        panic("NeuronProcessor NSR register not initialized properly.");
    }
    int block_size = 32;
    int right_bound = (rvne_nsr::numRegs / block_size) - 1;
    if (rs2_val > right_bound) {
        panic("Invalid rs2_val (%d), must be [0, %d]", rs2_val, right_bound);
    }
    int start_idx = rs2_val * block_size;
    std::fill(nsr.begin() + start_idx,
              nsr.begin() + start_idx + block_size, static_cast<rvne_nsr::valType>(0));
}

void
NeuronProcessor::clra_ns()
{
    if (nsr.empty()) {
        panic("NeuronProcessor NSR register not initialized properly.");
    }
    std::fill(nsr.begin(), nsr.end(), static_cast<rvne_nsr::valType>(0));
}

void
NeuronProcessor::clra_v()
{
    if (ncr.empty() || nvr.empty() || nsr.empty() || npr.empty()) {
        panic("NeuronProcessor registers not initialized properly.");
    }
    std::fill(ncr.begin(),
              ncr.end(),
              static_cast<rvne_ncr::valType>(0));
    std::fill(nvr.begin(),
              nvr.end(),
              static_cast<rvne_nvr::valType>(0));
    // todo：指令集spec文档中对nsr和npr仅复位了前256项
    std::fill(nsr.begin(),
              nsr.end(),
              static_cast<rvne_nsr::valType>(0));
    std::fill(npr.begin(),
              npr.end(),
              static_cast<rvne_npr::valType>(0));
}

RegVal
NeuronProcessor::coprocessorExec(NeuronInst instClass,
                                 RegVal rs1,
                                 RegVal rs2,
                                 uint8_t* mem)
{
    RegVal ret = 0;
    switch (instClass)
    {
    case LA_WV:
        la_wv(rs1, rs2);
        break;
    case LA_SV:
        la_sv(rs1, rs2);
        break;
    case MOVW_V:
        movw_v(rs1, rs2);
        break;
    case MOVD_V:
        movd_v(rs1, rs2);
        break;
    case MOVW_WV:
        movw_wv(rs1, rs2);
        break;
    case MOVD_WV:
        movd_wv(rs1, rs2);
        break;
    case MOVW_SV:
        movw_sv(rs1, rs2);
        break;
    case MOVD_SV:
        movd_sv(rs1, rs2);
        break;
    case MOVD_G:
        ret = movd_g(rs1);
        break;
    case MOVW_SO:
        movw_so(rs1, rs2);
        break;
    case MOVA_SO:
        mova_so();
        break;
    case ACCH:
        acch(rs1, rs2);
        break;
    case ACCA:
        acca(rs1, rs2);
        break;
    case ACCM:
        accm(rs2);
        break;
    case DOTH:
        doth(rs1, rs2);
        break;
    case DOTA:
        dota(rs1, rs2);
        break;
    case UPDS:
        upds(rs1, rs2);
        break;
    case UPDG:
        updg(rs1, rs2);
        break;
    case UPDA:
        upda();
        break;
    case MAC_NS:
        ret = mac_ns(rs1, rs2);
        break;
    case CLRS_NS:
        clrs_ns(rs2);
        break;
    case CLRG_NS:
        clrg_ns(rs2);
        break;
    case CLRA_NS:
        clra_ns();
        break;
    case CLRA_V:
        clra_v();
        break;
    }
    return ret;
}

}
}
