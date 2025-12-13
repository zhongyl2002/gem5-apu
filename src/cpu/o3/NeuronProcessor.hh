#ifndef __RVNE_HH__
#define __RVNE_HH__

#include <iostream>
#include <memory>

#include "base/logging.hh"
#include "cpu/reg_class.hh"
#include "cpu/regfile.hh"
#include "debug/RVNE.hh"

namespace gem5
{

namespace o3
{

enum NeuronInst:uint8_t
{
    LA_WV,
    LA_SV,
    MOVW_V,
    MOVD_V,
    MOVW_WV,
    MOVD_WV,
    MOVW_SV,
    MOVD_SV,
    MOVD_G,
    MOVW_SO,
    MOVA_SO,
    ACCH,
    ACCA,
    ACCM,
    DOTH,
    DOTA,
    UPDS,
    UPDG,
    UPDA,
    MAC_NS,
    CLRS_NS,
    CLRG_NS,
    CLRA_NS,
    CLRA_V,
};

struct rvne_wvr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_svr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_sor
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_ncr
{
    static const std::string name;
    static const int numRegs;
    using valType = int;
};

struct rvne_nvr
{
    static const std::string name;
    static const int numRegs;
    using valType = int;
};

struct rvne_nsr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint8_t;
};

struct rvne_npr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint8_t;
};

struct rvne_ntr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_vtr
{
    static const std::string name;
    static const int numRegs;
    using valType = int;
};

struct rvne_rpr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_lpr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_rvr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_rcr
{
    static const std::string name;
    static const int numRegs;
    using valType = uint32_t;
};

struct rvne_spm
{
    static const std::string name;
    static const int len;
    using valType = uint8_t;
};

class NeuronProcessor
{

private:
    std::vector<rvne_wvr::valType> wvr;
    std::vector<rvne_svr::valType> svr;
    std::vector<rvne_sor::valType> sor;
    std::vector<rvne_ncr::valType> ncr;
    std::vector<rvne_nvr::valType> nvr;
    std::vector<rvne_nsr::valType> nsr;
    std::vector<rvne_npr::valType> npr;
    std::vector<rvne_ntr::valType> ntr;
    std::vector<rvne_vtr::valType> vtr;
    std::vector<rvne_rpr::valType> rpr;
    std::vector<rvne_lpr::valType> lpr;
    std::vector<rvne_rvr::valType> rvr;
    std::vector<rvne_rcr::valType> rcr;
    std::vector<rvne_spm::valType> spm;

    enum movExtRegType
    {
        WVRIdx,
        SVRIdx,
        SORIdx,
        NCRIdx,
        NVRIdx,
        NSRIdx,
        NPRIdx,
        NTRIdx,
        VTRIdx,
        RPRIdx,
        LPRIdx,
        RVRIdx,
        RCRIdx
    };

private:
    /**
     * @brief Move data into a specific register helper. The switch-case body
     * of mov instruction
     * @tparam regInfo rvne reg info struct, e.g. rvne_wvr
     * @param reg the reg to be written
     * @param data the data to be written
     * @param is32Bit whether the data is 32-bit which decides the block size
     * @param idx the index to write to
     */
    template <typename regInfo>
    void
    movToExtHelper(std::vector<typename regInfo::valType>& reg,
                   uint64_t data,
                   bool is32Bit,
                   uint32_t idx);

    /**
     * @brief Move data into a GPR helper.
     * The switch-case body of mov instruction
     * @tparam regInfo rvne reg info struct, e.g. rvne_wvr
     * @param reg the reg to be read
     * @param idx the index to be read
     */
    template <typename regInfo>
    RegVal
    movToGPRHelper(std::vector<typename regInfo::valType>& reg,
                   uint32_t idx);

    /**
     * @brief Move data from one extended register to another
     *        and clean the source register.
     *
     * @note sizeof(srcElem) = sizeof(dstElem)
     * @tparam srcRegInfo Source register info struct, e.g. rvne_sor
     * @tparam dstRegInfo Destination register info struct, e.g. rvne_svr
     * @param srcIdx Source register index
     * @param dstIdx Destination register index
     */
    template <typename srcRegInfo, typename dstRegInfo>
    void
    extRegCutHelper(std::vector<typename srcRegInfo::valType>& src,
                    uint32_t srcIdx,
                    std::vector<typename dstRegInfo::valType>& dst,
                    uint32_t dstIdx);

    /**
     * @brief accumulation compute function, partial sum of wvr * svr.
     *
     * @note wvrBlockSize and svrBlockSize should match
     * @param wvrBlockSize # of elements per block in wvr
     * @param svrBlockSize same as wvrBlockSize
     * @return RegVal
     */
    int
    accCompute(uint32_t wvrBlockIdx,
               uint32_t wvrBlockSize,
               uint32_t svrBlockIdx,
               uint32_t svrBlockSize);

    /**
     * @brief entrance of acch and acca, Call accCompute for each.
     */
    void
    accHelper(int wvrBlockSize,
              int svrBlockSize,
              RegVal rs1_val,
              RegVal rs2_val);

    /**
     * @brief LIF neuron state update helper
     * @return int make a new spike or not
     */
    int LIF(int32_t &I, int32_t &V, uint8_t &S, uint8_t &period, bool excitNeuron);
    /**
     * @brief statue update helper
     *
     * @param flattenNeuronIdx the idx of neuron to be updated
     * @param flattenOutSpikeIdx the idx of out spike to be stored
     */
    void updateHelper(uint32_t flattenNeuronIdx, uint32_t flattenOutSpikeIdx);

public:
    NeuronProcessor();
    ~NeuronProcessor() {}

    /**
     * @{
     * @name SPM Read Instructions
     */
    /**
     * @brief Load 512-bit data from SPM to WVR
     */
    void la_wv(RegVal rs1_val, RegVal rs2_val);
    /**
     * @brief Load 512-bit data from SPM to SVR
     */
    void la_sv(RegVal rs1_val, RegVal rs2_val);
    /** @} */

    /**
     * @{
     * @name Data Move Instructions
     */
    /**
     * @brief mov low 32-bit GPR content to movExtRegType[2-12]
     * @param rs1_val GPR source value, R[rs1][31:0]
     * @param rs2_val destination register index
     */
    void movw_v(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief mov 64-bit GPR content to movExtRegType[2-12]
     * @param rs1_val GPR source value, R[rs1]
     * @param rs2_val destination register index
     */
    void movd_v(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief WVR[R[rs2][31:0]] = R[rs1][31:0].
     * @param rs1_val GPR source value, R[rs1]
     * @param rs2_val GPR source value, R[rs2]
     */
    void movw_wv(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief {WVR[R[rs2][31:0]*2],WVR[R[rs2][31:0]*2+1]} = R[rs1].
     * @param rs1_val GPR source value, R[rs1]
     * @param rs2_val GPR source value, R[rs2]
     */
    void movd_wv(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief SVR[R[rs2][31:0]] = R[rs1][31:0].
     * @param rs1_val GPR source value, R[rs1]
     * @param rs2_val GPR source value, R[rs2]
     */
    void movw_sv(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief {SVR[R[rs2][31:0]*2],SVR[R[rs2][31:0]*2+1]} = R[rs1].
     * @param rs1_val GPR source value, R[rs1]
     * @param rs2_val GPR source value, R[rs2]
     */
    void movd_sv(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief mov 64-bit movExtRegType[0-12] content to GPR
     *
     * @param rs1_val ext reg idx
     * @return RegVal content of ext reg
     */
    RegVal movd_g(RegVal rs1_val);

    /**
     * @brief mov SOR[R[rs1]] to SVR[R[rs2]] and clean SOR[R[rs1]]
     *
     * @param rs1_val source ext reg idx, which will be cleaned after mov
     * @param rs2_val destination ext reg idx
     */
    void movw_so(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief mov all SOR to SVR and clean SOR
     */
    void mova_so();

    /** @} */

    /**
     * @{
     * @name Accumulation Instructions
     */

    /**
     * @brief ncr = ncr + sor * wvr
     */
    void acch(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief ncr = ncr + sor + wvr
     */
    void acca(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief ncr = ncr + sor + wvr
     */
    void accm(RegVal rs2_val);

    /** @} */

    /**
     * @{
     * @name Dot Product Instructions
     */
    /**
     * @brief doth with 32 neurons
     */
    void doth(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief dota with 128 neurons
     */
    void dota(RegVal rs1_val, RegVal rs2_val);

    /** @} */

    /**
     * @{
     * @name State Update Instructions
     */
    /**
     * @brief update single neuron state
     */
    void upds(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief update one group(32 neurons per group) state
     */
    void updg(RegVal rs1_val, RegVal rs2_val);

    /**
     * @brief update all neurons state
     */
    void upda();

    /** @} */

    /**
     * @{
     * @name FC Compute Instruction
     */
    /**
     * @brief R[rd]= \sum_{i=0}^{7} NSR[R[rs2]*8 + i] * R[rs1][i*8+7:i*8].
     *
     * @param rs1_val data from general purpose register,
     *                signed 8-bit weights packed in 64-bit
     * @param rs2_val NSR group index in [0, 127],
     *                unsigned 8-bit state
     * @return RegVal FC compute result, which will store in R[rd]
     */
    RegVal mac_ns(RegVal rs1_val, RegVal rs2_val);

    /** @} */


    /**
     * @{
     * @name Clear Instructions
     */
    /**
     * @brief NSR[R[rs2]] = 0.
     * @param rs2_val [0, 1023]
     */
    void clrs_ns(RegVal rs2_val);

    /**
     * @brief NSR[R[rs2]*32] ~ NSR[R[rs2]*32+31] = 0.
     * @param rs2_val [0, 31]
     */
    void clrg_ns(RegVal rs2_val);

    /**
     * @brief NSR = 0.
     */
    void clra_ns();

    /**
     * @brief NCR[0]~NCR[1023] = 0;
     *        NVR[0]~NVR[1023] = 0;
     *        NSR[0]~NSR[1023] = 0;
     *        NPR[0]~NPR[1023] = 0.
     */
    void clra_v();

    /** @} */

    /**
     * @{
     * @name Interface to CPU
     */
    /**
     * @brief Coprocessor operate interface
     *
     * @param instClass Neuron instruction
     * @param rs1 Source register 1
     * @param rs2 Source register 2
     * @param mem Pointer to memory which is used by mem inst (NOT USED NOW)
     * @return RegVal Result of the operation
     */
    RegVal
    coprocessorExec(NeuronInst instClass,
                    RegVal rs1,
                    RegVal rs2,
                    uint8_t* mem);

    /** @} */

};

} // namespace o3

} // namespace gem5

#endif
