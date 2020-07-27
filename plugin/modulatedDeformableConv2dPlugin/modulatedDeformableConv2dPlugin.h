#ifndef TRT_MODULATED_DEFORMABLE_CONV2D_PLUGIN_H
#define TRT_MODULATED_DEFORMABLE_CONV2D_H
#include "cudnn.h"
#include "kernel.h"
#include "plugin.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class ModulatedDeformableConv2d : public IPluginV2DynamicExt
{
public:
    ModulatedDeformableConv2d(const std::pair<int, int> stride, const std::pair<int, int> padding,
        const std::pair<int, int> dilation, const std::pair<int, int> kernel_size, const int out_channels,
        const int in_channels, const Weights& weights, const Weights& bias, int im2col_step, bool clone=false);

    ModulatedDeformableConv2d(const void* buffer, size_t length);

    ~ModulatedDeformableConv2d() override = default;

    IPluginV2DynamicExt* clone() const override;

    int initialize() override;
    void terminate() override;
    void destroy() override;

    int getNbOutputs() const override;
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
    DimsExprs getOutputDimensions(int index, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void detachFromContext() override;

private:
    cublasHandle_t mCublas{};
    std::pair<int, int> mStride;
    std::pair<int, int> mPadding;
    std::pair<int, int> mDilation;
    std::pair<int, int> mKernelSize;
    int mOutChannels, mInChannels, mIm2colStep;
    Weights mWeights;
    Weights mBias;
    const char* mPluginNamespace;
};

class ModulatedDeformableConv2dPluginCreator : public BaseCreator
{
public:
    ModulatedDeformableConv2dPluginCreator();

    ~ModulatedDeformableConv2dPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_MODULATED_DEFORMABLE_CONV2D_PLUGIN_H