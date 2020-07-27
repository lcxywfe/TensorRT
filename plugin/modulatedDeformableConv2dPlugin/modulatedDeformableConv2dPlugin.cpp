#include "modulatedDeformableConv2dPlugin.h"
#include <cuda_fp16.h>


using namespace nvinfer1;
using nvinfer1::plugin::ModulatedDeformableConv2d;
using nvinfer1::plugin::ModulatedDeformableConv2dPluginCreator;

namespace
{
// for DEBUG
using DType = __half;

const char* MODULATED_DEFORMABLE_CONV2D_PLUGIN_VERSION{"001"};
const char* MODULATED_DEFORMABLE_CONV2D_PLUGIN_NAME{"ModulatedDeformableConv2d_TRT"};

// copy host float values to device T values
template <typename T>
Weights copyToDevice(const void* hostData, int64_t count);

template <>
Weights copyToDevice<float>(const void* hostData, int64_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, count};
}

template <>
Weights copyToDevice<__half>(const void* hostData, int64_t count)
{
    void* deviceData;
    void* deviceTmp;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(__half)));
    CUASSERT(cudaMalloc(&deviceTmp, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceTmp, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    dcnFloat2half(static_cast<const float*>(deviceTmp), static_cast<__half*>(deviceData), count);
    CUASSERT(cudaFree(deviceTmp));
    return Weights{DataType::kHALF, deviceData, count};
}

template <typename T>
void serializeFromDevice(char*& hostBuffer, const Weights& deviceWeights)
{
    CUASSERT(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(T), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(T);
}

template<typename T>
Weights deserializeToDevice(const char*& hostBuffer, int64_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(T)));
    CUASSERT(cudaMemcpy(deviceData, hostBuffer, count * sizeof(T), cudaMemcpyHostToDevice));
    DataType data_type = DataType::kFLOAT;
    if (std::is_same<T, __half>::value)
        data_type = DataType::kHALF;
    Weights w{data_type, deviceData, count};
    hostBuffer += count * sizeof(T);
    return w;
}

} // namespace


// ==========================ModulatedDeformableConv2d==========================

ModulatedDeformableConv2d::ModulatedDeformableConv2d(const std::pair<int, int> stride,
    const std::pair<int, int> padding, const std::pair<int, int> dilation, const std::pair<int, int> kernel_size,
    const int out_channels, const int in_channels, const Weights& weights, const Weights& bias, int im2col_step,
    bool clone)
    : mStride(stride)
    , mPadding(padding)
    , mDilation(dilation)
    , mKernelSize(kernel_size)
    , mOutChannels(out_channels)
    , mInChannels(in_channels)
    , mIm2colStep(im2col_step)
{
    ASSERT(weights.type == bias.type);
    ASSERT(weights.count == mOutChannels * mInChannels * mKernelSize.first * mKernelSize.second);
    ASSERT(bias.count == out_channels);

    if (clone)
    {
        // clone
        void* deviceWeights;
        void* deviceBias;
        CUASSERT(cudaMalloc(&deviceWeights, weights.count * sizeof(DType)));
        CUASSERT(cudaMalloc(&deviceBias, bias.count * sizeof(DType)));
        CUASSERT(cudaMemcpy(deviceWeights, weights.values, weights.count * sizeof(DType), cudaMemcpyDeviceToDevice));
        CUASSERT(cudaMemcpy(deviceBias, bias.values, bias.count * sizeof(DType), cudaMemcpyDeviceToDevice));
        mWeights = {weights.type, deviceWeights, weights.count};
        mBias = {bias.type, deviceBias, bias.count};
    }
    else
    {
        // created by plugin creator, copy float to DType
        mWeights = copyToDevice<DType>(weights.values, weights.count);
        mBias = copyToDevice<DType>(bias.values, bias.count);
    }
    cublasCreate(&mCublas);
}

ModulatedDeformableConv2d::ModulatedDeformableConv2d(const void* buffer, size_t length)
{
    const char *d = static_cast<const char*>(buffer);
    mStride = {read<int>(d), read<int>(d)};
    mPadding = {read<int>(d), read<int>(d)};
    mDilation = {read<int>(d), read<int>(d)};
    mKernelSize = {read<int>(d), read<int>(d)};

    mOutChannels = read<int>(d);
    mInChannels = read<int>(d);

    int weights_count = read<int>(d);
    mWeights = deserializeToDevice<DType>(d, weights_count);
    int bias_count = read<int>(d);
    mBias = deserializeToDevice<DType>(d, bias_count);
    int mIm2colStep = read<int>(d);
    ASSERT(d == static_cast<const char*>(buffer) + length);

    cublasCreate(&mCublas);
}

IPluginV2DynamicExt* ModulatedDeformableConv2d::clone() const
{
    IPluginV2DynamicExt* plugin = new ModulatedDeformableConv2d(
        mStride, mPadding, mDilation, mKernelSize, mOutChannels, mInChannels, mWeights, mBias, mIm2colStep, true);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

int ModulatedDeformableConv2d::initialize()
{
    return 0;
};

void ModulatedDeformableConv2d::terminate()
{
    CUBLASASSERT(cublasDestroy(mCublas));
}

void ModulatedDeformableConv2d::destroy()
{
    CUASSERT(cudaFree(const_cast<void*>(mWeights.values)));
    CUASSERT(cudaFree(const_cast<void*>(mBias.values)));
    delete this;
}

int ModulatedDeformableConv2d::getNbOutputs() const
{
    return 1;
};

DataType ModulatedDeformableConv2d::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(nbInputs == 3);
    ASSERT(index == 0);
    ASSERT(inputTypes[0] == inputTypes[1] && inputTypes[0] == inputTypes[2]);
    ASSERT(inputTypes[0] == nvinfer1::DataType::kFLOAT || inputTypes[0] == nvinfer1::DataType::kHALF);
    return inputTypes[0];
}

DimsExprs ModulatedDeformableConv2d::getOutputDimensions(
    int index, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)
{
    ASSERT(nbInputs == 3);
    ASSERT(index == 0);
    ASSERT(inputs[0].nbDims == 4 && inputs[1].nbDims == 4 && inputs[2].nbDims == 4);
    ASSERT(inputs[0].d[1]->isConstant() && inputs[0].d[1]->getConstantValue() % mInChannels == 0);
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.constant(mOutChannels * (inputs[0].d[1]->getConstantValue() / mInChannels));
    output.d[2] = inputs[0].d[2];
    output.d[3] = inputs[0].d[3];
    return output;
}

bool ModulatedDeformableConv2d::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    return (inOut[pos].type == mWeights.type && inOut[pos].format == nvinfer1::PluginFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

void ModulatedDeformableConv2d::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    // TODO remove the limitation
    ASSERT(in[0].desc.dims.d[1] / mInChannels == 1);  // group == 1
    ASSERT(in[1].min.d[1] / (2 * mKernelSize.first * mKernelSize.second) == 1);  // deformable group == 1
    ASSERT(mStride.first == 1 && mStride.second == 1 && mPadding.first == 1 && mPadding.second == 1
        && mKernelSize.first == 3 && mKernelSize.second == 3);  // in shape == out shape

    ASSERT(nbInputs == 3 && nbOutputs == 1);
    ASSERT(in[0].desc.dims.d[1] % mInChannels == 0 && out[0].desc.dims.d[1] % mOutChannels == 0);
    ASSERT(in[0].desc.dims.d[1] / mInChannels == out[0].desc.dims.d[1] / mOutChannels);
    ASSERT(in[1].min.d[1] == in[1].max.d[1] && in[1].min.d[1] % (2 * mKernelSize.first * mKernelSize.second) == 0);
    ASSERT(in[2].min.d[1] == in[2].max.d[1] && in[2].min.d[1] % (mKernelSize.first * mKernelSize.second) == 0);
    ASSERT(in[1].min.d[1] / in[2].min.d[1] == 2);
}

size_t ModulatedDeformableConv2d::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nboutputs) const
{
    int n_parallel_imgs = get_greatest_divisor_below_bound(inputs[0].dims.d[0], mIm2colStep);
    int out_h = ((inputs[0].dims.d[2] + 2 * mPadding.first - mKernelSize.first) / mStride.first) + 1;
    int out_w = ((inputs[0].dims.d[3] + 2 * mPadding.second - mKernelSize.second) / mStride.second) + 1;
    size_t column_size = mInChannels * mKernelSize.first * mKernelSize.second * n_parallel_imgs * out_h * out_w * sizeof(DType);
    size_t outbuf_size = out_h * out_w * mOutChannels * inputs[0].dims.d[0] * sizeof(DType);
    return column_size + outbuf_size;
}

int ModulatedDeformableConv2d::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    DeformConv2d_forward(
        static_cast<const DType*>(inputs[0]),
        static_cast<const DType*>(inputs[1]),
        static_cast<const DType*>(inputs[2]),
        static_cast<const DType*>(mWeights.values),
        static_cast<const DType*>(mBias.values),
        static_cast<DType*>(workspace),
        static_cast<DType*>(outputs[0]),
        inputDesc[0].dims.d[0], mInChannels, inputDesc[0].dims.d[2], inputDesc[0].dims.d[3],
        mOutChannels, mKernelSize.first, mKernelSize.second,
        mStride,
        mPadding,
        mDilation,
        inputDesc[0].dims.d[1] / mInChannels,
        inputDesc[1].dims.d[1] / (2 * mKernelSize.first * mKernelSize.second),
        mIm2colStep,
        stream,
        mCublas
    );
}

size_t ModulatedDeformableConv2d::getSerializationSize() const
{
    ASSERT(mWeights.type == mBias.type);
    if (mWeights.type == DataType::kFLOAT)
        return sizeof(int) * 13 + sizeof(float) * (mWeights.count + mBias.count);
    else
        return sizeof(int) * 13 + sizeof(__half) * (mWeights.count + mBias.count);
}


void ModulatedDeformableConv2d::serialize(void* buffer) const
{
    char* d = static_cast<char*>(buffer);
    write(d, mStride.first);
    write(d, mStride.second);
    write(d, mPadding.first);
    write(d, mPadding.second);
    write(d, mDilation.first);
    write(d, mDilation.second);
    write(d, mKernelSize.first);
    write(d, mKernelSize.second);
    write(d, mOutChannels);
    write(d, mInChannels);

    ASSERT(mWeights.type == mBias.type);
    write(d, static_cast<int>(mWeights.count));
    if (mWeights.type == DataType::kFLOAT)
    {
        serializeFromDevice<float>(d, mWeights);
    }
    else
    {
        serializeFromDevice<__half>(d, mWeights);
    }
    write(d, static_cast<int>(mBias.count));
    if (mBias.type == DataType::kFLOAT)
    {
        serializeFromDevice<float>(d, mBias);
    }
    else
    {
        serializeFromDevice<__half>(d, mBias);
    }
    write(d, mIm2colStep);
    ASSERT(d == static_cast<char*>(buffer) + getSerializationSize());
}

const char* ModulatedDeformableConv2d::getPluginType() const
{
    return MODULATED_DEFORMABLE_CONV2D_PLUGIN_NAME;
}

const char* ModulatedDeformableConv2d::getPluginVersion() const
{
    return MODULATED_DEFORMABLE_CONV2D_PLUGIN_VERSION;
}

void ModulatedDeformableConv2d::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = "";
}

const char* ModulatedDeformableConv2d::getPluginNamespace() const
{
    return mPluginNamespace;
}

void ModulatedDeformableConv2d::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

void ModulatedDeformableConv2d::detachFromContext()
{
};

// ====================ModulatedDeformableConv2dPluginCreator===================

PluginFieldCollection ModulatedDeformableConv2dPluginCreator::mFC{};
std::vector<PluginField> ModulatedDeformableConv2dPluginCreator::mPluginAttributes;

ModulatedDeformableConv2dPluginCreator::ModulatedDeformableConv2dPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("kernel_size", nullptr, PluginFieldType::kINT32, 2));
    mPluginAttributes.emplace_back(PluginField("out_channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("in_channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("im2col_step", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ModulatedDeformableConv2dPluginCreator::getPluginName() const
{
    return MODULATED_DEFORMABLE_CONV2D_PLUGIN_NAME;
}

const char* ModulatedDeformableConv2dPluginCreator::getPluginVersion() const
{
    return MODULATED_DEFORMABLE_CONV2D_PLUGIN_VERSION;
}

const PluginFieldCollection* ModulatedDeformableConv2dPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* ModulatedDeformableConv2dPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::vector<float> weights_values(0);
    std::vector<float> bias_values(0);
    const PluginField* fields = fc->fields;
    std::pair<int, int> stride, padding, dilation, kernel_size;
    int out_channels, in_channels, im2col_step;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "im2col_step"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 1);
            im2col_step = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "stride"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 2);
            const auto* w = static_cast<const int*>(fields[i].data);
            stride = {*w, *(w + 1)};
        }
        else if (!strcmp(attr_name, "padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 2);
            const auto* w = static_cast<const int*>(fields[i].data);
            padding = {*w, *(w + 1)};
        }
        else if (!strcmp(attr_name, "dilation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 2);
            const auto* w = static_cast<const int*>(fields[i].data);
            dilation = {*w, *(w + 1)};
        }
        else if (!strcmp(attr_name, "kernel_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 2);
            const auto* w = static_cast<const int*>(fields[i].data);
            kernel_size = {*w, *(w + 1)};
        }
        else if (!strcmp(attr_name, "out_channels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 1);
            out_channels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "in_channels"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            ASSERT(fields[i].length == 1);
            in_channels = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "weights"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            weights_values.reserve(fields[i].length);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < fields[i].length; j++)
            {
                weights_values.push_back(*w);
                w++;
            }
        }
        else
        {
            ASSERT(!strcmp(attr_name, "bias"));
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            bias_values.reserve(fields[i].length);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < fields[i].length; j++)
            {
                bias_values.push_back(*w);
                w++;
            }
        }
    }
    ASSERT(weights_values.size() == out_channels * in_channels * kernel_size.first * kernel_size.second);
    ASSERT(bias_values.size() == out_channels);
    Weights weights{DataType::kFLOAT, weights_values.data(), static_cast<int64_t>(weights_values.size())};
    Weights bias{DataType::kFLOAT, bias_values.data(), static_cast<int64_t>(bias_values.size())};

    ModulatedDeformableConv2d* obj = new ModulatedDeformableConv2d(
        stride, padding, dilation, kernel_size, out_channels, in_channels, weights, bias, im2col_step);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* ModulatedDeformableConv2dPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ModulatedDeformableConv2d::destroy()
    ModulatedDeformableConv2d* obj = new ModulatedDeformableConv2d(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
