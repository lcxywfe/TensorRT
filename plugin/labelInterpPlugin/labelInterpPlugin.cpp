/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel.h"
#include "labelInterpPlugin.h"
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::LabelInterpPlugin;
using nvinfer1::plugin::LabelInterpPluginCreator;

namespace
{

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

constexpr const char* LABEL_INTERP_PLUGIN_VERSION{"001"};
constexpr const char* LABEL_INTERP_PLUGIN_NAME{"LABEL_INTERP_TRT"};
}

LabelInterpPlugin::LabelInterpPlugin(const int interpMode)
    : mInterpMode(interpMode)
{
    ASSERT(mInterpMode == 4); // cubic
}

LabelInterpPlugin::LabelInterpPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mInterpMode);
}

LabelInterpPlugin::~LabelInterpPlugin()
{
    terminate();
}

int LabelInterpPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs LabelInterpPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    ASSERT(outputIndex == 0 && nbInputs == 2);
    ASSERT(inputs[0].nbDims == 4 && inputs[1].nbDims == 4);
    nvinfer1::DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[0].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = inputs[1].d[2];
    output.d[3] = inputs[1].d[3];
    return output;
}

int LabelInterpPlugin::initialize()
{
    return 0;
}

void LabelInterpPlugin::terminate() {}

size_t LabelInterpPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const
{
    ASSERT(nbInputs == 2 && inputs[0].dims.nbDims == 4 && inputs[1].dims.nbDims == 4);

    size_t src_rows = inputs[0].dims.d[2];
    size_t dst_rows = inputs[1].dims.d[2];
    size_t src_cols = inputs[0].dims.d[3];
    size_t dst_cols = inputs[1].dims.d[3];

    size_t dst_area_size = dst_rows * dst_cols;
    size_t src_area_size = src_rows * src_cols;

    bool enlarge = dst_area_size > src_area_size;
    bool shrink = dst_area_size <= src_area_size;

    bool use_vector = (enlarge && (dst_area_size <= 500 * 500)) ||
                      (shrink && (dst_area_size <= 1000 * 1000));

    if (!use_vector) {
        int coef_size = 4;

        return dst_rows * coef_size * sizeof(float) +  //! dev_coef_row
               dst_rows * sizeof(int) +                //! dev_sr
               dst_cols * coef_size * sizeof(float) +  //! dev_coef_col
               dst_cols * sizeof(int);                 //! dev_sc
    }
    return 0;
}

int LabelInterpPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    ASSERT(inputDesc[0].dims.nbDims == 4 && inputDesc[1].dims.nbDims == 4);
    ASSERT(outputDesc[0].dims.nbDims == 4);
    ASSERT(inputDesc[1].dims.d[2] == outputDesc[0].dims.d[2] && inputDesc[1].dims.d[3] == outputDesc[0].dims.d[3]);

    int BATCH = inputDesc[0].dims.d[0];
    int C = inputDesc[0].dims.d[1];
    int IH = inputDesc[0].dims.d[2];
    int IW = inputDesc[0].dims.d[3];
    int OH = outputDesc[0].dims.d[2];
    int OW = outputDesc[0].dims.d[3];

    for (int batch = 0; batch < BATCH; ++batch) {
        for (int c = 0; c < C; ++c) {
            auto diff_in = batch * C * IH * IW + c * IH * IW;
            auto diff_out = batch * C * OH * OW + c * OH * OW;

            const float* src =
                    static_cast<const float* const>(inputs[0]) + diff_in;
            float* dst = static_cast<float*>(outputs[0]) + diff_out;
            resize_cubic(src, dst, IH, IW, OH, OW, IW, OW, workspace,
                         stream);
        }
    }
    return 0;
}

size_t LabelInterpPlugin::getSerializationSize() const
{
    return serialized_size(mInterpMode);
}

void LabelInterpPlugin::serialize(void* buffer) const
{
    serialize_value(&buffer, mInterpMode);
}

bool LabelInterpPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kNCHW
        && inOut[pos].type == inOut[0].type);
}

const char* LabelInterpPlugin::getPluginType() const
{
    return LABEL_INTERP_PLUGIN_NAME;
}

const char* LabelInterpPlugin::getPluginVersion() const
{
    return LABEL_INTERP_PLUGIN_VERSION;
}

void LabelInterpPlugin::destroy()
{
    delete this;
}

IPluginV2DynamicExt* LabelInterpPlugin::clone() const
{
    auto plugin = new LabelInterpPlugin{mInterpMode};
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Set plugin namespace
void LabelInterpPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mNamespace = pluginNamespace;
}

const char* LabelInterpPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

nvinfer1::DataType LabelInterpPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

void LabelInterpPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

void LabelInterpPlugin::detachFromContext() {}

void LabelInterpPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{

}

// =========================LabelInterpPluginCreator============================

PluginFieldCollection LabelInterpPluginCreator::mFC{};
std::vector<PluginField> LabelInterpPluginCreator::mPluginAttributes;

LabelInterpPluginCreator::LabelInterpPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LabelInterpPluginCreator::getPluginName() const
{
    return LABEL_INTERP_PLUGIN_NAME;
}

const char* LabelInterpPluginCreator::getPluginVersion() const
{
    return LABEL_INTERP_PLUGIN_VERSION;
}

const PluginFieldCollection* LabelInterpPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* LabelInterpPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(!strcmp(fields[0].name, "mode"));
    ASSERT(fields[0].type == PluginFieldType::kINT32);
    const int mode = *(static_cast<const int*>(fields[0].data));

    LabelInterpPlugin* obj = new LabelInterpPlugin(mode);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* LabelInterpPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    LabelInterpPlugin* obj = new LabelInterpPlugin{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
