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
    return 0;
}

int LabelInterpPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

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
    for (int i = 0; i < nbInputs; i++)
    {
        for (int j = 0; j < in[i].desc.dims.nbDims; j++)
        {
            // Do not support dynamic dimensions
            ASSERT(in[i].desc.dims.d[j] != -1);
        }
    }
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
