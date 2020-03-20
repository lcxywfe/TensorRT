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
#ifndef TRT_CAFFE_RESIZE_PLUGIN_H
#define TRT_CAFFE_RESIZE_PLUGIN_H
#include "plugin.h"
#include "serialize.hpp"
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
class LabelInterpPlugin final : public nvinfer1::IPluginV2DynamicExt
{

public:
    LabelInterpPlugin(const int interpMode);
    LabelInterpPlugin(void const* serialData, size_t serialLength);

    LabelInterpPlugin() = delete;

    ~LabelInterpPlugin() override;

    int getNbOutputs() const override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(
        int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2DynamicExt* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) override;

    void detachFromContext() override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

private:
    int mInterpMode;
    std::string mNamespace;
};

class LabelInterpPluginCreator : public BaseCreator
{
public:
    LabelInterpPluginCreator();

    ~LabelInterpPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_CAFFE_RESIZE_PLUGIN_H
