#include "onnxruntime_cxx_api.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

using std::vector;
using std::cout;
using std::endl;

/* 
 * Testing c++ api usage from:
 * https://github.com/microsoft/onnxruntime/blob/1.1.0-test/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
 * 
 * Model trained by bbtautau ml
 */

class OrtInferenceWrapper
{
private:
    Ort::Env m_env;                                // onnxruntime environment
    Ort::Session m_session;                        // inference session
    Ort::AllocatorWithDefaultOptions m_allocator;  // default allocator, how/when to free?
    Ort::MemoryInfo m_meminfo;                     // meminfo
    size_t m_num_input_nodes;                      // number of input nodes
    size_t m_num_output_nodes;                     // number of output nodes
    std::vector<int64_t> m_input_node_dims;        // input vector dimensions
    std::vector<int64_t> m_output_node_dims;       // output vector dimensions
    std::vector<const char*> m_input_node_names;   // input node names
    std::vector<const char*> m_output_node_names;  // output node names
    
public:
    OrtInferenceWrapper(const char *model_path, const char *env_name="test")
        : m_env{ORT_LOGGING_LEVEL_WARNING, env_name}
        , m_session{m_env, model_path, Ort::SessionOptions{}}
        , m_allocator{}
        , m_meminfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)}
        , m_num_input_nodes{m_session.GetInputCount()}
        , m_num_output_nodes{m_session.GetOutputCount()}
        , m_input_node_names{m_num_input_nodes}
        , m_output_node_names{m_num_output_nodes}
    {
    }

    void Init(bool check=true)
    {
        if (check)
        {
            CheckInput();
            CheckOutput();
        }
    }

    vector<float> GetOutputSimplified(vector<float>& input_tensor_values) // simplified input shape
    {
        // create input tensor object from data values
        // 2 = dim(1, input_dim)
        // simplified input tensor (vector) and size
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(m_meminfo, 
            input_tensor_values.data(), (size_t)input_tensor_values.size(), m_input_node_dims.data(), 2);
        // assert(input_tensor.IsTensor());

        // score model & input tensor, get back output tensor
        auto output_tensors = m_session.Run(
            Ort::RunOptions{nullptr}, m_input_node_names.data(), &input_tensor, 1, m_output_node_names.data(), 1);
        // assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();

        // simplified output model
        size_t actual_output_node_dim = m_output_node_dims.back();
        vector<float> output(actual_output_node_dim);
        for (int i = 0; i < actual_output_node_dim; i++)
            output[i] = std::exp(floatarr[i]);
        
        return output;
    }

private:
    void CheckInput()
    {
        printf("Number of inputs  = %zu\n", m_num_input_nodes);
        // iterate over all input nodes
        for (int i = 0; i < m_num_input_nodes; i++) {
            // print input node names
            char* input_name = m_session.GetInputName(i, m_allocator);
            printf("Input %d : name=%s\n", i, input_name);
            m_input_node_names[i] = input_name;

            // print input node types
            Ort::TypeInfo type_info = m_session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Input %d : type=%d\n", i, type);

            // print input shapes/dims
            m_input_node_dims = tensor_info.GetShape();
            printf("Input %d : num_dims=%zu\n", i, m_input_node_dims.size());
            for (int j = 0; j < m_input_node_dims.size(); j++)
                printf("Input %d : dim %d=%jd\n", i, j, m_input_node_dims[j]);
        }
    }

    void CheckOutput()
    {
        printf("Number of outputs = %zu\n", m_num_output_nodes);
        // iterate over all output nodes
        for (int i = 0; i < m_num_output_nodes; i++) {
            // print output node names
            char* output_name = m_session.GetOutputName(i, m_allocator);
            printf("Output %d : name=%s\n", i, output_name);
            m_output_node_names[i] = output_name;

            // print output node types
            Ort::TypeInfo type_info = m_session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Output %d : type=%d\n", i, type);

            // print output shapes/dims
            m_output_node_dims = tensor_info.GetShape();
            printf("Output %d : num_dims=%zu\n", i, m_output_node_dims.size());
            for (int j = 0; j < m_output_node_dims.size(); j++)
                printf("Output %d : dim %d=%jd\n", i, j, m_output_node_dims[j]);
        }
    }
};
