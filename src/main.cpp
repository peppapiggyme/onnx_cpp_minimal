#include "OrtInferenceWrapper.hpp"

#include <iostream>

using namespace std;

int main()
{
    auto obj = OrtInferenceWrapper("res/model.onnx", "myOnnxEnv");

    obj.init();

    size_t input_tensor_size = 5;
    std::vector<float> input_tensor_values(input_tensor_size);

    // might be your loop tree
    for (size_t i = 0; i < 10; i++)
    {
        cout << i << endl;
        // initialize input data with values in [0.0, 1.0]
        for (size_t i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = (float)0.f + (float)rand() / (float)RAND_MAX * 0.25f;
        
        obj.run(input_tensor_values);
    }
    
    return 0;
}