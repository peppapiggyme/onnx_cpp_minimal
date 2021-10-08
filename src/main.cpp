#include "OrtInferenceWrapper.hpp"

#include <iostream>

using namespace std;

int main()
{
    // Instantiate the wrapper
    auto ortWrapper = OrtInferenceWrapper("res/model.onnx", "myOnnxEnv");

    // Initialisation
    ortWrapper.Init();

    // You should know your input-output model
    // here dim=5 features (mhh, mbb, mtautau, dRbb, dRtautau) and dim=2 output (bkg, sig)
    // input features must be normalised to [-1, 1] as defined in bbtautau package
    size_t input_tensor_size = 5;
    size_t output_tensor_size = 2;

    // dummy input vector
    std::vector<float> input_tensor_values(input_tensor_size);

    // mimic looping over a tree
    for (size_t i = 0; i < 10; i++)
    {
        printf("Event %lu , Input = [", i);
        // initialize input data with values in [0.0, 0.25] -> signal like
        for (size_t i = 0; i < input_tensor_size; i++)
        {
            input_tensor_values[i] = (float)0.f + (float)rand() / (float)RAND_MAX * 0.25f;
            printf("%.2f,", input_tensor_values[i]);
        }
        printf("]\n");
        
        // inference with onnx
        const auto& out = ortWrapper.GetOutputSimplified(input_tensor_values);
        for (size_t i = 0; i < output_tensor_size; i++)
            printf("Score for class [%s] =  %f\n", i == 0 ? "Bkg" : "Sig", out[i]);

    }
    
    exit(0);
}