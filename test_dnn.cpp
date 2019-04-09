//
// Created by binbin on 19-4-6.
//

#include "onnx.pb.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include "tkdnn.h"

using namespace std;

typedef struct LayerInfo{

    std::string layer_type;
    std::vector<std::string> layer_inputs;
    std::vector<std::string> layer_outputs;
    std::string layer_params;

};


int calculateTensorDims
        (
                const onnx::GraphProto& graph_proto,
                std::map<int, std::map<std::string, std::string>>& net,
                std::map<int, std::map<std::string, std::vector<int>>>& tensorDims
        )
{
    std::map<std::string, std::vector<int>> input_tensor_dim_map;

    //Inputs to the graph.
    for(int i=0; i < graph_proto.input_size(); i++) {
        const onnx::ValueInfoProto& value_info_proto = graph_proto.input(i);
        std::string layer_input = value_info_proto.name();
        std::vector<int> dims;

        const onnx::TypeProto& type_proto = value_info_proto.type();
        const onnx::TypeProto::Tensor& tensor = type_proto.tensor_type();
        const onnx::TensorShapeProto& tensor_shape = tensor.shape();

        for(int j=0; j < tensor_shape.dim_size(); j++) {
            const onnx::TensorShapeProto::Dimension& dimension = tensor_shape.dim(j);
            dims.push_back(dimension.dim_value());
        }

        input_tensor_dim_map[layer_input] = dims;
    }

    for(int i=0; i < net.size(); i++) {
        std::map<std::string, std::string> layer_details = net.find(i)->second;
        std::string layer_type = layer_details.find("type")->second;
        std::string layer_input = layer_details.find("input")->second;
        std::string layer_output = layer_details.find("output")->second;

        int in_w, in_h, in_c, in_n;
        int out_w, out_h, out_c, out_n;

        std::vector<int> output_dims;
        std::vector<int> input_dims;

           input_dims=input_tensor_dim_map.find(layer_input)->second;
           in_n=input_dims[0];
           in_c=input_dims[1];
           in_h=input_dims[2];
           in_w=input_dims[3];



        std::map<std::string, std::vector<int>> in_out_map;
        in_out_map[layer_input] = input_dims;

        if(layer_type == "Conv") {
            std::string layer_weights = " ";
            std::vector<int> weight_dims, bias_dims;
            if(layer_details.size() > 4) {
                layer_weights = layer_details.find("weights")->second;
                weight_dims = input_tensor_dim_map.find(layer_weights)->second;
            }
            std::string layer_bias = " ";
            if(layer_details.size() > 5) {
                layer_bias = layer_details.find("bias")->second;
                bias_dims = input_tensor_dim_map.find(layer_bias)->second;
            }
            std::string params = layer_details.find("params")->second;

            int kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h;
            std::stringstream ss(params);
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h;

            out_w = ((in_w + 2 * (pad_w) - kernel_w - (kernel_w - 1) * (dilation_w - 1))/stride_w) + 1;
            out_h = ((in_h + 2 * (pad_h) - kernel_h - (kernel_h - 1) * (dilation_h - 1))/stride_h) + 1;
            out_c = weight_dims[0];
            out_n = in_n;

            std::cout<<out_n<<" | "
                     <<out_c<<" | "
                     <<out_h<<" | "
                     <<out_w<<" | "<<std::endl;


            if(layer_details.size() > 4) {
                in_out_map[layer_weights] = weight_dims;
            }

            if(layer_details.size() > 5) {
                in_out_map[layer_bias] = bias_dims;
            }
        }
        else if(layer_type == "Relu") {
            out_w = in_w;
            out_h = in_h;
            out_c = in_c;
            out_n = in_n;
            std::cout<<out_n<<" | "
                     <<out_c<<" | "
                     <<out_h<<" | "
                     <<out_w<<" | "<<std::endl;
        }
        else if(layer_type == "LRN") {
            out_w = in_w;
            out_h = in_h;
            out_c = in_c;
            out_n = in_n;
            std::cout<<out_n<<" | "
                     <<out_c<<" | "
                     <<out_h<<" | "
                     <<out_w<<" | "<<std::endl;
        }
        else if(layer_type == "Dropout") {
            out_w = in_w;
            out_h = in_h;
            out_c = in_c;
            out_n = in_n;
            std::cout<<out_n<<" | "
                     <<out_c<<" | "
                     <<out_h<<" | "
                     <<out_w<<" | "<<std::endl;
        }
        else if(layer_type == "MaxPool") {
            std::string params = layer_details.find("params")->second;
            std::stringstream ss(params);
            int kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h;
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h;

            out_w = static_cast<int>(ceil( static_cast<float> (in_w + 2 * pad_w + stride_w - kernel_w)/stride_w));
            out_h = static_cast<int>(ceil( static_cast<float> (in_h + 2 * pad_h + stride_h - kernel_h)/stride_h));
            if(pad_h > 0) if((out_h-1) * stride_h >= (in_h + pad_h)) out_h = out_h - 1;
            if(pad_w > 0) if((out_w-1) * stride_w >= (in_w + pad_w)) out_w = out_w - 1;

            out_c = in_c;
            out_n = in_n;
            std::cout<<out_n<<" | "
                     <<out_c<<" | "
                     <<out_h<<" | "
                     <<out_w<<" | "<<std::endl;
        }
        else if(layer_type == "Gemm") {

            std::string layer_weights = " ";
            std::vector<int> weight_dims, bias_dims;
            std::vector<int> weight_dims_gemm;
            if(layer_details.size() > 4) {
                layer_weights = layer_details.find("weights")->second;
                weight_dims = input_tensor_dim_map.find(layer_weights)->second;
                weight_dims_gemm.push_back(in_w);
                weight_dims_gemm.push_back(in_h);
                weight_dims_gemm.push_back(in_c);
                weight_dims_gemm.push_back(weight_dims[0]);

            }
            std::string layer_bias = " ";
            if(layer_details.size() > 5) {
                layer_bias = layer_details.find("bias")->second;
                bias_dims = input_tensor_dim_map.find(layer_bias)->second;
            }

            out_n = 1;
            out_c = weight_dims[0];
            out_h = 1;
            out_w = 1;
            std::cout<<out_n<<" | "
                     <<out_c<<" | "
                     <<out_h<<" | "
                     <<out_w<<" | "<<std::endl;

            if(layer_details.size() > 4) {
                in_out_map[layer_weights] = weight_dims_gemm;
            }

            if(layer_details.size() > 5) {
                in_out_map[layer_bias] = bias_dims;
            }
        }

        output_dims.push_back(out_n);
        output_dims.push_back(out_c);
        output_dims.push_back(out_h);
        output_dims.push_back(out_w);
        input_tensor_dim_map[layer_output] = output_dims;
        in_out_map[layer_output] = output_dims;

        tensorDims[i] = in_out_map;
    }

    return 0;
}

static int get_tensor_proto_data_size(const onnx::TensorProto& tp)
{
    if (tp.has_raw_data())
    {
        const std::string& raw_data = tp.raw_data();
        int size = (int)raw_data.size() / 4;
        return size;
    }
    else if (tp.data_type() == 1)
    {
        return tp.float_data_size();
    }

    return 0;
}

const float *get_data_tensor_proto(const  onnx::TensorProto& tensor_proto){

    assert(!tensor_proto.raw_data().empty() || !tensor_proto.float_data().empty() ||
           !tensor_proto.double_data().empty() || !tensor_proto.int64_data().empty());

    const  float *data=NULL;

    if (tensor_proto.data_type() == onnx::TensorProto_DataType_FLOAT) {

        int size = get_tensor_proto_data_size(tensor_proto);

        if(tensor_proto.has_raw_data())
        {
            data = (float*)(tensor_proto.raw_data().c_str());
            //for(int k=0; k<size; ++k)
            //std::cout<<data[k]<<" ";
        }
        else{
            data = tensor_proto.float_data().data();
        }
    }

    else if (tensor_proto.data_type() == onnx::TensorProto_DataType_DOUBLE){

    }

    else if (tensor_proto.data_type() == onnx::TensorProto_DataType_FLOAT16){

    }

    else if (tensor_proto.data_type() == onnx::TensorProto_DataType_INT8){

    }

    else if (tensor_proto.data_type() == onnx::TensorProto_DataType_BOOL){

    }

    else{
        std::cout<<"Unsupported data type: " +tensor_proto.data_type()<<std::endl;
    }

    return data;
}

void get_graph_tensors(const onnx::GraphProto& graph_proto, std::map<std::string, const float*>& weights, vector<int> &out_channels)
{
    for (int j=0; j<graph_proto.initializer_size(); j++)
    {
        const onnx::TensorProto& tensor_proto = graph_proto.initializer(j);
        std::cout<<" tensor name = "<<tensor_proto.name()<<std::endl;

        int size = get_tensor_proto_data_size(tensor_proto);

        const float *data = get_data_tensor_proto(tensor_proto);

        weights[tensor_proto.name()] = data;
    }
};

void get_layer_params(const onnx::NodeProto& node_proto)
{
    std::cout<<node_proto.op_type()<<std::endl;
    std::cout<<node_proto.output_size()<<std::endl;
    std::cout<<node_proto.output(0)<<std::endl;

    for(int i = 0; i < node_proto.attribute_size(); i++)
    {
        onnx::AttributeProto attribute_proto = node_proto.attribute(i);
        std::string attribute_name = attribute_proto.name();
        std::cout<<"attribute name = "<<attribute_name<<std::endl;


        if(attribute_name == "kernel_shape")
        {
            assert(attribute_proto.ints_size() == 2);

            {

            }
        }
    }
}

void formatFileName(std::string& str, const std::string& from, const std::string& to)
{
    //Written to avoid conflicts with file creation with filenames that contain "/"
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}
int dumpOnnxModel(const onnx::GraphProto& graph_proto, std::map<int, std::map<std::string, std::string>>& net)
{
    for(int i=0; i < graph_proto.initializer_size(); i++)
    {
        const onnx::TensorProto& tensor_proto = graph_proto.initializer(i);
        std::string layer_name = tensor_proto.name().substr(0,tensor_proto.name().find_first_of("_"));
        std::cout<<"layer name = "<<layer_name<<std::endl;
        //net.find(layer_name);
        int tensor_size  = 1;

        for(int j = 0; j < tensor_proto.dims_size(); j++) {

            tensor_size *= tensor_proto.dims(j);
        }
        //std::cout<<std::endl;

        std::string weight_file_name = tensor_proto.name();
        //std::cout<<"weight_file_name = "<<weight_file_name<<std::endl;
        formatFileName(weight_file_name, "/", "_");
        std::string fileName_weights = "../weights/" + weight_file_name + ".f32";
        std::cout<<"data type = "<<tensor_proto.data_type()<<std::endl;
        if(tensor_proto.data_type() == onnx::TensorProto_DataType_FLOAT) {

            FILE * fs;
            fs = fopen(fileName_weights.c_str(), "wb");
            if(!fs) {
                std::cout << "ERROR: Unable to create a file, make sure weights folder is writable." << std::endl;
                return -1;
            }
            std::string raw_data_val = tensor_proto.raw_data();
            const char * val = raw_data_val.c_str();

            int count = 0;
            for(int k = 0; k < tensor_size*4 - 4; k+=4) {
                float weight;
                char b[] = {val[k], val[k+1], val[k+2], val[k+3]};
                memcpy(&weight, &b, sizeof(float));
                fwrite(&weight, sizeof(float), 1, fs);
                count++;
            }

            fclose(fs);
            std::cout << "INFO: Weights dumped for: " << tensor_proto.name() <<  std::endl;
        }

        else if(tensor_proto.data_type() == onnx::TensorProto_DataType_INT64){
              if(tensor_size <= 1)
              {
                  return -1;
              } else{

                  std::cout<<"data type = INT64"<<std::endl;
              }
        }
        else {
            std::cout <<"ERROR: Unsupported data types will be supported in future." << std::endl;
            return -1;
        }
    }

    return 0;
}

int getLayerParams(const onnx::NodeProto& node_proto, std::string& params)
{
    std::string layer_type = node_proto.op_type();

    std::cout<<"layer type = "<<layer_type<<std::endl;
    if(layer_type == "Conv") { //convolution

        int pad_h, pad_w;
        int stride_h, stride_w;
        int kernel_h, kernel_w;
        int dilation_h = 1, dilation_w = 1;

        for(int i =0; i < node_proto.attribute_size() ; i++) {
            const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
            if(attribute_name == "strides") {
                stride_h = attribute_proto.ints(0);
                stride_w = attribute_proto.ints(1);
            }
            else if(attribute_name == "pads") {
                pad_h = attribute_proto.ints(0);
                pad_w = attribute_proto.ints(1);
            }
            else if(attribute_name == "kernel_shape") {
                kernel_h = attribute_proto.ints(0);
                kernel_w = attribute_proto.ints(1);
            }
        }

        params = std::to_string(kernel_w)
                 + " " + std::to_string(kernel_h)
                 + " " + std::to_string(stride_w)
                 + " " + std::to_string(stride_h)
                 + " " + std::to_string(pad_w)
                 + " " + std::to_string(pad_h)
                 + " " + std::to_string(dilation_w)
                 + " " + std::to_string(dilation_h);

    }
    else if(layer_type == "MaxPool") { //maxpool

        int pad_h, pad_w;
        int stride_h, stride_w;
        int kernel_h, kernel_w;

        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
            if(attribute_name == "strides") {
                stride_h = attribute_proto.ints(0);
                stride_w = attribute_proto.ints(1);
            }
            else if(attribute_name == "pads") {
                pad_h = attribute_proto.ints(0);
                pad_w = attribute_proto.ints(1);
            }
            else if(attribute_name == "kernel_shape") {
                kernel_h = attribute_proto.ints(0);
                kernel_w = attribute_proto.ints(1);
            }

        }

        params = std::to_string(kernel_w)
                 + " " + std::to_string(kernel_h)
                 + " " + std::to_string(stride_w)
                 + " " + std::to_string(stride_h)
                 + " " + std::to_string(pad_w)
                 + " " + std::to_string(pad_h);

    }
    else if(layer_type == "GlobalAveragePool"){
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }

    }
    else if(layer_type == "LRN") { //lrn

        int lrn_local_size;
        float alpha, beta, bias;

        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto& attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
            if(attribute_name == "size") {
                lrn_local_size = attribute_proto.i();
            }
            else if(attribute_name == "alpha") {
                alpha = attribute_proto.ints(0);
            }
            else if(attribute_name == "beta") {
                beta = attribute_proto.ints(0);
            }
            else if(attribute_name == "bias") {
                bias = attribute_proto.ints(0);
            }
        }

        params = std::to_string(lrn_local_size)
                 + " " + std::to_string(alpha)
                 + " " + std::to_string(beta)
                 + " " + std::to_string(bias);

    }

    else if(layer_type == "Concat") { //Concatenation
        int channel_axis;
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;

            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }

            std::cout<<attribute_proto.f()<<std::endl;

            if(attribute_name == "axis")
            {
                channel_axis = 1;//As cudnn/TENSORRT do concat in channel dimension.[NCHW]
            }
            params = std::to_string(channel_axis);
        }
    }

    else if(layer_type == "LeakyRelu") { //prelu
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }

    else if(layer_type == "BatchNormalization")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }

    }
    else if(layer_type == "ConvTranspose")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }

        }
    }

    else if(layer_type == "Flatten")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }

    else if(layer_type == "Gemm")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "MaxRoiPool")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.f()<< std::endl;
        }
    }
    else if(layer_type == "PRelu")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Pad")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.f()<< std::endl;
        }
    }
    else if(layer_type == "Reshape")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Slice")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Split")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Squeeze")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Unsqueeze")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Transpose")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "Upsample")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<","<< attribute_proto.ints_size()<< std::endl;
            for(int k=0; k<attribute_proto.ints_size();++k)
            {
                std::cout<<attribute_proto.ints(k)<<std::endl;
            }
        }
    }
    else if(layer_type == "LSTM")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<std::endl;
        }
    }
    else if(layer_type == "GRU")
    {
        for(int i=0; i < node_proto.attribute_size(); i++) {
            const onnx::AttributeProto &attribute_proto = node_proto.attribute(i);
            std::string attribute_name = attribute_proto.name();
            std::cout<<"attribute name = "<<attribute_name<<std::endl;
        }
    }

    return 0;
}

int parseOnnxGraph(const onnx::GraphProto& graph_proto, std::map<int, std::map<std::string, std::string> >& net)
{
    if(graph_proto.has_name()) {
        std::cout << "INFO: Extracting the weights for : " << graph_proto.name() << std::endl;
    }

    std::cout << "INFO: Extracting the network structure for : " << graph_proto.name() << std::endl;
    std::cout<<"graph_proto.node_size() = "<<graph_proto.node_size()<<std::endl;
    for(int i=0; i < graph_proto.node_size(); i++) {
        const onnx::NodeProto node_proto = graph_proto.node(i);
        std::string params;
        getLayerParams(node_proto, params);

        std::map<std::string, std::string> layer_details;

        std::string layer_type = node_proto.op_type();
        std::string layer_input = node_proto.input(0);

        std::string layer_output = node_proto.output(0);

        layer_details["type"] = layer_type;
        layer_details["input"] = layer_input;
        layer_details["output"] = layer_output;
        layer_details["params"] = params;

        std::cout<<"layer_input = "<<layer_input<<std::endl;
        //std::cout<<"lay params = "<<params<<std::endl;
        //std::cout<<"layer_output = "<<layer_output<<std::endl;

        if(node_proto.input_size() > 1) {
            std::string layer_weights = node_proto.input(1);
            layer_details["weights"] = layer_weights;
        }

        if(node_proto.input_size() > 2) {
            std::string layer_bias = node_proto.input(2);
            layer_details["bias"] = layer_bias;
        }

        net[i] = layer_details;
        //if(layer_type == "Conv")
        //    net[layer_input] = layer_details;
    }

    std::cout<<"graph_proto.initializer_size() = "<<graph_proto.initializer_size()<<std::endl;
    if(dumpOnnxModel(graph_proto, net) < 0) {
        std::cout << "ERROR: Unable to dump weights from onnx model. " << std::endl;
        return -1;
    }
    else {
        std::cout << "RESULT: Weights and bias extraction successful" << std::endl;
    }

    return 0;
}

int loadOnnxModelFile(const onnx::GraphProto& graph_proto, const char * fileName, std::map<int, std::map<std::string, std::string> >& net)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    if(parseOnnxGraph(graph_proto, net) < 0) {
        std::cout << "ERROR: Unable to parse ONNX model." << std::endl;
        return -1;
    } else{
        return 1;
    }

}

int writeGDF
        (
                std::ofstream& ofsGDF,
                std::map<int, std::map<std::string, std::string>> net,
                std::map<int, std::map<std::string, std::vector<int>>> tensorDims
        )
{
    std::cout << "INFO: Writing the GDF " << std::endl;
    ofsGDF << "import vx_nn" << std::endl;
    ofsGDF << std::endl;

    //network input.
    std::map<std::string, std::string> first_layer = net.find(0)->second;
    std::map<std::string, std::vector<int>> first_layer_dims = tensorDims.find(0)->second;
    auto&& layer_input = first_layer.find("input")->second;
    auto& input_dims = first_layer_dims.find(layer_input)->second;
    formatFileName(layer_input, "/", "_");
    ofsGDF << "data layer " << layer_input << " = tensor:4,{" << input_dims[3] << "," << input_dims[2] << "," << input_dims[1] << "," << input_dims[0] << "},"
           << "VX_TYPE_FLOAT32,0" << std::endl;
    ofsGDF << "read " << layer_input << " input.f32" << std::endl;

    for(int i=0; i < net.size(); i++) {
        std::map<std::string, std::string> layer_details = net.find(i)->second;
        std::map<std::string, std::vector<int>> in_out_map = tensorDims.find(i)->second;

        auto&& layer_type = layer_details.find("type")->second;
        auto&& layer_output = layer_details.find("output")->second;
        auto&& layer_input = layer_details.find("input")->second;

        //output dims.
        auto& output_dims = in_out_map.find(layer_output)->second;
        formatFileName(layer_output, "/", "_");
        ofsGDF << "data " << layer_output << " = tensor:4,{" << output_dims[3] << "," << output_dims[2] << "," << output_dims[1] << "," << output_dims[0] << "},"
               << "VX_TYPE_FLOAT32,0" << std::endl;

        //TODO: Generate dims of layers and create nodes.
        if(layer_type == "Conv") {
            auto&& layer_params = layer_details.find("params")->second;
            int kernel_w, kernel_h, pad_w, pad_h, dilation_w, dilation_h, stride_w, stride_h;
            std::stringstream ss(layer_params);
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h;
            auto&& layer_weights = layer_details.find("weights")->second;
            auto& weight_dims = in_out_map.find(layer_weights)->second;

            if(layer_details.size() > 4) {
                formatFileName(layer_weights, "/", "_");
                ofsGDF << "data " << layer_weights << " = tensor:4,{" << weight_dims[3] << "," << weight_dims[2] << "," << weight_dims[1] << ","
                       << weight_dims[0] << "}," << "VX_TYPE_FLOAT32,0" << std::endl;
                ofsGDF << "init " << layer_weights << " weights/" << layer_weights << ".f32" << std::endl;
            }

            std::string layer_bias;
            if(layer_details.size() > 5) {
                layer_bias = layer_details.find("bias")->second;
                auto& bias_dims = in_out_map.find(layer_bias)->second;
                formatFileName(layer_bias, "/", "_");
                ofsGDF << "data " << layer_bias << " = tensor:1,{" << bias_dims[0] << "},VX_TYPE_FLOAT32,0" << std::endl;
                ofsGDF << "init " << layer_bias << " weights/" << layer_bias << ".f32" << std::endl;
            }
            else if(layer_details.size() == 5) {
                layer_bias = layer_output + "_b";
                ofsGDF << "data " << layer_bias << " = tensor:1,{" << weight_dims[0] << "},VX_TYPE_FLOAT32,0" << std::endl;
            }

            //conv params.
            ofsGDF << "data " << layer_output << "_params =" << " scalar:VX_TYPE_NN_CONV_PARAMS,{" << pad_w << "," << pad_h << ","
                   << "VX_CONVERT_POLICY_SATURATE" << "," << "VX_ROUND_POLICY_TO_NEAREST_EVEN" << ",VX_NN_DS_SIZE_ROUNDING_FLOOR,"
                   << dilation_w - 1 << "," << dilation_h - 1 << "}" << std::endl;

            //conv node.
            ofsGDF << "node org.khronos.nn_extension.convolution_layer " << layer_input << " " << layer_weights << " "
                   << layer_bias << " " << layer_output << "_params" << " " << layer_output << std::endl;

        }
        else if(layer_type == "Relu") {
            ofsGDF << "data " << layer_output << "_mode = " << "scalar:VX_TYPE_ENUM,VX_NN_ACTIVATION_RELU" << std::endl;
            ofsGDF << "data " << layer_output << "_param_a = " << "scalar:VX_TYPE_FLOAT32,0" << std::endl;
            ofsGDF << "data " << layer_output << "_param_b = " << "scalar:VX_TYPE_FLOAT32,0" << std::endl;
            ofsGDF << "node org.khronos.nn_extension.activation_layer " << layer_input << " " << layer_output << "_mode" << " "
                   << layer_output << "_param_a" << " " << layer_output << "_param_b" << " " << layer_output << std::endl;
        }
        else if(layer_type == "MaxPool") {
            auto&& layer_params = layer_details.find("params")->second;
            std::stringstream ss(layer_params);
            int kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h;
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h;

            ofsGDF << "data " << layer_output << "_type = " << "scalar:VX_TYPE_ENUM,VX_NN_POOLING_MAX" << std::endl;
            ofsGDF << "data " << layer_output << "_kernel_w = " << "scalar:VX_TYPE_SIZE," << kernel_w << std::endl;
            ofsGDF << "data " << layer_output << "_kernel_h = " << "scalar:VX_TYPE_SIZE," << kernel_h << std::endl;
            ofsGDF << "data " << layer_output << "_pad_w = " << "scalar:VX_TYPE_SIZE," << pad_w << std::endl;
            ofsGDF << "data " << layer_output << "_pad_h = " << "scalar:VX_TYPE_SIZE," << pad_h << std::endl;
            ofsGDF << "data " << layer_output << "_roundPolicy = " << "scalar:VX_TYPE_ENUM,VX_ROUND_POLICY_TO_NEAREST_EVEN" << std::endl;
            ofsGDF << "node " << "org.khronos.nn_extension.pooling_layer " << layer_input << " "
                   << layer_output << "_type " << layer_output << "_kernel_w " << layer_output << "_kernel_h "
                   << layer_output << "_pad_w " << layer_output << "_pad_h "
                   << layer_output << "_roundPolicy"
                   << " " << layer_output
                   << std::endl;
        }
        else if(layer_type == "LRN") {
            auto&& layer_params = layer_details.find("params")->second;
            std::stringstream ss(layer_params);
            int lrn_local_size;
            float alpha, beta, bias;
            ss >> lrn_local_size >> alpha >> beta >> bias;

            ofsGDF << "data " << layer_output << "_mode = " << "scalar:VX_TYPE_ENUM,VX_NN_NORMALIZATION_ACROSS_MAPS" << std::endl;
            ofsGDF << "data " << layer_output << "_size = " << "scalar:VX_TYPE_SIZE," << lrn_local_size << std::endl;
            ofsGDF << "data " << layer_output << "_alpha = " << "scalar:VX_TYPE_FLOAT32," << alpha << std::endl;
            ofsGDF << "data " << layer_output << "_beta = " << "scalar:VX_TYPE_FLOAT32," << beta << std::endl;
            ofsGDF << "data " << layer_output << "_bias = " << "scalar:VX_TYPE_FLOAT32," << bias << std::endl;
            ofsGDF << "node org.khronos.nn_extension.normalization_layer " << layer_input << " "
                   << layer_output << "_mode "
                   << layer_output << "_size "
                   << layer_output << "_alpha "
                   << layer_output << "_beta "
                   << layer_output << "_bias "
                   << std::endl;
        }
        else if(layer_type == "Gemm") {
            int stride_w, stride_h, pad_w, pad_h,dilation_w, dilation_h;
            stride_w = 1; stride_h = 1; pad_w = 0; pad_h = 0; dilation_w = 1; dilation_h = 1;

            auto&& layer_weights = layer_details.find("weights")->second;
            auto& weight_dims = in_out_map.find(layer_weights)->second;

            if(layer_details.size() > 4) {
                formatFileName(layer_weights, "/", "_");
                ofsGDF << "data " << layer_weights << " =tensor:4,{" << weight_dims[0] << "," << weight_dims[1] << "," << weight_dims[2]
                       << "," << weight_dims[3] << "}," << "VX_TYPE_FLOAT32,0" << std::endl;
                ofsGDF << "init " << layer_weights << " weights/" << layer_weights << ".f32" << std::endl;
            }

            std::string layer_bias;
            if(layer_details.size() > 5) {
                layer_bias = layer_details.find("bias")->second;
                std::vector<int> bias_dims = in_out_map.find(layer_bias)->second;
                formatFileName(layer_bias, "/", "_");
                ofsGDF << "data " << layer_bias << " = tensor:1,{" << bias_dims[0] << "}, VX_TYPE_FLOAT32,0" << std::endl;
                ofsGDF << "init " << layer_bias << " weights/" << layer_bias << ".f32" << std::endl;
            }
            else if(layer_details.size() == 5) {
                layer_bias = layer_output + "_b";
                ofsGDF << "data " << layer_bias << " = tensor:1,{" << weight_dims[3] << "},VX_TYPE_FLOAT32,0" << std::endl;
            }

            ofsGDF << "data " << layer_output << "_params = " << "scalar:VX_TYPE_NN_CONV_PARAMS,{" << pad_w << "," << pad_h << ","
                   << "VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,VX_NN_DS_SIZE_ROUNDING_FLOOR,0,0}" << std::endl;

            ofsGDF << "node org.khronos.nn_extension.convolution_layer " << layer_input << " " << layer_weights << " " << layer_bias
                   << " " << layer_output << "_params" << " " << layer_output << std::endl;
        }
        else if(layer_type == "Dropout") {
            ofsGDF << "node org.khronos.openvx.copy " << layer_input << " " << layer_output << std::endl;
        }
        else if(layer_type == "Softmax") {
            ofsGDF << "node org.khronos.nn_extension.softmax_layer " << layer_input << " " << layer_output << std::endl;
        }

        if(i == net.size() - 1) {
            ofsGDF << "write " << layer_output << " output.f32" << std::endl;
        }

        ofsGDF << std::endl;
    }

    return 0;
}

int generate_tkdnn(
        std::ofstream& ofDNN,
        const onnx::GraphProto& graph_proto,
        std::map<int, std::map<std::string, std::string>>& net,
        std::map<int, std::map<std::string, std::vector<int>>>& tensorDims) {
    //add header file
    ofDNN << "#include <iostream>\n"
          << "#include \"tkdnn.h\"\n" << std::endl;

    //add weights paths
    for (int i = 0; i < graph_proto.initializer_size(); i++) {
        const onnx::TensorProto &tensor_proto = graph_proto.initializer(i);
        std::string weight_file_name = tensor_proto.name();
        formatFileName(weight_file_name, "/", "_");
        ofDNN << "std::string " << weight_file_name << " = \"../weights/" << weight_file_name << ".fp32\";"
              << std::endl;
    }

    //add main function
    ofDNN << "void main(){\n";

    ofDNN << " //network input.\n";

    std::map<std::string, std::string> first_layer = net.find(0)->second;
    std::map<std::string, std::vector<int>> first_layer_dims = tensorDims.find(0)->second;
    auto &&layer_input = first_layer.find("input")->second;
    auto &input_dims = first_layer_dims.find(layer_input)->second;
    //formatFileName(layer_input, "/", "_");
    //ofsGDF << "data layer " << layer_input << " = tensor:4,{" << input_dims[3] << "," << input_dims[2] << "," << input_dims[1] << "," << input_dims[0] << "},"
    ofDNN << " tk::dnn::dataDim_t dim(" << input_dims[3] << "," << input_dims[2] << "," << input_dims[1] << "," << input_dims[0] << ");\n";
    ofDNN << " tk::dnn::Network net(dim);\n";
    for (int i = 0; i < net.size(); i++) {
        std::map<std::string, std::string> layer_details = net.find(i)->second;
        std::map<std::string, std::vector<int>> in_out_map = tensorDims.find(i)->second;

        auto &&layer_type = layer_details.find("type")->second;
        auto &&layer_output = layer_details.find("output")->second;
        auto &&layer_input = layer_details.find("input")->second;

        //output dims.
        auto &output_dims = in_out_map.find(layer_output)->second;
        //formatFileName(layer_output, "/", "_");
        //std::cout<< "data " << layer_output << " = tensor:4,{" << output_dims[3] << "," << output_dims[2] << "," << output_dims[1] << "," << output_dims[0] << "},"
        //       << "VX_TYPE_FLOAT32,0" << std::endl;

        //TODO: Generate dims of layers and create nodes.
        if (layer_type == "Conv") {
            auto &&layer_params = layer_details.find("params")->second;
            int kernel_w, kernel_h, pad_w, pad_h, dilation_w, dilation_h, stride_w, stride_h;
            std::stringstream ss(layer_params);
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h >> dilation_w >> dilation_h;
            auto&& layer_weights = layer_details.find("weights")->second;
            std::string layer_bias;
            if(layer_details.size() > 5) {
                layer_bias = layer_details.find("bias")->second;
            }
            else if(layer_details.size() == 5) {
                layer_bias = layer_output + "_b";
            }

            formatFileName(layer_output, "/", "_");
            formatFileName(layer_weights, "/", "_");
            formatFileName(layer_bias, "/", "_");
            ofDNN << " tk::dnn::Conv2d " << layer_output << "(&net,"
                  << output_dims[1]<< ", "
                  << kernel_h << ", " << kernel_w << ", "
                  << stride_h<<", "<< stride_w<<", "
                  << pad_h<<", "<<pad_w<<", "
                  << layer_weights<<", " <<layer_bias<<", "
                  << "false)"<< ";\n";
        }
        else if(layer_type == "Relu") {

            formatFileName(layer_output, "/", "_");
            ofDNN << " tk::dnn::Activation "<<layer_output<<"(&net, tk::dnn::ACTIVATION_LEAKY);\n";
        }
        else if(layer_type == "MaxPool") {
            auto&& layer_params = layer_details.find("params")->second;
            std::stringstream ss(layer_params);
            int kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h;
            ss >> kernel_w >> kernel_h >> stride_w >> stride_h >> pad_w >> pad_h;

            formatFileName(layer_output, "/", "_");
            ofDNN << " tk::dnn::Pooling " << layer_output << "(&net,"
                  << kernel_h << ", " << kernel_w << ", "
                  << stride_h<<", "<< stride_w<<", "
                  //<< pad_h<<", "<<pad_w<<", "
                  <<"tk::dnn::POOLING_MAX)"<< ";\n";
        }
        else if(layer_type == "GlobalAveragePool")
        {
            if(i >= 1) {
                std::map<std::string, std::string> pre_layer_details = net.find(i -1)->second;
                std::map<std::string, std::vector<int>> pre_out_map = tensorDims.find(i - 1)->second;

                auto &&pre_layer_output = pre_layer_details.find("output")->second;

                //output dims.
                auto &pre_output_dims = pre_out_map.find(pre_layer_output)->second;
                std::cout<< "pre_layer_output " << pre_layer_output << " = tensor:4,{" << pre_output_dims[3] << "," << pre_output_dims[2] << "," << pre_output_dims[1] << "," << pre_output_dims[0] << "},"<<std::endl;

                formatFileName(layer_output, "/", "_");
                ofDNN << " tk::dnn::Pooling " << layer_output << "(&net,"
                      << pre_output_dims[3] << ", " << pre_output_dims[2] << ", "
                      << 1 << ", " << 1 << ", "
                      << "tk::dnn::POOLING_AVERAGE)" << ";\n";
            }
        }
        else if(layer_type == "Concat"){
            formatFileName(layer_output, "/", "_");
            ofDNN << " tk::dnn::Layer *"<<layer_output<<"["<<layer_input.size()<<"]"<< "={ ";
            for(int k=0; k<layer_input.size()-1;++k)
            {
                ofDNN <<"&"<< layer_input[k]<<",";
            }
            ofDNN <<"&"<< layer_input[layer_input.size() -1]<<"};"<<std::endl;

            ofDNN << " tk::dnn::Concat" << layer_output << "(&net,"<<layer_output<<","<<layer_input.size()<<")\n";
        }
        else if(layer_type == "Softmax"){

            formatFileName(layer_output, "/", "_");
            ofDNN << " tk::dnn::Softmax " << layer_output << "(&net)"<< ";\n";
        }

    }

    ofDNN<<"}\n";
}

int main(int argc, char** argv) {

    onnx::ModelProto model;

    std::string model_path = argv[1]; //"../squeezenet.onnx";
    std::ifstream in(model_path, std::ios_base::binary);
    model.ParseFromIstream(&in);
    in.close();
    //std::cout<<model.graph().input().size()<<"\n";
    std::cout<<model.graph().name()<<std::endl;
    //std::cout<<model.graph().node().size() <<std::endl;

    const onnx::GraphProto graph_proto  = model.graph();
    /*int node_count = graph_proto .node_size();
    int initializer_count = graph_proto .initializer_size();

    std::map<std::string, const float*> weights;
    std::vector<int> out_channels;

    get_graph_tensors(graph_proto, weights, out_channels);

    int layersSize = graph_proto.node_size();

    tk::dnn::dataDim_t dim(1, 3, 416, 416, 1);
    tk::dnn::Network net(dim);

    for(int li = 0; li < layersSize; li++){

        const onnx::NodeProto& node_proto = model.graph().node(li);

        get_layer_params(node_proto);

    }

    std::cout<<"graph_proto.input_size()="<<graph_proto.input_size()<<std::endl;
    for(int i=0; i < graph_proto.input_size(); i++) {
        const onnx::ValueInfoProto& value_info_proto = graph_proto.input(i);
        std::string layer_input = value_info_proto.name();

        std::cout<<"layer_input = "<<layer_input<<std::endl;

        std::vector<int> dims;

        const onnx::TypeProto& type_proto = value_info_proto.type();
        const onnx::TypeProto::Tensor& tensor = type_proto.tensor_type();
        const onnx::TensorShapeProto& tensor_shape = tensor.shape();
        std::cout<<"tensor_shape = "<<tensor_shape.dim_size() <<std::endl;

        for(int j=0; j < tensor_shape.dim_size(); j++) {
            const onnx::TensorShapeProto::Dimension& dimension = tensor_shape.dim(j);
            std::cout<<"dimension.dim_value() = "<<dimension.dim_value() <<std::endl;
            dims.push_back(dimension.dim_value());
        }

        //input_tensor_dim_map[layer_input] = dims;
    }
    */
    std::map<int, std::map<std::string, std::string> > net;

    if(loadOnnxModelFile(graph_proto, model_path.c_str(), net) < 0) {
        return -1;
    }
    else {
        std::cout << "INFO: Network structure is extracted successfully." << std::endl;
    }
    //for(int k=0; k<16; ++k)
    //    std::cout<<weights["fire3/squeeze1x1_b_0"][k]<<std::endl;

    //calculate tensor dimensions.
    std::map<int, std::map<std::string, std::vector<int>>> tensorDims;
    if(calculateTensorDims(graph_proto, net, tensorDims) < 0) {
        std::cout << "ERROR: Unable to calculate tensor dims" << std::endl;
    }
    else {
        std::cout << "INFO: Tensor Dim calculation successful" << std::endl;
    }

    //write gdf
    //std::ofstream ofsGDF("../net.gdf", std::ios::binary);
    //if(writeGDF(ofsGDF, net, tensorDims) < 0) {
    //    std::cout << "ERROR: Unable to write into GDF file" << std::endl;
    //}
    //else {
    //    std::cout << "RESULT: GDF Generation is successful." << std::endl;
    //}

    //generate tkdnn files
    std::ofstream ofDNN("../main.cc", std::ios::binary);
    generate_tkdnn(ofDNN, graph_proto, net, tensorDims);

}
