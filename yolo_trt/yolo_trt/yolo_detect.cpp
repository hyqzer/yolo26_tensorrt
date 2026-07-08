#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>

#include "pch.h"

using namespace std;
using namespace cv;

// tensorrt日志
class TRTLogger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING)
			cout << "[TRT]" << msg << endl;
	}
}gLogger;

// 检查结构体
struct DetectBox {
	float x1, y1, x2, y2;
	float conf;
	int cls;
};

// 构建 TensorRT Engine
nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxPath, bool useFP16 = true)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "parse onnx failed!" << std::endl;
        return nullptr;
    }

    // Timing Cache 制造config，config包含优化策略、运行时间限制，精度限制等
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (useFP16 && builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // 制造序列化网络
    nvinfer1::IHostMemory* modelStream = builder->buildSerializedNetwork(*network, *config);

    // 反序列化
    // createInferRuntime 创建 Runtime（准备加载引擎） 用于唤醒已经编译好的引擎
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    // 把序列化的引擎数据（二进制）变成可用的引擎对象（反序列化）
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size());

    modelStream->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
    runtime->destroy();
    return engine;
}

// 保存 engine 到文件
void saveEngine(nvinfer1::ICudaEngine* engine, const std::string& path)
{
    nvinfer1::IHostMemory* mem = engine->serialize();
    std::ofstream f(path, std::ios::binary);
    f.write((char*)mem->data(), mem->size());
    f.close();
    mem->destroy();
}

// 从文件加载 engine
nvinfer1::ICudaEngine* loadEngine(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return nullptr;
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);
    nvinfer1::IRuntime* rt = nvinfer1::createInferRuntime(gLogger);
    auto engine = rt->deserializeCudaEngine(buf.data(), sz);
    rt->destroy();
    return engine;
}


// leterbox 
Mat letterbox(const Mat src, int target_w, int target_h, float& scale, Point2i& offset) {
	float h = src.rows;
	float w = src.cols;
	scale = min(target_w / w, target_h / h);

	float scale_w = w * scale;
	float scale_h = h * scale;

	float left = (target_w - scale_w) / 2;
	float top = (target_h - scale_h) / 2;   // （左上角坐标）

	offset.x = left;
	offset.y = top;

	Mat result = Mat::zeros(target_h, target_w, CV_8UC3);
	result.setTo(Scalar(114, 114, 114));

	Mat resizeImage;
	resize(src, resizeImage, Size(scale_w, scale_h));

	resizeImage.copyTo(result(Rect(left, top, scale_w, scale_h)));

	return result;
}



int yolo_infer() {
    const string onnx_path = "G:\\weight\\yolo26s.onnx";
    const string engine_path = "G:\weight\yolo26s.enginge";
    const int IMG_h = 640;
    const int IMG_w = 640;
    const float CONF_THRESH = 0.5;
    const float IOU_THRESH = 0.5;

    // 加载/构建 engine
    nvinfer1::ICudaEngine* engine = nullptr;
    ifstream f(engine_path);
    if (f.is_open()) {
        engine = loadEngine(engine_path);
        cout << "load enginge success" << endl;
    }
    else {
        cout << "build engine form onnx..." << endl;
        engine = buildEngineFromOnnx(onnx_path, true);
        saveEngine(engine, engine_path);
        cout << "build engine success, engine saved done" << endl;
    }
    if (!engine) return -1;

    // 推理
    nvinfer1::IExecutionContext* ctx = engine->createExecutionContext();

    // 获取输入尺寸
    int input_index = engine->getBindingIndex("images");
    int output_index = engine->getBindingIndex("output0");
    nvinfer1::Dims input_dims = engine->getBindingDimensions(input_index);
    nvinfer1::Dims output_dims = engine->getBindingDimensions(output_index);

    // 输入输出显存
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t input_size = 1 * 3 * IMG_h * IMG_w * sizeof(float);
    size_t output_size = 1 * output_dims.d[1] * output_dims.d[2] * sizeof(float);
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    void* bindings[] = { d_input, d_output };

    // 读取图片

    Mat img = imread("C:\\Users\\hyqer\\Desktop\\img.jpg");
    if (img.empty()) {
        cout << "image wrong" << endl;
        return -1;
    }
    Mat img_rgb;
    cvtColor(img, img_rgb, COLOR_BGR2RGB);

    // letterbox 处理
    float scale;
    Point2i offset;
    Mat letter = letterbox(img_rgb, IMG_w, IMG_h, scale, offset);
    vector<float> host_input(3 * IMG_h * IMG_w);

    // HWC ->CWH / 归一化
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < IMG_h; y++) {
            for (int x = 0; x < IMG_w; x++) {
                host_input[c * IMG_h * IMG_w + y * IMG_h + x] =
                    letter.at<Vec3d>(y, x)[c];
            }
        }
    }

    //// 方法2：直接使用 cv::dnn::blobFromImage（最简单）
    //cv::Mat blob = cv::dnn::blobFromImage(letter, 1.0 / 255.0,
    //    cv::Size(IMG_W, IMG_H),
    //    cv::Scalar(), true, false);
    //memcpy(host_input, blob.data, IMG_H * IMG_W * 3 * sizeof(float));

    cudaMemcpy(d_input, host_input.data(), input_size, cudaMemcpyHostToDevice);

    // 推理
    ctx->executeV2(bindings);

    // 拷贝到 CPU
    vector<float> host_output(output_size / sizeof(float));
    cudaMemcpy(host_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    // 解析
    //[1, 84, 8400]，d[2] = 8400
    int num_box = output_dims.d[2];
    int nc = output_dims.d[1] - 4;
    vector<DetectBox> boxes;
    for (int i = 0; i < num_box; i++) {
        float cx = host_output[0 * num_box + i];
        float cy = host_output[1 * num_box + i];
        float w = host_output[2 * num_box + i];
        float h = host_output[3 * num_box + i];

        float max_conf = 0;
        int cls_id = 0;
        for (int c = 0; c < nc; c++) {
            float conf = host_output[(4 + c) * num_box + i];
            if (conf > max_conf) {
                max_conf = conf;
                cls_id = c;
            }
        }
        if (max_conf < CONF_THRESH) continue;

        // 映射回到原坐标
        float x1 = cx - w / 2.f;
        float y1 = cy - h / 2.f;
        float x2 = cx + w / 2.f;
        float y2 = cy + h / 2.f;
        // 去除 letterbox 偏移缩放
        x1 = (x1 - offset.x) / scale;
        y1 = (y1 - offset.y) / scale;
        x2 = (x2 - offset.x) / scale;
        y2 = (y2 - offset.y) / scale;

        boxes.push_back({ x1, y1, x2, y2, max_conf, cls_id });

    }
    cout << boxes.size() << endl;

}
