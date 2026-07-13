#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#ifdef _WIN32
	#ifdef YOLOTRT_EXPORTS  // 该宏名在项目属性里定义
		#define YOLO_API __declspec(dllexport)
	#else
		#define YOLO_API __declspec(dllimport)
	#endif
#else
	#define YOLO_API 
#endif

#ifdef __cplusplus
extern "C" {
#endif
	YOLO_API int yolo_infer(
		const char* IMG_PATH,
		const char* onnx_path,
		const char* engine_path,
		int IMG_h = 640,
		int IMG_w = 640,
		float CONF_THRESH = 0.5,
		float IOU_THRESH = 0.5
	);
#ifdef __cplusplus
}
#endif
		
