#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <openslide/openslide.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "tile_slide.h"
#include <iostream>
#include <stdlib.h>
#include <memory>
#include <future>
#include <ctime>
#include <chrono>
#include <string>

std::string StringReplace(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

at::Tensor compute_final_class(torch::jit::script::Module classifier, at::Tensor input) {
  at::Tensor output = classifier.forward({input.to(at::kCUDA)}).toTensor();
  return output;
}

at::Tensor feature_postproc(at::IValue out) {
  // TODO : gather location of top-N boxes & NMS
  at::IValue feature_ivalue = out.toList()[1];
  at::Tensor feature_tensor = feature_ivalue.toTensor().to(at::kCPU); // N x Length x Channel
  return feature_tensor;
}

void print_time_usage(clock_t previous_time, const char* str){
  clock_t present_time = clock();
  double time_len = ((double)(present_time-previous_time)/CLOCKS_PER_SEC)*1000;
  std::cout<<str<<" used "<<time_len<<"ms"<<"\n";
}

at::Tensor compute_one_slide(
  int tileside=8288, 
  const char* svs_path="/mnt/sda_8t/WSI_SVS/sfy4/positive/1157108 0893020.svs", 
  torch::jit::script::Module module=torch::jit::load("../../get_model_cpp/detector_yolco_input1x3x8288x8288.pt"),
  int thread_num=4,
  bool verbose=false,
  const char* save_path="/home/weiziquan/WSI_analysis/release_yolco/torch_cpp/output/features",
  int top_n=100,
  bool save_sequence=true
){
  // 计时
  clock_t _start, _end;
  _start = clock();
  double data_endtime = 0, model_endtime = 0;
  // data-loading ***************************
  slide_params slideParams(/*tile w=*/tileside, /*tile h=*/tileside, /*level=*/0);
  slide_loader loader(svs_path, &slideParams);
  if (verbose)
    loader.print_all_property();
  slide_loader *subloaders = new slide_loader[thread_num];
  for (int i=0; i<thread_num; i++)
    loader.split_loader(subloaders + i, (int) (loader.end_id/4) * i, (int) (loader.end_id/4) * (i+1));
  std::vector<at::Tensor> input_list;
  std::vector<std::future<void>> thread_handle(sizeof(std::future<void>) * thread_num);
  for (int i=0; i<thread_num; i++) 
    thread_handle[i] = std::async(std::launch::async, &slide_loader::loop_loader, subloaders+i, &input_list);
  std::vector<std::future<at::Tensor>> async_postproc(sizeof(std::future<at::Tensor>) * loader.end_id);
  _end = clock();
  if (verbose)
    std::cout<<"Init time:"<<((double)(_end-_start)/CLOCKS_PER_SEC)*1000<<"ms"<< '\n';
  else {
    auto now_time = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now_time);
    std::cout<<svs_path<<"\n";
    std::cout<<"Start at "<<std::ctime(&current_time)<<"\n";
  }
  clock_t present_time = clock();
  // wsi-processing ***************************
  int current_postproc = 0;
  at::IValue out;
  while(current_postproc < loader.end_id) {
    if (input_list.size() > current_postproc) {
      if (verbose) {
        print_time_usage(present_time, "\nstart getting tile");
        present_time = clock();
      }
      at::Tensor input = input_list[current_postproc];
      if (verbose) {
        print_time_usage(present_time, "end getting tile");
        present_time = clock();
        print_time_usage(present_time, "\nstart infer");
        present_time = clock();
      }
      out = module.forward({input.to(at::kCUDA)});
      if (verbose) {
        print_time_usage(present_time, "end infer");
        present_time = clock();
        print_time_usage(present_time, "\nstart postprocessing");
        present_time = clock();
      }
      async_postproc[current_postproc] = std::async(std::launch::async, &feature_postproc, out);// 异步后处理
      if (verbose) {
        print_time_usage(present_time, "end postprocessing");
        present_time = clock();
      }
      current_postproc += 1;
      // TODO: erase computed input from input_list
      // input = input_list.erase(input);
    }
  }
  // post-processing ***************************
  clock_t start = clock();
  std::vector<at::Tensor> outputs;//(at::Tensor) 
  for (int i=0; i<loader.end_id; i++) {
    outputs.push_back(async_postproc[i].get());
  }
  at::Tensor output = torch::cat(at::TensorList(outputs), 1);
  at::Tensor index = output.slice(-1, 768, 769).argsort(/*dim=*/1, /*descending=*/true).squeeze();
  output = output.index_select(1, index.slice(0, 0, top_n)); // gather features of top-N boxes
  clock_t end = clock();
  if (verbose)
    std::cout<<"Post-processing time:"<<((double)(end-start)/CLOCKS_PER_SEC)*1000<<"ms"<< '\n';
  // Save .pt  *****************************
  if (save_sequence) {
    std::string svs_name = std::string(svs_path + std::string(svs_path).find_last_of('/'));
    std::string save_name = std::string(save_path) + StringReplace(svs_name, std::string(".svs"), std::string(".pt"));
    torch::save(output, save_name.c_str());
  }
  // *************************************
  auto now_time = std::chrono::system_clock::now();
  std::time_t current_time = std::chrono::system_clock::to_time_t(now_time);
  std::cout<<"End at "<<std::ctime(&current_time)<<"\n";
  return output;
}


int main(int argc, const char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: example-app <tile-side> <path-to-detector-model> <SVS-path1> <SVS-path2> ...\n";
    return -1;
  }
  
  torch::jit::script::Module module, classifier;
  torch::NoGradGuard no_grad;
  // at::Tensor example;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // module = torch::jit::load("../../get_model_cpp/detector_yolco_input1x3x8288x8288.pt");
    // classifier = torch::jit::load("/home/weiziquan/WSI_analysis/release_yolco/get_model_cpp/classifier_transformer_input1x100x768.pt");
    module = torch::jit::load(argv[2]);
    classifier = torch::jit::load(argv[3]);
    // example = classifier.forward({torch::randn({1, 100, 768}).to(at::kCUDA)}).toTensor();
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // std::vector<at::Tensor> feature_sequences = {compute_one_slide()};
  // int sliden = 1;
  std::vector<at::Tensor> feature_sequences;
  int sliden = 0;
  while (sliden+4 < argc){
    feature_sequences[sliden] = compute_one_slide(atoi(argv[1]), argv[sliden+4], module);
    sliden += 1;
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
  for (int i=0; i<sliden; i++) {
    at::Tensor score = compute_final_class(classifier, feature_sequences[i].slice(2, 0, 768)).to(at::kCPU);
    std::cout<<argv[i + 4]<<" has LABEL <1>"<<"\n";
    std::cout<<"model gives SCORE <"<<score.squeeze().slice(0, 99, 100)<<">\n";
  }
  
  return 0;

}