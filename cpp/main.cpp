#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <openslide/openslide.h>
#include <c10/cuda/CUDACachingAllocator.h>
// #include <torchvision/ops/cpu/roi_align_common.h>
// #include <torchvision/ops/ops.h>
#include <torchvision/ops/nms.h>

#include "argparse/argparse.hpp"
#include "tile_slide.h"
#include "timer.cpp"

#include <fstream>
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
  auto now_time = std::chrono::system_clock::now();
  std::time_t current_time = std::chrono::system_clock::to_time_t(now_time);
  std::cout<<"Classification End at "<<std::ctime(&current_time)<<"\n";
  return output;
}

at::Tensor feature_postproc(at::IValue out, bool half_prec) {
  at::IValue feature_ivalue = out.toList()[1];
  at::Tensor feature_tensor = feature_ivalue.toTensor().to(at::kCPU); // N x Length x Channel
  if (half_prec) 
    feature_tensor = feature_tensor.to(at::kFloat);
  return feature_tensor;
}

void det_postproc(at::IValue out, std::string save_name, bool half_prec, float conf_thres, float iou_thres, int num_class) {
  int fg_score_dim = 4;
  int cls_score_dim = 5;
  at::IValue det_ivalue = out.toList()[0];
  at::Tensor det_tensor = det_ivalue.toTensor().squeeze().to(at::kCPU); // N x 7
  auto max_tuple = det_tensor.slice(-1, cls_score_dim, cls_score_dim+num_class).max(1, true);
  at::Tensor class_conf = std::get<0>(max_tuple);
  at::Tensor class_pred = std::get<1>(max_tuple);
  at::Tensor score = det_tensor.slice(-1, fg_score_dim, fg_score_dim+1) * class_conf;
  score = score.squeeze();
  at::Tensor ind = at::where(score > conf_thres)[0];
  score = score.index_select(0, ind);
  class_pred = class_pred.index_select(0, ind);
  at::Tensor box = det_tensor.slice(-1, 0, 4).squeeze().index_select(0, ind);
  box = at::cat(at::TensorList({
    box.slice(-1, 0, 1) - box.slice(-1, 2, 3)/2,
    box.slice(-1, 1, 2) - box.slice(-1, 3, 4)/2,
    box.slice(-1, 0, 1) + box.slice(-1, 2, 3)/2,
    box.slice(-1, 1, 2) + box.slice(-1, 3, 4)/2
  }), -1);
  at::Tensor nms_ind = vision::ops::nms(box, score, iou_thres);
  det_tensor = at::cat(
    at::TensorList({
      box.index_select(0, nms_ind), 
      score.index_select(0, nms_ind).unsqueeze(-1), 
      class_pred.index_select(0, nms_ind)
      }), -1
    );
  if (half_prec) 
    det_tensor = det_tensor.to(at::kFloat);
  std::ofstream save_file;
  save_file.open(save_name.c_str());
  save_file << det_tensor;
  save_file.close();
}

void print_current_time(){
    auto now_time = std::chrono::high_resolution_clock::now();
    std::time_t current_time = std::chrono::high_resolution_clock::to_time_t(now_time);
    std::cout<<std::ctime(&current_time)<<"\n";
}

at::Tensor compute_one_slide(
  int tileside, //=8288
  const char* svs_path, //="/mnt/sda_8t/WSI_SVS/sfy4/positive/1157108 0893020.svs", 
  torch::jit::script::Module module, //=torch::jit::load("../../get_model_cpp/detector_yolco_input1x3x7264x7264.pt"),
  int thread_num, //=4,
  bool verbose, //=true,
  const char* save_path, //="/home/weiziquan/WSI_analysis/release_yolco/torch_cpp/output/features",
  int top_n, //=100,
  float det_conf_thres,
  float nms_iou_thres,
  int num_class,
  bool save_sequence, //=false,
  bool half_prec, //=false
  bool only_det,
  bool normalize
  ){
  // data-loading ***************************
  slide_params slideParams(/*tile w=*/tileside, /*tile h=*/tileside, /*level=*/0);
  slide_loader loader(svs_path, &slideParams);
  int sub_len = (int) (loader.end_id/thread_num);
  float loader_num_float = loader.end_id/(sub_len*1.0);
  int loader_num = (int)std::ceil(loader_num_float);
  slide_loader *subloaders = new slide_loader[loader_num];
  for (int i=0; i<loader_num; i++) {
    int sub_start = sub_len * i;
    int sub_end = sub_len * (i+1);
    if (sub_start > loader.end_id)
      sub_start = loader.end_id;
    if (sub_end > loader.end_id)
      sub_end = loader.end_id;
    loader.split_loader(subloaders + i, sub_start, sub_end);
  }
  std::vector<input_object> input_list;
  std::vector<std::future<void>> thread_handle(sizeof(std::future<void>) * loader_num);
  for (int i=0; i<loader_num; i++) 
    thread_handle[i] = std::async(std::launch::async, &slide_loader::loop_loader, subloaders+i, &input_list, normalize);
  std::vector<std::future<at::Tensor>> async_postproc;
  std::vector<std::future<void>> async_postproc_det;
  if (only_det != true)
    async_postproc = std::vector<std::future<at::Tensor>>(sizeof(std::future<at::Tensor>) * loader.end_id);
  async_postproc_det = std::vector<std::future<void>>(sizeof(std::future<void>) * loader.end_id);
  std::cout<<svs_path<<"\nStart at ";
  print_current_time();
  Timer timer_var, timer_whole, timer_delay;
  double infer_time = 0, data_time = 0;
  // wsi-processing ***************************
  int current_postproc = 0;
  at::IValue out;
  double delay_time = 0;
  while(current_postproc < loader.end_id) {
    if (input_list.size() > current_postproc) {
      input_object obj = input_list[current_postproc];
      at::Tensor input = obj.input;
      int64_t samplex = obj.samplex;
      int64_t sampley = obj.sampley;
      if (verbose) {
        data_time += timer_var.elapsed();
        timer_var.reset();
      }
      if (half_prec)
        input = input.to(at::kHalf);
      
      // std::cout<<"Computing x:"<<samplex<<" y:"<<sampley<<"\n";
      out = module.forward({input.to(at::kCUDA)});
      // std::cout<<"Computed x:"<<samplex<<" y:"<<sampley<<"\n";
      if (only_det == true) {
        std::string svs_name = std::string(svs_path + std::string(svs_path).find_last_of('/'));
        std::string save_name = std::string(save_path) + StringReplace(
          svs_name, 
          std::string(".svs"), 
          std::string("_x") + std::to_string(samplex) + std::string("_y") + std::to_string(sampley) + std::string(".txt")
        );
        async_postproc_det[current_postproc] = std::async(std::launch::async, &det_postproc, out, save_name, half_prec, det_conf_thres, nms_iou_thres, num_class);// async post-processing
      } else {
        async_postproc[current_postproc] = std::async(std::launch::async, &feature_postproc, out, half_prec);// async post-processing
      }
      // input_list[current_postproc] = input_object(torch::empty({0}), 0, 0);
      current_postproc += 1;
      if (verbose) {
        infer_time += timer_var.elapsed();
        timer_var.reset();
      }
      delay_time = 0;
      timer_delay.reset();
    } else {
      double pre_delay_time = delay_time;
      delay_time += timer_delay.elapsed();
      timer_delay.reset();
      if (pre_delay_time != delay_time)
        std::cout<<"delay_time "<<delay_time<<"ms\n";
    }
    if (delay_time > 3000){
      break;
    }
  }
  if (verbose) {
    std::cout<<"infer mean time "<<infer_time / current_postproc<<" ms\n";
    std::cout<<"data mean time "<<data_time / current_postproc<<" ms\n";
    std::cout<<"end all tiles "<<timer_whole.elapsed()<<" ms\n";
  }
  // input_list.clear();
  // post-processing ***************************
  at::Tensor output;
  if (only_det == true) {
    if (verbose)
      timer_var.reset();
    for (int i=0; i<loader.end_id; i++) {
      async_postproc_det[i].get();
    }
    if (verbose) 
      std::cout<<"post-processing "<<timer_var.elapsed()<<" ms\n";
    output = torch::empty({0});
  } else {
    if (verbose)
      timer_var.reset();
    std::vector<at::Tensor> outputs;//(at::Tensor) 
    for (int i=0; i<loader.end_id; i++) {
      outputs.push_back(async_postproc[i].get());
    }
    output = torch::cat(at::TensorList(outputs), 1);
    at::Tensor index = output.slice(-1, 768, 769).argsort(/*dim=*/1, /*descending=*/true).squeeze();
    output = output.index_select(1, index.slice(0, 0, top_n)); // gather features of top-N boxes
    if (verbose) 
      std::cout<<"post-processing "<<timer_var.elapsed()<<" ms\n";
    // Save features  *****************************
    if (save_sequence) {
      if (verbose)
        timer_var.reset();
      std::string svs_name = std::string(svs_path + std::string(svs_path).find_last_of('/'));
      std::string save_name = std::string(save_path) + StringReplace(svs_name, std::string(".svs"), std::string(".pt"));
      torch::save(output, save_name.c_str());
      if (verbose)
        std::cout<<"save feature seq "<<timer_var.elapsed()<<" ms\n";
    }
    // *************************************
  }
  std::cout<<"\nDetection End at ";
  print_current_time();
  return output;
}


int main(int argc, const char* argv[]) {
  argparse::ArgumentParser program("main");
  program.add_argument("-s", "--input_side")
    .help("side length of input tile image")
    .required()
    .default_value(std::string("5120")); 
  program.add_argument("-c", "--classifier")
    .help("classifier's path")
    .required()
    .default_value(std::string("../get_model_cpp/classifier_transformer_input1x100x768.pt")); 
  program.add_argument("-d", "--detector")
    .help("detector's path")
    .required()
    .default_value(std::string("../../model_zoo/detector_yolox_l_input1x3x5120x5120.pt")); 
  program.add_argument("-cthr", "--classifier_thres")
    .help("set classifier conf thres")
    .required()
    .default_value(std::string("0.184")); 
  program.add_argument("--thread_num")
    .help("thread number for data loader")
    .required()
    .default_value(std::string("4")); 
  program.add_argument("--feature_num")
    .help("feature sequence number to classifer (need to be the same as classifier's input size)")
    .required()
    .default_value(std::string("100")); 
  program.add_argument("-o", "--output_dir")
    .help("specify the output dir.")
    .required()
    .default_value(std::string("./outputs/"));
  program.add_argument("-dthr", "--detector_thres")
    .help("set detector conf thres")
    .required()
    .default_value(std::string("0.5")); 
  program.add_argument("-nmsthr", "--nms_thres")
    .help("set nms iou thres")
    .required()
    .default_value(std::string("0.1")); 
  program.add_argument("-nclass", "--num_class")
    .help("num_class")
    .required()
    .default_value(std::string("1")); 
  program.add_argument("--normalize")
    .help("normalize the input image (unused for yolco)")
    .default_value(false)
    .implicit_value(true); 
  program.add_argument("--half")
    .help("half precision")
    .default_value(false)
    .implicit_value(true); 
  program.add_argument("--only_det")
    .help("do only the detection")
    .default_value(false)
    .implicit_value(true); 
  program.add_argument("--save_feature")
    .help("save the feature sequence of WSI")
    .default_value(false)
    .implicit_value(true); 
  program.add_argument("--verbose")
    .help("display the verbose info")
    .default_value(false)
    .implicit_value(true); 
  program.add_argument("input_slides")
    .help("list of input slide path")
    .remaining();
  
  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr << "Error throwed in argparse" << std::endl;
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }
  
  // std::filesystem::create_directory(program["--output_dir"]);
  torch::jit::script::Module module, classifier;
  // at::Tensor example;
  bool is_only_det = program.get<bool>("--only_det");
  bool normalize = program.get<bool>("--normalize");
  std::string detector_path = program.get<std::string>("--detector");
  std::string classifier_path = program.get<std::string>("--classifier");
  std::string output_dir = program.get<std::string>("--output_dir");
  std::cout<<"detector_path: "<<detector_path<<"\n";
  if (is_only_det != true) {
    std::cout<<"classifier_path: "<<classifier_path<<"\n";
  }
  std::cout<<"is_only_det: "<<is_only_det<<"\n";
  std::cout<<"output_dir: "<<output_dir<<"\n";
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(detector_path.c_str());
    if (is_only_det != true) {
      classifier = torch::jit::load(classifier_path.c_str());
    }
  }
  catch (const c10::Error& e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "error loading the model\n";
    return -1;
  }

  if (program["--half"] == true)
    std::cout<<"using Half precision\n";
  std::vector<std::string> files = program.get<std::vector<std::string>>("input_slides");
  if (files.size() == 0) {
    std::cerr << "No WSI is provided\n";
    return -1;
  }
  std::cout << files.size() << " WSI files provided" << std::endl;
  int input_side = stoi(program.get<std::string>("--input_side"));
  int thread_num = stoi(program.get<std::string>("--thread_num"));
  int feature_num = stoi(program.get<std::string>("--feature_num"));
  float cls_thres = stof(program.get<std::string>("--classifier_thres"));
  float det_thres = stof(program.get<std::string>("--detector_thres"));
  float nms_thres = stof(program.get<std::string>("--nms_thres"));
  int num_class = stoi(program.get<std::string>("--num_class"));
  std::cout<<"num_class: "<<num_class<<"\n";
  for (std::string& file : files) {
    torch::NoGradGuard no_grad;
    at::Tensor feature_sequence = compute_one_slide(
      input_side, 
      file.c_str(), 
      module, 
      thread_num,
      program["--verbose"] == true, 
      output_dir.c_str(),
      feature_num,
      det_thres,
      nms_thres,
      num_class,
      program["--save_feature"] == true,
      program["--half"] == true,
      is_only_det == true, //program["--only_det"] == true
      normalize == true
    ).detach();
    c10::cuda::CUDACachingAllocator::emptyCache();
    if (program["--only_det"] == true)
      continue;
    at::Tensor score = compute_final_class(classifier, feature_sequence.slice(2, 0, 768)).to(at::kCPU).squeeze().slice(0, 99, 100).detach();
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::string pred;
    if (score.item<float>() >= cls_thres) 
      pred = std::string("positive");
    else
      pred = std::string("negative");
    std::cout<<"**********************\n"<<file<<" has LABEL <1>"<<"\n";
    std::cout<<"model gives CLASS <"<<pred<<">\n**********************\n";
  }
  
  return 0;

}