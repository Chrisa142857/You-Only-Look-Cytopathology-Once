#include "tile_slide.h"

int get_os_property(openslide_t *slide, const char* propName)
{
	const char *property = openslide_get_property_value(slide, propName);
	if (property == NULL) {
		return 0;
	}

	std::stringstream strValue;
	strValue << property;
	int intValue;
	strValue >> intValue;

	return intValue;
}

slide_loader::slide_loader(const char* path, slide_params *P, bool use_fore_g)
{
    slide = openslide_open(path);
    if (! slide ) {
        std::cerr<<path<<" Slide Cannot Open"<<"\n";
    }
    Params = P;
    slide_loader::preload_tile_location(use_fore_g);
}

slide_loader::slide_loader(const char* path, slide_params *P)
{
    slide = openslide_open(path);
    if (! slide ) {
        std::cerr<<path<<" Slide Cannot Open"<<"\n";
    }
    Params = P;
    slide_loader::preload_tile_location(false);
}

void slide_loader::split_loader(slide_loader *subloader, int start, int end) 
{
    (*subloader).slide = slide;
    (*subloader).Params = Params;
    (*subloader).start_id = start;
    (*subloader).end_id = end;
    (*subloader).current_id = start;
}

slide_loader::slide_loader()
{

}

slide_loader::~slide_loader()
{
    if (slide) {
	    openslide_close(slide);
        slide=nullptr;
    }
}

void slide_loader::preload_tile_location(bool only_fg) 
{
	openslide_get_level_dimensions(slide, /*level=*/0, &slidew, &slideh);
    boundx = get_os_property(slide, "openslide.bounds-x");
    boundy = get_os_property(slide, "openslide.bounds-y");
    // Set xlist, ylist
    int64_t x_start, y_start, x_end, y_end, x_remain, y_remain;
    x_remain = slidew % Params[0].tileW;
    y_remain = slideh % Params[0].tileH;
    x_start = (int64_t) (x_remain / 2);
    y_start = (int64_t) (y_remain / 2);
    x_end = slidew - (x_remain - x_start);
    y_end = slideh - (y_remain - y_start);
    int x_num = (int) ((slidew - x_remain) / Params[0].tileW);
    int y_num = (int) ((slideh - y_remain) / Params[0].tileH);
    Params[0].xlist = new int64_t[x_num * y_num];
    Params[0].ylist = new int64_t[x_num * y_num];
    int i = 0;
    for (int64_t x=x_start; x<x_end; x+=Params[0].tileW) {
        for (int64_t y=y_start; y<y_end; y+=Params[0].tileH) {
            Params[0].xlist[i] = x + boundx;
            Params[0].ylist[i] = y + boundy;
            i += 1;
        }
    }
    end_id = i;
    std::cout<<end_id<<" tiles in the slide"<<"\n";
    if (only_fg) {
        // TODO
        // [1] threshold foreground.
        // [2] remove background in xlist, ylist.
    }
    
}

void slide_loader::loop_loader(std::vector<at::Tensor> *inputs){
    while (true) {
        at::Tensor input;
        int flag = this->load_one_tensor(&input);
        if (flag == 0)
            break;
        (*inputs).push_back(input);
    }
}

int slide_loader::load_one_tensor(at::Tensor* inputs)
{
    if (current_id < start_id || current_id >= end_id) {
        return 0;
    }
    // tile_obj object;
    // object.x = xlist[current_id];
    // object.y = ylist[current_id];
    // object.w = tileW;
    // object.h = tileH;
    // object.level = level;
    uint32_t* tile = new uint32_t[Params[0].tileW * Params[0].tileH * 4];
    openslide_read_region(slide, tile, Params[0].xlist[current_id], Params[0].ylist[current_id], Params[0].level, Params[0].tileW, Params[0].tileH);
    current_id += 1;
    // *********************************************************************
    // 对值: 根据opencv确认数据类型正确的从openslide变为torch
    // cv::Mat image = cv::Mat(tileH, tileW, CV_8UC4, tile, cv::Mat::AUTO_STEP).clone();
    // cv::Mat img;
    // cv::cvtColor(image, img, cv::COLOR_RGBA2RGB);
    // cv::Mat channels[3];
    // cv::split(img, channels);
    // cv::Scalar mean[3];
    // cv::Scalar std[3];
    // cv::meanStdDev(channels[0], mean[0], std[0]);
    // cv::meanStdDev(channels[1], mean[1], std[1]);
    // cv::meanStdDev(channels[2], mean[2], std[2]);
    // std::cout<<channels[0](cv::Rect(0, 0, 10, 10))<<"\n";
    // std::cout<<"mean of R "<<mean[0][0]/255.0<<"mean of G "<<mean[1][0]/255.0<<"mean of B "<<mean[2][0]/255.0<<"\n";
    // at::Tensor inputs = ToTensor(img).to(torch::kFloat32).transpose(1, 3);
    // *********************************************************************
    *inputs = torch::from_blob(tile, {1, Params[0].tileW, Params[0].tileH, 4}, torch::kByte).to(torch::kFloat32); // openslide (RGBA)
    *inputs = (*inputs).slice(/*dim=*/3, /*start=*/0, /*end=*/3).transpose(1, 3).transpose(2, 3); // 取RGB、[N, W, H, C] -> [N, C, H, W]
    *inputs /= 255.0; 
    // *********************************************************************
    // 对值
    // std::cout << inputs.slice(/*dim=*/1, /*start=*/0, /*end=*/1).slice(/*dim=*/2, /*start=*/0, /*end=*/10).slice(/*dim=*/3, /*start=*/0, /*end=*/10) << '\n';
    // std::cout << "mean of R " << inputs.slice(/*dim=*/1, /*start=*/0, /*end=*/1).mean();
    // std::cout << "mean of G " << inputs.slice(/*dim=*/1, /*start=*/1, /*end=*/2).mean();
    // std::cout << "mean of B " << inputs.slice(/*dim=*/1, /*start=*/2, /*end=*/3).mean() << '\n';
    // *********************************************************************
    return 1;
}

void slide_loader::print_all_property(){
  int pi = 0;
  while(*(openslide_get_property_names(slide)+pi)) {
    std::cout<<*(openslide_get_property_names(slide)+pi)<<"\n";
    int pj = 0;
    while(*(openslide_get_property_value(slide, *(openslide_get_property_names(slide)+pi))+pj)) {
      std::cout<<*(openslide_get_property_value(slide, *(openslide_get_property_names(slide)+pi))+pj);
      pj += 1;
    }
    std::cout<<"\n";
    pi += 1;
  }
}


// tile_obj::tile_obj(uint32_t *tile_buffer, int64_t x, int64_t y, int64_t w, int64_t h, int32_t level)
// {
//     tile_obj::x = x;
//     tile_obj::y = y;
//     tile_obj::w = w;
//     tile_obj::h = h;
//     tile_obj::level = level;
//     tile = tile_buffer;
//     if (tile == nullptr)
//         tile = new uint32_t[w * h * 4];
//     else {
//         delete[]tile;
//         tile = new uint32_t[w * h * 4];
//     }
// }

// tile_obj::tile_obj()
// {
//     if (tile == nullptr)
//         tile = new uint32_t[w * h * 4];
//     else {
//         delete[]tile;
//         tile = new uint32_t[w * h * 4];
//     }
// }