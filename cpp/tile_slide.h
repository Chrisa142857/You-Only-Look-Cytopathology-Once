#ifndef TILE_SLIDE_H
#define TILE_SLIDE_H
#include <openslide/openslide.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

class slide_params {
    public:
        int64_t tileW=0; 
        int64_t tileH=0; 
        int32_t level=0;
        int64_t* xlist;
        int64_t* ylist;
        bool need_value=true;
        slide_params();
        slide_params(int64_t w, int64_t h, int32_t l) { // init parameters
            tileW = w; tileH = h; level = l;
            need_value = false;
        };
};


class input_object {
    public:
        at::Tensor input;
        int64_t samplex;
        int64_t sampley;
        input_object(at::Tensor i, int64_t x, int64_t y){
            input = i;
            samplex = x;
            sampley = y;
        };
};


class slide_loader {
    private:
        void preload_tile_location(bool use_fore_g);
    public:
        // For crop tile
        openslide_t* slide=nullptr;
        slide_params* Params;
        int start_id = 0;
        int end_id = 0;
        int current_id = 0;
        // For tile x,y,w,h
        int boundx = 0;
        int boundy = 0;
        int64_t slidew = 0;
        int64_t slideh = 0;
        // Functions
        slide_loader();
        slide_loader(const char* path, slide_params *P); // extract every tile
        slide_loader(const char* path, slide_params *P, bool use_fore_g); // extract foreground tile
        ~slide_loader(); // kill all loaders
        void print_all_property();
        int load_one_tensor(at::Tensor* inputs);
        void loop_loader(std::vector<input_object> *inputs);
        void split_loader(slide_loader *subloader, int start, int end);// for multiple loader
};

#endif