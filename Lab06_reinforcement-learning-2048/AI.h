#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include "2048.h"

class experience {
public:
    state sp;
    state spp;
};


class AI {
public:
    static void load_tuple_weights() {
        std::string filename = "xxx.weight";                   // put the name of weight file here
        std::ifstream in;
        in.open(filename.c_str(), std::ios::in | std::ios::binary);
        if (in.is_open()) {
            for (size_t i = 0; i < feature::list().size(); i++) {
                in >> *(feature::list()[i]);
                std::cout << feature::list()[i]->name() << " is loaded from " << filename << std::endl;
            }
            in.close();
        }
    }

    static void set_tuples() {

    }

    static int get_best_move(state s) {         // return best move dir

    }

};