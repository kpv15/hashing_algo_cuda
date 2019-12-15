//
// Created by grzegorz on 15.12.2019.
//

#ifndef INYNIERKA_RESULTCOMPARATOR_H
#define INYNIERKA_RESULTCOMPARATOR_H


#include <vector>
#include <string>

class ResultComparator {
    std::vector<std::string> filesNames;

public:
    explicit ResultComparator(std::vector<std::string> &filesNames) : filesNames(filesNames) {};

    bool compare();
};

#endif //INYNIERKA_RESULTCOMPARATOR_H
