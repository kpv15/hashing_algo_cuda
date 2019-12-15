//
// Created by grzegorz on 15.12.2019.
//

#include <iostream>
#include <cstring>
#include <fstream>
#include "ResultComparator.h"

bool ResultComparator::compare() {
    std::vector<std::ifstream *> files;
    unsigned int n = 0, current_n;
    unsigned int length = 0, current_length;

    std::cout << "opening and checking file sizes" << std::endl;
    for (auto &fileName:filesNames) {
        std::ifstream *file = new std::ifstream(fileName);

        *file >> current_n;
        *file >> current_length;

        if ((current_length != length && length > 0) || (current_n != n && n > 0)) {
            std::cout << "files have different sizes" << std::endl;
            return false; //todo add file closing
        }
        files.push_back(file);
    }
    std::cout << "comparing file contexts" << std::endl;

    char *buffer = new char[length + 1];
    char *buffer2 = new char[length + 1];
    strcpy(buffer, "");
    for (unsigned int i = 0; i < n + 1; i++) {
        for (auto file:files) {
            if (strcmp(buffer, ""))
                file->getline(buffer, length + 1);
            else {
                file->getline(buffer2, length + 1);
                if (strcmp(buffer, buffer2)) {
                    std::cout << "different data in line " << i << std::endl;
                    return false; //todo add file closing
                }
            }
        }
    }

    std::cout << "file has the same contexts" << std::endl;
    std::cout << "closing files" << std::endl;
    for (auto &file:files) {
        file->close();
        delete file;
    }
    files.clear();
    return true;
}
