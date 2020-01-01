//
// Created by grzegorz on 15.12.2019.
//

#include <iostream>
#include <cstring>
#include <fstream>
#include "include/ResultComparator.h"

bool ResultComparator::compare() {
    std::vector<std::ifstream *> files;
    unsigned int n = 0, current_n = 0;
    unsigned int length = 0, current_length = 0;

    std::cout << "opening and checking file sizes" << std::endl;
    for (unsigned int i = 0; i < filesNames.size(); i++) {
        auto *file = new std::ifstream(filesNames[i]);

        *file >> current_n;
        *file >> current_length;

        if (i != 0 && ((current_length != length && length > 0) || (current_n != n && n > 0))) {
            std::cout << "files have different sizes" << std::endl;
            return false; //todo add file closing
        } else {
            n = current_n;
            length = current_length;
        }
        files.push_back(file);
    }

    std::cout << "comparing file contexts" << std::endl;

    unsigned int files_n = files.size();
    char **buffer = new char *[files_n];
    for (unsigned int i = 0; i < files_n; i++) {
        buffer[i] = new char[length + 1];
        strcpy(buffer[i], "");
    }
    for (unsigned int i = 0; i < n + 1; i++) {
        for (unsigned int j = 0; j < files_n; j++) {
            files[j]->getline(buffer[j], length + 1);
        }
        if (i == 0) continue;
        for (unsigned int j = 1; j < files_n; j++) {
            if (strcmp(buffer[0], buffer[j]) != 0) {
                std::cout << "different data in line " << i + 1 << std::endl;
                return false; //todo add file closing and repair compare
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
    for (unsigned int i = 0; i < files_n; i++)
        delete[] buffer[i];
    delete[] buffer;

    return true;
}
