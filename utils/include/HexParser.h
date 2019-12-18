//
// Created by grzegorz on 20.11.2019.
//

#ifndef INYNIERKA_HEXPARSER_H
#define INYNIERKA_HEXPARSER_H

#include <ostream>
#include <iomanip>


class HexParser {
    std::ostream *output = nullptr;
    unsigned int digestLength;
    unsigned char *word = nullptr;

public:
    explicit HexParser(unsigned int digestLength) : digestLength(digestLength) {}

    inline void print() {
        for (unsigned int i = 0; i < digestLength; i++)
            *output << std::hex
                    << std::setw(2)
                    << std::setfill('0')
                    << static_cast<unsigned int>(word[i])
                    << std::dec;
    }

    HexParser &operator()(unsigned char *word) {
        this->word = word;
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &out, HexParser &hexParser);

};

std::ostream &operator<<(std::ostream &out, HexParser &hexParser) {
    hexParser.output = &out;
    hexParser.print();
    return out;
}

#endif //INYNIERKA_HEXPARSER_H
