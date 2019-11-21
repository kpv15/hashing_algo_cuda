//
// Created by grzegorz on 20.11.2019.
//

#ifndef INYNIERKA_HEXPARSER_H
#define INYNIERKA_HEXPARSER_H

#include <ostream>
#include <iomanip>


class HexParser {
    std::ostream &output;
    unsigned int digestLength;
public:
    HexParser(std::ostream &output, unsigned int digestLength) : output(output), digestLength(digestLength) {}

    inline void print(std::string digest) {
        for (unsigned char c: digest)
            output << std::hex

                   << std::setw(2)
                   << std::setfill('0')
                   << static_cast<unsigned int>(c);
    }
};


#endif //INYNIERKA_HEXPARSER_H
