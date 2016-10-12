/*
 * Copyright(C) 2016, Blake C. Lucas, Ph.D. (img.science@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "LevelSetToy.h"
#include "MultiLevelSetToy.h"
#include "SuperPixelToy.h"

int main(int argc, char *argv[]) {
	std::string filePath;
	int index = -1;
	try {
		if(argc==1){
			std::cout << "Usage: " << argv[0]<< " [example index]\nToy Examples:"<< std::endl;
			std::cout << "[0] Original Level Set" << std::endl;
			std::cout << "[1] Spring Level Set" << std::endl;
			std::cout << "[2] Second Order Spring Level Set" << std::endl;
			std::cout << "[3] Multi-Object Level Set" << std::endl;
			std::cout << "[4] Multi-Object Spring Level Set" << std::endl;
			std::cout << "[5] Super-Pixel Level Set" << std::endl;
			std::cout << ">> Enter Example Number: ";
			std::cin >> index;
		} else {
			index = std::atoi(argv[1]);
		}
		if (index < 3) {
			LevelSetToy app(index);
			app.run();
		}
		else if (index <5) {
			MultiLevelSetToy app(index-3);
			app.run();
		}
		else if (index == 5) {
			SuperPixelToy app(0);
			app.run();
		}
		return 0;
	} catch (std::exception& e) {
		std::cout << "Main Error: " << e.what() << std::endl;
		std::getchar();
		return 1;
	}
	

}
