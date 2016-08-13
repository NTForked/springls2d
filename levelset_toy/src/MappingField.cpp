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
#include "MappingField.h"
#include <AlloySparseSolve.h>
namespace aly {
	void SolveLaplacianMapping(const Image1f& src, const Image1f& tar, Image2f& vectorField, int iterations) {
		const int nbrX[] = { 0,0,-1,1 };
		const int nbrY[] = { 1,-1,0,0 };
		std::vector<int2> entries;
		std::map<size_t, size_t> lookup;
		vectorField.resize(src.width, src.height);
		SparseMatrix1f A;
		Vector1f b;
		Vector1f x;
		size_t N = 0;
		Image1f out(src.width,src.height);
		for (int i = 0;i < src.width;i++) {
			for (int j = 0;j < src.height;j++) {
				float sVal = src(i, j).x;
				float tVal = tar(i, j).x;
				float val = std::min(sVal, tVal);
				if (sVal <= 0.0f&&tVal >= 0.0f) {
					size_t n = j*src.width + i;
					lookup[n] = entries.size();
					entries.push_back(int2(i,j));
				}
				else if(sVal>0.0){
					out(i, j) = float1(-1.0f);
				}
				else if (tVal<0.0) {
					out(i, j) = float1(1.0f);
				}
			}
		}
		N = entries.size();
		for (int n = 0;n < N;n++) {
			int2 pos = entries[n];
			float sVal = src(pos).x;
			float tVal = tar(pos).x;
			float val = std::min(sVal, tVal);
			bool add = false;
			for (int nn = 0;nn < 4;nn++) {
				int ii = pos.x + nbrX[nn];
				int jj = pos.y + nbrY[nn];
				if (src(ii, jj).x*sVal < 0.0f|| tar(ii, jj).x*tVal < 0.0f) {
					size_t m = ii + src.width*jj;
					if (lookup.find(m) == lookup.end()) {
						lookup[m] = entries.size();
						entries.push_back(int2(ii, jj));
					}
				}
			}
		}
		size_t M = entries.size();
		A.resize(M,M);
		b.resize(M,float1(0.0f));
		x.resize(M, float1(0.0f));
		for (int n = 0;n < M;n++) {
			int2 pos = entries[n];
			float sVal = src(pos).x;
			float tVal = tar(pos).x;
			float val = std::min(sVal, tVal);
			if (n < N) {
				for (int nn = 0;nn < 4;nn++) {
					int ii = pos.x + nbrX[nn];
					int jj = pos.y + nbrY[nn];
					size_t m = ii + src.width*jj;
					size_t j = lookup[m];
					A(n, j) += float1(1.0f);
					A(n, n) += float1(-1.0f);
				}
			} else {
				float sg = -sign(tVal);
				x[n] = sg;
				b[n] = sg;
				A(n, n) = float1(1.0f);
			}
		}
		SolveBICGStab(b, A, x, src.width*2, 0.0f, [=](int iter, double err) {
			std::cout << "Laplace Solve " << iter << ") " << err << std::endl;
			return true;
		});
		for (int n = 0;n < M;n++) {
			int2 pos = entries[n];
			out(pos) = float1(x[n]);
		}
		for (int n = 0;n < N;n++) {
			int2 pos = entries[n];
			float v21 = out(pos.x + 1, pos.y).x;
			float v12 = out(pos.x, pos.y + 1).x;
			float v10 = out(pos.x, pos.y - 1).x;
			float v01 = out(pos.x - 1, pos.y).x;
			float2 grad;
			grad.x = 0.5f*(v21 - v01);
			grad.y = 0.5f*(v12 - v10);
			float len = max(1E-6f, length(grad));
			vectorField(pos) = (grad / std::max(1E-6f, len));
		}
		/*
		WriteImageToRawFile(GetDesktopDirectory() + ALY_PATH_SEPARATOR + "out.xml", out);
		WriteImageToRawFile(GetDesktopDirectory() + ALY_PATH_SEPARATOR + "src.xml", src);
		WriteImageToRawFile(GetDesktopDirectory() + ALY_PATH_SEPARATOR + "tar.xml", tar);
		WriteImageToRawFile(GetDesktopDirectory() + ALY_PATH_SEPARATOR + "vecfield.xml",vectorField);
		*/
	}
}