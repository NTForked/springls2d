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
	void SolveGradientVectorFlow(const Image1f& src, Image2f& vectorField,bool normalize) {
		const int nbrX[] = { 0,0,-1,1 };
		const int nbrY[] = { 1,-1,0,0 };
		vectorField.resize(src.width, src.height);
		SparseMatrix2f A;
		Vector2f b;
		Vector2f x;
		size_t N = 0;
		Image1b mask(src.width,src.height);
		mask.set(bool1(false));
		for (int i = 0;i < src.width;i++) {
			for (int j = 0;j < src.height;j++) {
				float sVal = src(i, j).x;
				bool masked = false;
				for (int nn = 0;nn < 4;nn++) {
					int ii = i + nbrX[nn];
					int jj = j + nbrY[nn];
					if (src(ii, jj).x*sVal < 0.0f) {
						mask(i,j) = bool1(true);
						masked = true;
						break;
					}
				}
				if (!masked) {
					N++;
				}
			}
		}
		size_t M = src.width*src.height;
		A.resize(M, M);
		b.resize(M, float2(0.0f));
		x.resize(M, float2(0.0f));
		for (int i = 0;i < src.width;i++) {
			for (int j = 0;j < src.height;j++) {
				int idx = i+j*src.width;
				if (mask(i, j).x) {
					float v21 = src(i + 1, j).x;
					float v12 = src(i, j + 1).x;
					float v10 = src(i, j - 1).x;
					float v01 = src(i - 1, j).x;
					float v11 = src(i, j).x;
					float2 grad;
					if (v11 < 0.0f) {
						grad.x = std::max(v11 - v01,0.0f)+ std::min(v21 - v11,0.0f);
						grad.y = std::max(v11 - v10,0.0f)+ std::min(v12 - v11,0.0f);
					}
					else {
						grad.x = std::min(v11 - v01, 0.0f) + std::max(v21 - v11, 0.0f);
						grad.y = std::min(v11 - v10, 0.0f) + std::max(v12 - v11, 0.0f);
					}
					float len = max(1E-6f, length(grad));
					grad = -sign(v11)*(grad / std::max(1E-6f, len));
					x[idx] = grad;
					b[idx] = grad;
					A(idx, idx) = float2(1.0f);
				}
				else {
					for (int nn = 0; nn < 4; nn++) {
						int ii = i + nbrX[nn];
						int jj = j + nbrY[nn];
						if (ii >= 0 && ii < src.width&&jj>=0 && jj < src.height) {
							A(idx, ii + src.width*jj) += float2(1.0f);
							A(idx, idx) += float2(-1.0f);
						}
					}
				}
			}
		}
		SolveVecBICGStab(b, A, x, src.width, 1E-10f);
		vectorField.set(x.data);
		const float minSpeed = 0.1f;
		const float captureDist = 1.5f;
		if (normalize) {
			for (int i = 0;i < src.width;i++) {
				for (int j = 0;j < src.height;j++) {
					float d = std::abs(src(i, j).x);
					vectorField(i, j) = aly::normalize(vectorField(i, j))*(minSpeed + (1.0f - minSpeed)*clamp(d, 0.0f, captureDist) / captureDist);
				}
			}
		}
	}
}