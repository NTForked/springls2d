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
#include "SpringlsSecondOrder.h"
#include "AlloyApplication.h"
namespace aly {

	
	SpringlsSecondOrder::SpringlsSecondOrder(const std::shared_ptr<SpringlCache2D>& cache) :SpringLevelSet2D(cache){
		setName("Second Order Spring Level Set");
	}
	
	void SpringlsSecondOrder::computeForce(size_t idx, float2& f1, float2& f2, float2& f) {
		f1 = float2(0.0f);
		f2 = float2(0.0f);
		f = float2(0.0f);
		float2 p = contour.particles[idx];
		float2 p1 = contour.points[2 * idx];
		float2 p2 = contour.points[2 * idx + 1];
		if (pressureImage.size() > 0&& pressureParam.toFloat()!=0.0f) {
			float2 v1 = normalize(contour.points[2 * idx + 1] - contour.points[2 * idx]);
			float2 norm = float2(-v1.y, v1.x);
			float2 pres = pressureParam.toFloat()*norm*pressureImage(p.x, p.y).x;
			f = pres;
			f1 = f;
			f2 = f;
		}
		
		float2x2 M,A;
		if (vecFieldImage.size() > 0&& advectionParam.toFloat()!=0.0f) {
			float w = advectionParam.toFloat();
			f1 += vecFieldImage(p1.x, p1.y)*w;
			f2 += vecFieldImage(p2.x, p2.y)*w;
			f  += vecFieldImage(p.x , p.y )*w;
		}
		float2 k1, k2, k3, k4;
		k4 = contour.velocities[2][idx];
		k3 = contour.velocities[1][idx];
		k2 = contour.velocities[0][idx];
		k1 = f;

		contour.velocities[3][idx] = k4;
		contour.velocities[2][idx] = k3;
		contour.velocities[1][idx] = k2;
		contour.velocities[0][idx] = k1;
		float2 correction = f;
		if (mSimulationIteration>=4) {
			f = (1.0f / 6.0f)*(k1 + 2.0f * k2 + 2.0f * k3 + k4);
		} else if (mSimulationIteration == 3) {
			f = (1.0f / 4.0f)*(k1 + 2.0f * k2 + k3);
		} if (mSimulationIteration == 2) {
			f = (1.0f / 2.0f)*(k1 + k2);
		}
		//Approximate RK4 correction for vertices to save memory.
		f1 += f-correction;
		f2 += f-correction;
	}

}