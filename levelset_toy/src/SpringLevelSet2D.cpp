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
#include "SpringLevelSet2D.h"
namespace aly {
	SpringLevelSet2D::SpringLevelSet2D(const std::shared_ptr<SpringlCache2D>& cache) :ActiveContour2D("Spring Level Set 2D",cache){
	}
	void SpringLevelSet2D::setSpringls(const Vector2f& particles, const Vector2f& points) {
		contour.particles = particles;
		contour.correspondence = particles;
		contour.points = points;
		contour.updateNormals();
	}
	bool SpringLevelSet2D::init() {
		ActiveContour2D::init();
		if (contour.particles.size() == 0) {
			contour.points.clear();
			for (std::list<uint32_t> curve : contour.indexes) {
				size_t count = 0;
				uint32_t first, prev=0;
				if (curve.size() > 1) {
					for (uint32_t idx : curve) {
						if (count == 0) {
							first = idx;
						}
						else {
							contour.particles.push_back(0.5f*(contour.vertexes[prev] + contour.vertexes[idx]));
							contour.points.push_back(contour.vertexes[prev]);
							contour.points.push_back(contour.vertexes[idx]);
						}
						count++;
						prev = idx;
					}
					contour.particles.push_back(0.5f*(contour.vertexes[prev] + contour.vertexes[first]));
					contour.points.push_back(contour.vertexes[prev]);
					contour.points.push_back(contour.vertexes[first]);
				}
			}
			contour.correspondence = contour.particles;
			contour.updateNormals();
		}
	}
	void SpringLevelSet2D::cleanup() {
		ActiveContour2D::cleanup();
	}
	bool SpringLevelSet2D::stepInternal() {
		ActiveContour2D::stepInternal();
	}
	void SpringLevelSet2D::setup(const aly::ParameterPanePtr& pane) {
		ActiveContour2D::setup(pane);
	}

}