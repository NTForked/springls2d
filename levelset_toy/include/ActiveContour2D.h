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

#ifndef INCLUDE_ACTIVEManifold2D_H_
#define INCLUDE_ACTIVEManifold2D_H_
#include <AlloyVector.h>
#include <AlloyImage.h>
#include <AlloyIsoContour.h>
#include <vector>
#include <list>
#include <tuple>
#include <mutex>
#include "Simulation.h"
#include "SpringlCache2D.h"
namespace aly {
	class ActiveManifold2D: public Simulation {
	protected:
		std::shared_ptr<SpringlCache2D> cache;
		IsoContour isoContour;
		Manifold2D contour;
		bool preserveTopology;
		bool clampSpeed;

		Number advectionParam;
		Number pressureParam;
		Number curvatureParam;
		Number targetPressureParam;
		
		const float MAX_DISTANCE = 3.5f;
		const int maxLayers = 3;
		bool requestUpdateContour;
		Image1f initialLevelSet;
		Image1f levelSet;
		Image1f swapLevelSet;
		Image1f pressureImage;
		Image2f vecFieldImage;
		std::vector<float> deltaLevelSet;
		std::vector<int2> activeList;
		std::mutex contourLock;
		bool getBitValue(int i);
		void rescale(aly::Image1f& pressureForce);
		void pressureMotion(int i, int j, size_t index);
		void pressureAndAdvectionMotion(int i, int j, size_t index);
		void advectionMotion(int i, int j, size_t index);
		void applyForces(int i, int j, size_t index, float timeStep);
		void plugLevelSet(int i, int j, size_t index);
		void updateDistanceField(int i, int j, int band, size_t index);
		int deleteElements();
		int addElements();
		virtual float evolve(float maxStep);
		void rebuildNarrowBand();

		bool updateContour();
		void applyForcesTopoRule(int i, int j, int offset, size_t index, float timeStep);
		virtual bool stepInternal() override;
	public:
		ActiveManifold2D(const std::shared_ptr<SpringlCache2D>& cache=nullptr);
		ActiveManifold2D(const std::string& name,const std::shared_ptr<SpringlCache2D>& cache = nullptr);
		void setCurvature(float c) {
			curvatureParam.setValue(c);
		}
		void setPressure(const Image1f& img, float weight, float target) {
			pressureParam.setValue(weight);
			targetPressureParam.setValue(target);
			pressureImage = img;
			rescale(pressureImage);
		}
		void setPressure(const Image1f& img, float weight) {
			pressureParam.setValue(weight);
			pressureImage = img;
			rescale(pressureImage);
		}
		void setPressure(const Image1f& img) {
			pressureImage = img;
			rescale(pressureImage);
		}
		void setVectorField(const Image2f& img, float f) {
			advectionParam.setValue(f);
			vecFieldImage = img;
		}
		Manifold2D* getContour();

		virtual bool init()override;
		virtual void cleanup() override;
		virtual void setup(const aly::ParameterPanePtr& pane) override;

		void setInitialDistanceField(const Image1f& img) {
			initialLevelSet = img;
		}
	};
}

#endif /* INCLUDE_ACTIVEManifold2D_H_ */
