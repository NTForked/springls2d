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

#ifndef INCLUDE_MultiActiveContour2D_H_
#define INCLUDE_MultiActiveContour2D_H_
#include <AlloyVector.h>
#include <AlloyImage.h>
#include <AlloyMultiIsoContour.h>
#include <vector>
#include <list>
#include <tuple>
#include <mutex>
#include "Simulation.h"
#include "SpringlCache2D.h"
namespace aly {
	class MultiActiveContour2D: public Simulation {
	protected:
		std::shared_ptr<SpringlCache2D> cache;
		MultiIsoContour isoContour;
		Manifold2D contour;
		std::vector<int> labelList;
		bool preserveTopology;
		bool clampSpeed;

		Number advectionParam;
		Number pressureParam;
		Number curvatureParam;
		Number targetPressureParam;
		
		const float MAX_DISTANCE = 3.5f;
		const int maxLayers = 3;
		bool requestUpdateContour;
		bool requestUpdateOverlay;
		Image1f initialLevelSet;
		Image1i initialLabels;
		Image1f levelSet;
		Image1i labelImage;
		Image1f swapLevelSet;
		Image1i swapLabelImage;
		Image1f pressureImage;
		Image2f vecFieldImage;
		std::vector<float> deltaLevelSet;
		std::vector<int> objectIds;
		std::vector<int2> activeList;
		std::vector<int> forceIndexes;
		std::mutex contourLock;
		bool getBitValue(int i);
		void rescale(aly::Image1f& pressureForce);
		void pressureMotion(int i, int j, size_t index);
		void pressureAndAdvectionMotion(int i, int j, size_t index);
		void advectionMotion(int i, int j, size_t index);
		void applyForces(int i, int j, size_t index, float timeStep);
		void plugLevelSet(int i, int j, size_t index);
		void updateDistanceField(int i, int j, int band);
		int deleteElements();
		int addElements();
		virtual float evolve(float maxStep);
		void rebuildNarrowBand();
		void applyForcesTopoRule(int i, int j, int offset, size_t index, float timeStep);
		virtual bool stepInternal() override;
		float getLevelSetValue(int i, int j, int l) const;
		float getLevelSetValue(float i, float j, int l) const;
		float getSwapLevelSetValue(int i, int j, int l) const;
		std::map<int, aly::Color> lineColors;
	public:
		MultiActiveContour2D(const std::shared_ptr<SpringlCache2D>& cache=nullptr);
		MultiActiveContour2D(const std::string& name,const std::shared_ptr<SpringlCache2D>& cache = nullptr);
		void setCurvature(float c) {
			curvatureParam.setValue(c);
		}
		void setPressure(float c) {
			pressureParam.setValue(c);
		}
		aly::Color getColor(int l) {
			return lineColors[l];
		}
		bool updateContour();
		virtual bool updateOverlay();
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
		int getNumLabels() const {
			return (int)labelList.size();
		}
		int getLabel(int index) const {
			return (int)labelList[index];
		}
		virtual bool init()override;
		virtual void cleanup() override;
		virtual void setup(const aly::ParameterPanePtr& pane) override;
		void setInitial(const Image1f& img,const Image1i& labels) {
			initialLevelSet = img;
			initialLabels = labels;
		}
		void setInitial(const Image1i& labels);
	};
}

#endif /* INCLUDE_ACTIVEManifold2D_H_ */
