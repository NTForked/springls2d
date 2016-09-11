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
#ifndef INCLUDE_SUPERPIXELLEVELSET_H
#define INCLUDE_SUPERPIXELLEVELSET_H
#include "MultiActiveContour2D.h"
#include "SLIC.h"
namespace aly {
	class SuperPixelLevelSet : public MultiActiveContour2D {
	protected:
		float maxClusterDistance;
		float meanClusterDistance;
		bool updateClusterCenters;
		bool updateCompactness;
		aly::ImageRGBA referenceImage;
		Number pruneInterval;
		Number splitInterval;
		Number splitThreshold;
		Number superPixelCount;
		Number superPixelIterations;
		SuperPixels superPixels;
		void superPixelMotion(int i, int j, size_t index);
		virtual float evolve(float maxStep) override;
		virtual bool stepInternal() override;
		void reinit();
	public:
		SuperPixelLevelSet(const std::shared_ptr<SpringlCache2D>& cache = nullptr);
		SuperPixelLevelSet(const std::string& name, const std::shared_ptr<SpringlCache2D>& cache = nullptr);
		virtual void setup(const aly::ParameterPanePtr& pane) override;
		virtual bool init() override;
		void setReference(const ImageRGBA& img) {
			referenceImage = img;
		}
		virtual bool updateOverlay() override;
		void setPruneIterval(int v) {
			pruneInterval.setValue(v);
		}
		void setSplitThreshold(float v) {
			splitThreshold.setValue(v);
		}
		void setSplitIterval(int v) {
			splitInterval.setValue(v);
		}

	};
}
#endif