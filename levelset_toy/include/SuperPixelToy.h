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

#ifndef SUPERPIXELTOY_H_
#define SUPERPIXELTOY_H_

#include "AlloyApplication.h"
#include "AlloyVector.h"
#include "SuperPixelLevelSet.h"
#include "AlloyWorker.h"
#include "AlloyTimeline.h"
#include "Simulation.h"
#include "SpringlCache2D.h"
class SuperPixelToy: public aly::Application {
protected:
	int example;
	aly::Image1f gray;
	aly::ImageRGBA img;
	bool showCenters;
	int lastSimTime;
	bool running = false;
	aly::Number transparency;
	aly::AdjustableCompositePtr resizeableRegion;
	std::shared_ptr<aly::SuperPixelLevelSet> simulation;
	aly::IconButtonPtr playButton,stopButton;
    std::shared_ptr<aly::TimelineSlider> timelineSlider;
	aly::ImageGlyphPtr overlayGlyph;
public:
	std::shared_ptr<aly::SpringlCache2D> cache;
	SuperPixelToy(int example);
	virtual void draw(aly::AlloyContext* context) override;
	bool init(aly::Composite& rootNode);

};

#endif

