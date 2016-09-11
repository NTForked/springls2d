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
#include "Alloy.h"
#include "SuperPixelToy.h"
#include "AlloyDistanceField.h"
#include "AlloyIsoContour.h"
#include "SpringlsSecondOrder.h"
#include "GradientVectorFlow.h"
#include "SLIC.h"
using namespace aly;

SuperPixelToy::SuperPixelToy(int example) :
	Application(1600, 800, "Super-Pixel Level Set Toy", false), showCenters(true),showContours(true),example(example), lastSimTime(-1){
}


bool SuperPixelToy::init(Composite& rootNode) {
	cache = std::shared_ptr<SpringlCache2D>(new SpringlCache2D());
	ImageRGBA down;
	transparency = Float(1.0f);
	lineWidth = Float(1.0f);
	ReadImageFromFile(getFullPath("images/picnic.png"),down);
	DownSample(down,img);
	if (example == 0) {
		simulation = std::shared_ptr<SuperPixelLevelSet>(new SuperPixelLevelSet(cache));
		simulation->setCurvature(0.3f);
		simulation->setPressure(0.5f);
	}
	else {
		return false;
	}

	simulation->onUpdate = [this](uint64_t iteration, bool lastIteration) {
		if (lastIteration || (int)iteration == timelineSlider->getMaxValue().toInteger()) {
			stopButton->setVisible(false);
			playButton->setVisible(true);
			running = false;
		}
		AlloyApplicationContext()->addDeferredTask([this]() {
			timelineSlider->setUpperValue((int)simulation->getSimulationIteration());
			timelineSlider->setTimeValue((int)simulation->getSimulationIteration());
		});
	};

	simulation->setReference(img);
	simulation->init();
	BorderCompositePtr layout = BorderCompositePtr(new BorderComposite("UI Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), false));
	ParameterPanePtr controls = ParameterPanePtr(new ParameterPane("Controls", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	BorderCompositePtr controlLayout = BorderCompositePtr(new BorderComposite("Control Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));

	controls->setAlwaysShowVerticalScrollBar(false);
	controls->setScrollEnabled(false);
	controls->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controls->borderColor = MakeColor(getContext()->theme.DARK);
	controls->borderWidth = UnitPX(1.0f);

	controlLayout->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controlLayout->borderWidth = UnitPX(0.0f);
	CompositePtr renderRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	layout->setWest(controlLayout, UnitPX(400.0f));
	controlLayout->setCenter(controls);
	layout->setCenter(renderRegion);
	CompositePtr infoComposite = CompositePtr(new Composite("Info", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	infoComposite->backgroundColor = MakeColor(getContext()->theme.DARKER);
	infoComposite->borderColor = MakeColor(getContext()->theme.DARK);
	infoComposite->borderWidth = UnitPX(0.0f);
	playButton = IconButtonPtr(new IconButton(0xf144, CoordPerPX(0.5f, 0.5f, -35.0f, -35.0f), CoordPX(70.0f, 70.0f)));
	stopButton = IconButtonPtr(new IconButton(0xf28d, CoordPerPX(0.5f, 0.5f, -35.0f, -35.0f), CoordPX(70.0f, 70.0f)));
	playButton->borderWidth = UnitPX(0.0f);
	stopButton->borderWidth = UnitPX(0.0f);
	playButton->backgroundColor = MakeColor(getContext()->theme.DARKER);
	stopButton->backgroundColor = MakeColor(getContext()->theme.DARKER);
	playButton->foregroundColor = MakeColor(0, 0, 0, 0);
	stopButton->foregroundColor = MakeColor(0, 0, 0, 0);
	playButton->iconColor = MakeColor(getContext()->theme.LIGHTER);
	stopButton->iconColor = MakeColor(getContext()->theme.LIGHTER);
	playButton->borderColor = MakeColor(getContext()->theme.LIGHTEST);
	stopButton->borderColor = MakeColor(getContext()->theme.LIGHTEST);
	playButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			stopButton->setVisible(true);
			playButton->setVisible(false);
			cache->clear();
			int maxIteration = (int)std::ceil(simulation->getSimulationDuration() / simulation->getTimeStep());
			timelineSlider->setTimeValue(0);
			timelineSlider->setMaxValue(maxIteration);
			timelineSlider->setVisible(true);
			context->addDeferredTask([this]() {
				simulation->init();
				running = true;
			});
			return true;
		}
		return false;
	};
	stopButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			stopButton->setVisible(false);
			playButton->setVisible(true);
			running = false;
			return true;
		}
		return false;
	};
	stopButton->setVisible(false);
	infoComposite->add(playButton);
	infoComposite->add(stopButton);
	controlLayout->setSouth(infoComposite, UnitPX(80.0f));
	rootNode.add(layout);

	controls->addGroup("Simulation", true);
	simulation->setup(controls);
	controls->addGroup("Visualization", true);
	controls->addNumberField("Transparency", transparency, Float(0.0f), Float(1.0f));
	controls->addNumberField("Line Width", lineWidth, Float(0.0f), Float(4.0f));

	controls->addCheckBox("Show Centers", showCenters);
	controls->addCheckBox("Show Contours", showContours);
	timelineSlider = TimelineSliderPtr(
		new TimelineSlider("Timeline", CoordPerPX(0.0f, 1.0f, 0.0f, -80.0f), CoordPerPX(1.0f, 0.0f, 0.0f, 80.0f), Integer(0), Integer(0), Integer(0)));
	CompositePtr viewRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPerPX(1.0f, 1.0f, 0.0f, -80.0f)));
	timelineSlider->backgroundColor = MakeColor(AlloyApplicationContext()->theme.DARKER);
	timelineSlider->borderColor = MakeColor(AlloyApplicationContext()->theme.DARK);
	timelineSlider->borderWidth = UnitPX(0.0f);
	timelineSlider->onChangeEvent = [this](const Number& timeValue, const Number& lowerValue, const Number& upperValue) {

	};
	timelineSlider->setMajorTick(100);
	timelineSlider->setMinorTick(10);
	timelineSlider->setLowerValue(0);
	timelineSlider->setUpperValue(0);
	int maxIteration = (int)std::ceil(simulation->getSimulationDuration() / simulation->getTimeStep());
	timelineSlider->setMaxValue(maxIteration);
	timelineSlider->setVisible(true);
	timelineSlider->setModifiable(false);
	renderRegion->add(viewRegion);
	renderRegion->add(timelineSlider);

	float downScale = std::min((getContext()->getScreenWidth() - 350.0f) / img.width, (getContext()->getScreenHeight() - 80.0f) / img.height);
	resizeableRegion = AdjustableCompositePtr(
		new AdjustableComposite("Image", CoordPerPX(0.5, 0.5, -img.width * downScale * 0.5f, -img.height * downScale * 0.5f),
			CoordPX(img.width * downScale, img.height * downScale)));
	Application::addListener(resizeableRegion.get());
	ImageGlyphPtr imageGlyph = AlloyApplicationContext()->createImageGlyph(img, false);
	overlayGlyph = AlloyApplicationContext()->createImageGlyph(simulation->getContour()->overlay, false);
	DrawPtr drawContour = DrawPtr(new Draw("Contour Draw", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), [this](AlloyContext* context, const box2px& bounds) {
		int currentTime = timelineSlider->getTimeValue().toInteger();
		std::shared_ptr<CacheElement> elem = this->cache->get(currentTime);
		Contour2D* contour;
		if (elem.get() != nullptr) {
			contour = elem->getContour().get();
		}
		else {
			contour = simulation->getContour();
		}
		if (currentTime != lastSimTime) {
			overlayGlyph->set(contour->overlay, context);
			lastSimTime = currentTime;
		}
		NVGcontext* nvg = context->nvgContext;
		nvgLineCap(nvg, NVG_ROUND);
		float scale = bounds.dimensions.x / (float)img.width;
		if (0.05f*scale > 0.5f) {
			nvgStrokeWidth(nvg, 0.05f*scale);			
			nvgStrokeColor(nvg, Color(0.4f, 0.4f, 0.4f, 0.5f));
			nvgBeginPath(nvg);
			for (int i = 0;i < img.width;i++) {
				float2 pt = float2(0.5f + i, 0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgMoveTo(nvg, pt.x, pt.y);
				pt = float2(0.5f + i, 0.5f + img.height - 1.0f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgLineTo(nvg, pt.x, pt.y);
			}
			for (int j = 0;j < img.height;j++) {
				float2 pt = float2(0.5f, 0.5f + j);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgMoveTo(nvg, pt.x, pt.y);
				pt = float2(0.5f + img.width - 1.0f, 0.5f + j);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgLineTo(nvg, pt.x, pt.y);
			}
			nvgStroke(nvg);
		}
		NVGpaint imgPaint = nvgImagePattern(nvg, bounds.position.x,
			bounds.position.y, bounds.dimensions.x, bounds.dimensions.y, 0.f,
			overlayGlyph->handle, transparency.toFloat());
		nvgBeginPath(nvg);
		nvgFillColor(nvg, Color(COLOR_WHITE));
		nvgRect(nvg, bounds.position.x, bounds.position.y, bounds.dimensions.x,
			bounds.dimensions.y);
		nvgFillPaint(nvg, imgPaint);
		nvgFill(nvg);
		float circlRadius = 1.0f;
		nvgStrokeWidth(nvg,1.0f);
		nvgStrokeColor(nvg, Color(0.0, 0.0f, 0.0f, 1.0f));
		if (showCenters&&circlRadius*scale > 0.5f) {
			const Vector2f& centers = contour->clusterCenters;
			const Vector3f& colors = contour->clusterColors;
			int L = (int)centers.size();
			for (int n = 0;n < L;n++) {
				float2 pt = centers[n] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgFillColor(nvg, Color(LABAtoRGBA(float4(colors[n],1.0f))));
				nvgBeginPath(nvg);
				nvgEllipse(nvg, pt.x, pt.y, circlRadius*scale, circlRadius*scale);
				nvgFill(nvg);
				nvgStroke(nvg);
			}
		}
		if (showContours&&scale*lineWidth.toFloat()>0.5f) {
			nvgStrokeWidth(nvg, scale*lineWidth.toFloat());
			nvgStrokeColor(nvg, Color(0.0f, 0.0f, 0.0f, 1.0f));
			for (int n = 0;n < (int)contour->indexes.size();n++) {
				std::list<uint32_t> curve = contour->indexes[n];
				nvgPathWinding(nvg, NVG_CW);
				nvgBeginPath(nvg);
				bool firstTime = true;
				for (uint32_t idx : curve) {
					float2 pt = contour->vertexes[idx] + float2(0.5f);
					pt.x = pt.x / (float)img.width;
					pt.y = pt.y / (float)img.height;
					pt = pt*bounds.dimensions + bounds.position;
					if (firstTime) {
						nvgMoveTo(nvg, pt.x, pt.y);
					}
					else {
						nvgLineTo(nvg, pt.x, pt.y);
					}
					firstTime = false;
				}
				nvgStroke(nvg);
			}
		}
	}));
	
	GlyphRegionPtr glyphRegion = GlyphRegionPtr(new GlyphRegion("Image Region",imageGlyph, CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	glyphRegion->setAspectRule(AspectRule::Unspecified);
	glyphRegion->foregroundColor = MakeColor(COLOR_NONE);
	glyphRegion->backgroundColor = MakeColor(COLOR_NONE);
	glyphRegion->borderColor = MakeColor(COLOR_NONE);
	drawContour->onScroll = [this](AlloyContext* context, const InputEvent& event)
	{
		box2px bounds = resizeableRegion->getBounds(false);
		pixel scaling = (pixel)(1 - 0.1f*event.scroll.y);
		pixel2 newBounds = bounds.dimensions*scaling;
		pixel2 cursor = context->cursorPosition;
		pixel2 relPos = (cursor - bounds.position) / bounds.dimensions;
		pixel2 newPos = cursor - relPos*newBounds;
		bounds.position = newPos;
		bounds.dimensions = newBounds;
		resizeableRegion->setDragOffset(pixel2(0, 0));
		resizeableRegion->position = CoordPX(bounds.position - resizeableRegion->parent->getBoundsPosition());
		resizeableRegion->dimensions = CoordPX(bounds.dimensions);

		float2 dims = float2(img.dimensions());
		cursor = aly::clamp(dims*(event.cursor - bounds.position) / bounds.dimensions, float2(0.0f), dims);

		context->requestPack();
		return true;
	};

	resizeableRegion->add(glyphRegion);
	resizeableRegion->add(drawContour);

	resizeableRegion->setAspectRatio(img.width / (float)img.height);
	resizeableRegion->setAspectRule(AspectRule::FixedHeight);
	resizeableRegion->setDragEnabled(true);
	resizeableRegion->setClampDragToParentBounds(false);
	resizeableRegion->borderWidth = UnitPX(2.0f);
	resizeableRegion->borderColor = MakeColor(AlloyApplicationContext()->theme.LIGHTER);

	glyphRegion->onMouseDown = [=](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			//Bring component to top by setting it to be drawn last.
			dynamic_cast<Composite*>(resizeableRegion->parent)->putLast(resizeableRegion);
			resizeableRegion->borderColor = MakeColor(AlloyApplicationContext()->theme.LIGHTEST);
		}
		return false;
	};
	glyphRegion->onMouseUp = [=](AlloyContext* context, const InputEvent& e) {
		resizeableRegion->borderColor = MakeColor(AlloyApplicationContext()->theme.LIGHTER);
		return false;
	};

	viewRegion->backgroundColor = MakeColor(getContext()->theme.DARKER);
	viewRegion->borderColor = MakeColor(getContext()->theme.DARK);
	viewRegion->borderWidth = UnitPX(1.0f);
	viewRegion->add(resizeableRegion);

	return true;
}
void SuperPixelToy::draw(AlloyContext* context) {
	if (running) {
		if (!simulation->step()) {
			running = false;
		}
	}
}

