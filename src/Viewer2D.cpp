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
#include "Viewer2D.h"
 /*
  * Copyright(C) 2015, Blake C. Lucas, Ph.D. (img.science@gmail.com)
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
#include "Viewer2D.h"
#include "AlloyDistanceField.h"
#include "AlloyIsoContour.h"
using namespace aly;

Viewer2D::Viewer2D() :
	Application(1200, 600, "Distance Field Example"), currentIso(0.0f) {
}
void Viewer2D::createTextLevelSet(aly::Image1f& distField, aly::Image1f& gray, int w, int h, const std::string& text, float textSize, float maxDistance) {
	GLFrameBuffer renderBuffer;
	//Render text to image
	NVGcontext* nvg = getContext()->nvgContext;
	renderBuffer.initialize(w, h);
	renderBuffer.begin(RGBAf(1.0f, 1.0f, 1.0f, 1.0f));
	nvgBeginFrame(nvg, w, h, 1.0f);
	nvgFontFaceId(nvg, getContext()->getFont(FontType::Bold)->handle);
	nvgFillColor(nvg, Color(0, 0, 0));
	nvgFontSize(nvg, textSize);
	nvgTextAlign(nvg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
	nvgText(nvg, w * 0.5f, h * 0.5f, text.c_str(), nullptr);
	nvgEndFrame(nvg);
	renderBuffer.end();
	ImageRGBAf img = renderBuffer.getTexture().read();
	FlipVertical(img);
	ConvertImage(img, gray);
	//Make boundary == 0, outside == 0.5 inside == -0.5
	gray -= float1(0.5f);
	DistanceField2f df;
	//Solve distance field out to +/- 40 pixels
	df.solve(gray, distField, maxDistance);
	gray = (-gray + float1(0.5f));
}
aly::Image1f Viewer2D::createCircleLevelSet(int w, int h, float2 center, float r) {
	aly::Image1f levelSet(w, h);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			levelSet(i, j).x = distance(float2((float)i, (float)j), center) - r;
		}
	}
	return levelSet;
}
bool Viewer2D::init(Composite& rootNode) {
	int w = 256;
	int h = 256;
	Image1f gray;
	Image1f distField;
	float maxDistance = 128;
	createTextLevelSet(distField, gray, w, h, "A", 200.0f, maxDistance);
	ConvertImage(gray, img);
	cache = std::shared_ptr<SpringlCache2D>(new SpringlCache2D());
	simulation = std::shared_ptr<ActiveContour2D>(new ActiveContour2D(cache));
	simulation->onUpdate = [this](uint64_t iteration, bool lastIteration) {
		if (lastIteration) {
			stopButton->setVisible(false);
			playButton->setVisible(true);
		}
		AlloyApplicationContext()->addDeferredTask([this]() {
			timelineSlider->setUpperValue((int)simulation->getSimulationIteration());
			timelineSlider->setTimeValue((int)simulation->getSimulationIteration());

		});
	};

	simulation->setInitialDistanceField(createCircleLevelSet(w, h, float2(0.5f*w, 0.5f*h), std::min(w, h)*0.25f));
	simulation->setPressure(gray, 1.0f, 0.5f);
	simulation->init();


	parametersDirty = true;
	frameBuffersDirty = true;

	BorderCompositePtr layout = BorderCompositePtr(new BorderComposite("UI Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), false));
	ParameterPanePtr controls = ParameterPanePtr(new ParameterPane("Controls", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	BorderCompositePtr controlLayout = BorderCompositePtr(new BorderComposite("Control Layout", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), true));

	controls->onChange = [this](const std::string& label, const AnyInterface& value) {

		parametersDirty = true;
	};

	float aspect = 6.0f;
	lineWidth = Float(2.0f);
	particleSize = Float(0.2f);

	lineColor = AlloyApplicationContext()->theme.DARK.toSemiTransparent(0.5f);
	pointColor = Color(255, 255, 255, 255);

	controls->setAlwaysShowVerticalScrollBar(false);
	controls->setScrollEnabled(false);
	controls->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controls->borderColor = MakeColor(getContext()->theme.DARK);
	controls->borderWidth = UnitPX(1.0f);

	controlLayout->backgroundColor = MakeColor(getContext()->theme.DARKER);
	controlLayout->borderWidth = UnitPX(0.0f);
	CompositePtr renderRegion = CompositePtr(new Composite("View", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	layout->setWest(controlLayout, UnitPX(300.0f));
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
			simulation->cancel();
			cache->clear();
			simulation->init();
			int maxIteration = (int)std::ceil(simulation->getSimulationDuration() / simulation->getTimeStep());
			timelineSlider->setTimeValue(0);
			timelineSlider->setMaxValue(maxIteration);
			timelineSlider->setVisible(true);
			simulation->execute();
			return true;
		}
		return false;
	};
	stopButton->onMouseDown = [this](AlloyContext* context, const InputEvent& e) {
		if (e.button == GLFW_MOUSE_BUTTON_LEFT) {
			stopButton->setVisible(false);
			playButton->setVisible(true);
			simulation->cancel();
			return true;
		}
		return false;
	};
	stopButton->setVisible(false);
	infoComposite->add(playButton);
	infoComposite->add(stopButton);
	controlLayout->setSouth(infoComposite, UnitPX(80.0f));
	rootNode.add(layout);
	ImageRGBA tmpImg(4, 4);
	tmpImg.set(RGBA(255, 255, 255, 255));
	controls->addGroup("Visualization", true);
	controls->addNumberField("Line Width", lineWidth, Float(1.0f), Float(10.0f), 5.5f);
	controls->addNumberField("Particle Size", particleSize, Float(0.0f), Float(1.0f), 5.5f);
	controls->addColorField("Point", pointColor);
	controls->addColorField("Line", lineColor);

	controls->addGroup("Simulation", true);
	simulation->setup(controls);
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

	float downScale = 1.0f;
	static int offsetIncrement = 0;
	pixel2 offset(50.0f * offsetIncrement, 50.0f * offsetIncrement);
	offsetIncrement++;
	if (img.width > 0 && img.height > 0) {
		downScale = std::min(650.0f / img.width, 580.0f / img.height);
	}
	resizeableRegion = AdjustableCompositePtr(
		new AdjustableComposite("Image", CoordPerPX(0.5, 0.5, -img.width * downScale * 0.5f + offset.x, -img.height * downScale * 0.5f + offset.y),
			CoordPX(img.width * downScale, img.height * downScale)));
	Application::addListener(resizeableRegion.get());
	ImageGlyphPtr imageGlyph = AlloyApplicationContext()->createImageGlyph(img, false);
	DrawPtr drawMarkers = DrawPtr(new Draw("Marker Draw", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), [this](AlloyContext* context, const box2px& bounds) {
		NVGcontext* nvg = context->nvgContext;
		std::shared_ptr<CacheElement> elem = this->cache->get(timelineSlider->getTimeValue().toInteger());
		Contour2D contour;
		if (elem.get() != nullptr) {
			contour = *elem->getContour();
		}
		else {
			contour = simulation->getContour();
		}
		nvgStrokeWidth(nvg, 4.0f);
		nvgStrokeColor(nvg, Color(255, 128, 64));
		nvgLineCap(nvg, NVG_ROUND);
		nvgBeginPath(nvg);
		for (int n = 0;n < (int)contour.indexes.size();n++) {
			std::list<uint32_t> curve = contour.indexes[n];
			bool firstTime = true;
			for (uint32_t idx : curve) {
				float2 pt = contour.vertexes[idx];
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
		}
		nvgStroke(nvg);
	}));
	GlyphRegionPtr glyphRegion = GlyphRegionPtr(new GlyphRegion("Image Region", imageGlyph, CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
	glyphRegion->setAspectRule(AspectRule::Unspecified);
	glyphRegion->foregroundColor = MakeColor(COLOR_NONE);
	glyphRegion->backgroundColor = MakeColor(COLOR_NONE);
	drawMarkers->onScroll = [this](AlloyContext* context, const InputEvent& event)
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
	drawMarkers->onMouseOver = [this](AlloyContext* context, const InputEvent& event) {
		box2px bbox = resizeableRegion->getBounds(true);
		float2 dims = float2(img.dimensions());
		float2 cursor = aly::clamp(dims*(event.cursor - bbox.position) / bbox.dimensions, float2(0.0f), dims);
		return false;
	};
	resizeableRegion->add(glyphRegion);
	resizeableRegion->add(drawMarkers);
	resizeableRegion->setAspectRatio(img.width / (float)img.height);
	resizeableRegion->setAspectRule(AspectRule::FixedHeight);
	resizeableRegion->setDragEnabled(true);
	resizeableRegion->setClampDragToParentBounds(false);
	resizeableRegion->backgroundColor = MakeColor(0, 0, 220, 255);
	resizeableRegion->borderWidth = UnitPX(1.0f);
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
	viewRegion->add(resizeableRegion);
	return true;
}

