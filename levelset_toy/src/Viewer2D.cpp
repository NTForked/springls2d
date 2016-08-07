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
#include "SpringLevelSet2D.h"
using namespace aly;

Viewer2D::Viewer2D() :
	Application(1300, 1000, "Level Set Segmenation Toy", false), currentIso(0.0f) {
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
	int w = 128;
	int h = 128;
	Image1f gray;
	Image1f distField;
	float maxDistance = 64;
	createTextLevelSet(distField, gray, w, h, "A", 100.0f, maxDistance);
	ConvertImage(gray, img);
	cache = std::shared_ptr<SpringlCache2D>(new SpringlCache2D());
	simulation = std::shared_ptr<ActiveContour2D>(new SpringLevelSet2D(cache));
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
	//createTextLevelSet(distField, gray, w, h, "Q", 200.0f, maxDistance);
	//simulation->setInitialDistanceField(distField);
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
	lineWidth = Float(4.0f);
	particleSize = Float(0.2f);

	lineColor = Color(0.0f, 0.5f, 0.5f, 1.0f);
	pointColor = Color(1.0f, 0.8f, 0.0f, 1.0f);
	springlColor = Color(0.5f, 0.5f, 0.5f, 1.0f);
	particleColor = Color(0.6f, 0.0f, 0.0f, 1.0f);
	normalColor = Color(0.0f, 0.8f, 0.0f, 0.5f);
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

	controls->addGroup("Simulation", true);
	simulation->setup(controls);
	controls->addGroup("Visualization", true);
	controls->addNumberField("Line Width", lineWidth, Float(1.0f), Float(20.0f), 6.0f);
	controls->addNumberField("Particle Size", particleSize, Float(0.0f), Float(1.0f), 6.0f);
	controls->addColorField("Element", springlColor);
	controls->addColorField("Particle", particleColor);
	controls->addColorField("Point", pointColor);
	controls->addColorField("Normal", normalColor);
	controls->addColorField("Line", lineColor);

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

	float downScale = std::min((getContext()->getScreenWidth()-350.0f) / img.width, (getContext()->getScreenHeight() - 80.0f) / img.height);
	resizeableRegion = AdjustableCompositePtr(
		new AdjustableComposite("Image", CoordPerPX(0.5, 0.5, -img.width * downScale * 0.5f, -img.height * downScale * 0.5f),
			CoordPX(img.width * downScale, img.height * downScale)));
	Application::addListener(resizeableRegion.get());
	ImageGlyphPtr imageGlyph = AlloyApplicationContext()->createImageGlyph(img, false);
	DrawPtr drawContour = DrawPtr(new Draw("Contour Draw", CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f), [this](AlloyContext* context, const box2px& bounds) {
		std::shared_ptr<CacheElement> elem = this->cache->get(timelineSlider->getTimeValue().toInteger());
		Contour2D contour;
		if (elem.get() != nullptr) {
			contour = *elem->getContour();
		}
		else {
			contour = simulation->getContour();

		}
		NVGcontext* nvg = context->nvgContext;
		nvgLineCap(nvg, NVG_ROUND);
		float scale = bounds.dimensions.x / (float)img.width;
		nvgStrokeColor(nvg, Color(0.8f,0.8f,0.8f,0.5f));
		if (0.05f*scale > 0.5f) {
			nvgStrokeWidth(nvg,0.05f*scale);
			nvgBeginPath(nvg);
			for (int i = 0;i < img.width;i++) {
				float2 pt = float2(0.5f+i,0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgMoveTo(nvg, pt.x, pt.y);
				pt = float2(0.5f + i, 0.5f + img.height-1.0f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgLineTo(nvg, pt.x, pt.y);
			}
			for (int j = 0;j < img.height;j++) {
				float2 pt = float2(0.5f, 0.5f+j);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgMoveTo(nvg, pt.x, pt.y);
				pt = float2(0.5f + img.width-1.0f, 0.5f + j);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgLineTo(nvg, pt.x, pt.y);
			}
			nvgStroke(nvg);
		}
		nvgStrokeColor(nvg, lineColor);
		nvgStrokeWidth(nvg, lineWidth.toFloat());
		nvgBeginPath(nvg);
		for (int n = 0;n < (int)contour.indexes.size();n++) {
			std::list<uint32_t> curve = contour.indexes[n];
			bool firstTime = true;
			for (uint32_t idx : curve) {
				float2 pt = contour.vertexes[idx] + float2(0.5f);
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

		if (0.1f*scale > 0.5f) {
			nvgStrokeColor(nvg, springlColor);
			nvgStrokeWidth(nvg, 0.1f*scale);
			for (int n = 0;n < (int)contour.points.size();n += 2) {
				float2 pt = contour.points[n] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgBeginPath(nvg);
				nvgMoveTo(nvg, pt.x, pt.y);

				pt = contour.points[n + 1] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;

				nvgLineTo(nvg, pt.x, pt.y);
				nvgStroke(nvg);
			}
		}

		if (0.05f*scale > 0.5f) {
			nvgStrokeColor(nvg, normalColor);
			nvgStrokeWidth(nvg, 0.05f*scale);
			for (int n = 0;n < (int)contour.normals.size();n++) {
				float2 pt = contour.particles[n] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgBeginPath(nvg);
				nvgMoveTo(nvg, pt.x, pt.y);
				pt = contour.particles[n] + SpringLevelSet2D::EXTENT*contour.normals[n] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgLineTo(nvg, pt.x, pt.y);
				nvgStroke(nvg);
			}
		}
		if (0.05f*scale > 0.5f) {
			nvgFillColor(nvg, pointColor);
			for (int n = 0;n < (int)contour.points.size();n++) {
				float2 pt = contour.points[n] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgBeginPath(nvg);
				nvgEllipse(nvg, pt.x, pt.y, 0.05f*scale, 0.05f*scale);
				nvgFill(nvg);
			}
		}

		if (0.1f*scale > 0.5f) {
			nvgFillColor(nvg, particleColor);
			for (int n = 0;n < (int)contour.particles.size();n++) {
				float2 pt = contour.particles[n] + float2(0.5f);
				pt.x = pt.x / (float)img.width;
				pt.y = pt.y / (float)img.height;
				pt = pt*bounds.dimensions + bounds.position;
				nvgBeginPath(nvg);
				nvgEllipse(nvg, pt.x, pt.y, 0.1f*scale, 0.1f*scale);
				nvgFill(nvg);
			}
		}
	}));
	GlyphRegionPtr glyphRegion = GlyphRegionPtr(new GlyphRegion("Image Region", imageGlyph, CoordPX(0.0f, 0.0f), CoordPercent(1.0f, 1.0f)));
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
	drawContour->onMouseOver = [this](AlloyContext* context, const InputEvent& event) {
		box2px bbox = resizeableRegion->getBounds(true);
		float2 dims = float2(img.dimensions());
		float2 cursor = aly::clamp(dims*(event.cursor - bbox.position) / bbox.dimensions, float2(0.0f), dims);
		return false;
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

