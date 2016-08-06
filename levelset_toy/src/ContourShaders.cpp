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
#include "ContourShaders.h"
namespace aly{
UnsignedDistanceShader::UnsignedDistanceShader(bool onScreen,
	const std::shared_ptr<AlloyContext>& context) :GLShader(onScreen, context), context(context),texture(true,context) {
	initialize({},
		R"(	#version 330
				layout(location = 0) in vec4 vp;
				out LINE {
					vec2 p0;
					vec2 p1;
				} line;
				void main() {
					line.p0=vp.xy;
					line.p1=vp.zw;
				})",
		R"(	#version 330
				in vec3 p0,p1;
				in vec3 normal,vert;
				in vec4 pos;
				uniform int width;
				uniform int height;
				uniform float max_distance;
				out vec4 FragColor;
				void main() {
					FragColor=vec4(1.0,1.0f,0.0f,1.0f);
					//gl_FragDepth=d;
				}
		)",
		R"(	#version 330
					layout (points) in;
					layout (triangle_strip, max_vertices=4) out;
					uniform float max_distance;
					uniform int width;
					uniform int height;
					in LINE {
						vec2 p0;
						vec2 p1;
					} line[];
					void main() {
					  vec4 q=vec4(0.0,0.0,0.0,1.0);
					  vec2 scale=vec2(2.0/float(width),2.0/float(height));
					  vec2 p0=line[0].p0;
					  vec2 p1=line[0].p1;
					  vec2 tan=normalize(p1-p0);	
					  vec2 norm=vec2(-tan.y,tan.x);
					  q.xy=scale*(p0+(norm-tan)*max_distance)-vec2(1.0f);
					  gl_Position=q;
					  EmitVertex();
					  q.xy=scale*(p0+(-norm-tan)*max_distance)-vec2(1.0f);
					  gl_Position=q;
					  EmitVertex();
					  q.xy=scale*(p1+(-norm+tan)*max_distance)-vec2(1.0f); 
					  gl_Position=q;
					  EmitVertex();
					  q.xy=scale*(p1+(norm+tan)*max_distance)-vec2(1.0f); 
					  gl_Position=q;
					  EmitVertex();
					  EndPrimitive();
					 })");
}
void UnsignedDistanceShader::init(int width, int height) {
	texture.initialize(width, height);
}
void UnsignedDistanceShader::draw(Contour2D& contour) {
	texture.begin();
	begin();
	set("width", texture.width());
	set("height", texture.height());
	set("max_distance", 4.0f);
	contour.draw();
	end();
	texture.end();
}
ImageRGBAf UnsignedDistanceShader::getUnsignedDistance() {
	return texture.getTexture().read();
}
}