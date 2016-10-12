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
	const std::shared_ptr<AlloyContext>& context) :GLShader(onScreen, context), context(context),texture(onScreen,context) {
	initialize({},
		R"(	#version 330
				layout(location = 0) in vec4 vp;
				layout(location = 1) in vec2 pp;
				out LINE {
					vec2 p0;
					vec2 p1;
					vec2 pt;
				} line;
				void main() {
					line.p0=vp.xy+vec2(0.5);
					line.p1=vp.zw+vec2(0.5);
					line.pt=pp+vec2(0.5);
				})",
		R"(	#version 330
				uniform int width;
				uniform int height;
				uniform float max_distance;
				out vec4 FragColor;
				in vec2 p0;
				in vec2 p1;
				in vec2 pos;
				void main() {
					float dist=0;
					float l2 = dot(p1-p0,p1-p0);
					if (l2<1E-6) {
						dist=min(distance(pos,p0), distance(pos, p1));
					} else {
						float t = dot(pos-p0,p1-p0) / l2;
						if (t < 0.0) {
							dist=distance(pos,p0);
						} else if (t > 1.0) {
							dist=distance(pos,p1); 
						} else {
							dist=distance(pos, p0 + t * (p1-p0));
						}
					}
					float c=dist/max_distance;
					if(dist<=max_distance){
						FragColor=vec4(1.0-c,1.0-c,1.0-c,1.0);
						gl_FragDepth=c;
					} else {
						discard;
					}
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
						vec2 pt;
					} line[];
					out vec2 p0;
					out vec2 p1;
					out vec2 pos;
					void main() {
					  vec4 q=vec4(0.0,0.0,1.0,1.0);
					  vec2 scale=vec2(2.0/float(width),2.0/float(height));
					  vec2 tan;
					  vec2 norm;
					  p0=line[0].p0;
					  p1=line[0].pt;

					  tan=normalize(p1-p0);	
					  norm=vec2(-tan.y,tan.x);

					  pos=(p0+(-norm-tan)*max_distance);
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();

					  pos=(p0+(norm-tan)*max_distance);
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();

					  pos=(p1+(-norm+tan)*max_distance); 
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();

					  pos=(p1+(norm+tan)*max_distance); 
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();
					
					  EndPrimitive();

					  p0=line[0].pt;
					  p1=line[0].p1;
					  tan=normalize(p1-p0);	
					  norm=vec2(-tan.y,tan.x);
					  pos=(p0+(-norm-tan)*max_distance);
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();

					  pos=(p0+(norm-tan)*max_distance);
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();

				      pos=(p1+(-norm+tan)*max_distance); 
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();

					  pos=(p1+(norm+tan)*max_distance); 
					  q.xy=scale*pos-vec2(1.0);
					  gl_Position=q;
					  EmitVertex();
					
					  EndPrimitive();

		})");
}
void UnsignedDistanceShader::init(int width, int height) {
	texture.initialize(width, height);
}
Image1f UnsignedDistanceShader::solve(Manifold2D& contour,float maxDistance) {
	texture.begin();
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	begin();
	set("width", texture.width());
	set("height", texture.height());
	set("max_distance", maxDistance);
	contour.draw();
	end();
	glEnable(GL_BLEND);
	texture.end();
	ImageRGBAf& tmp= texture.getTexture().read();
	int N = (int)tmp.size();
	Image1f out(tmp.width, tmp.height);
#pragma omp parallel for
	for (int n = 0;n < N;n++) {
		RGBAf c = tmp[n];
		out[n] = (c.w==0.0f)?maxDistance:(1.0f - c.z)*maxDistance;
	}
	return out;
}
}