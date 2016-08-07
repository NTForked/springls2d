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
#include "Contour2D.h"
#include "AlloyFileUtil.h"
#include "AlloyContext.h"
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>

namespace aly {

	Contour2D::Contour2D(bool onScreen, const std::shared_ptr<AlloyContext>& context):onScreen(onScreen),context(context),vao(0),vertexBuffer(0),dirty(false) , vertexCount(0){
	}
	Contour2D::~Contour2D() {
		context->begin(onScreen);
		if (glIsBuffer(vertexBuffer) == GL_TRUE)
			glDeleteBuffers(1, &vertexBuffer);
		if (vao != 0)
			glDeleteVertexArrays(1, &vao);
		context->end();
	}
	void Contour2D::draw() {
		if (dirty) {
			update();
		}
		context->begin(onScreen);
		if (vao > 0)
			glBindVertexArray(vao);
		if (vertexBuffer > 0) {
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
			glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
			glDrawArrays(GL_POINTS, 0, (GLsizei)(points.size()/2));
			glDisableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		glBindVertexArray(0);
		context->end();
	}
	void Contour2D::update() {
		context->begin(onScreen);
		if (vao == 0)
			glGenVertexArrays(1, &vao);
		if (points.size() > 0) {
			if (vertexCount != points.size()) {
				if (glIsBuffer(vertexBuffer) == GL_TRUE)
					glDeleteBuffers(1, &vertexBuffer);
				glGenBuffers(1, &vertexBuffer);
			}
			glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
			if (glIsBuffer(vertexBuffer) == GL_FALSE)
				throw std::runtime_error("Error: Unable to create vertex buffer");
			glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat) * 2 *points.size(),
				points.ptr(), GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			vertexCount =(int) points.size();
		}
		else {
			if (glIsBuffer(vertexBuffer) == GL_TRUE)
				glDeleteBuffers(1, &vertexBuffer);
			vertexCount = 0;
		}
		context->end();
		dirty = false;
	}
	void Contour2D::updateNormals() {
		normals.resize(points.size() / 2);
#pragma omp parallel for
		for (int i = 0;i < points.size();i += 2) {
			float2 norm = normalize(points[i+1] - points[i]);
			normals[i / 2] = float2(-norm.y,norm.x);
		}
	}
	void ReadContourFromFile(const std::string& file, Contour2D& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ifstream os(file);
			cereal::JSONInputArchive archive(os);
			archive(cereal::make_nvp("contour", params));
		}
		else if (ext == "xml") {
			std::ifstream os(file);
			cereal::XMLInputArchive archive(os);
			archive(cereal::make_nvp("contour", params));
		}
		else {
			std::ifstream os(file, std::ios::binary);
			cereal::PortableBinaryInputArchive archive(os);
			archive(cereal::make_nvp("contour", params));
		}
		params.setFile(file);
	}
	void WriteContourToFile(const std::string& file, Contour2D& params) {
		std::string ext = GetFileExtension(file);
		if (ext == "json") {
			std::ofstream os(file);
			cereal::JSONOutputArchive archive(os);
			archive(cereal::make_nvp("contour", params));
		}
		else if (ext == "xml") {
			std::ofstream os(file);
			cereal::XMLOutputArchive archive(os);
			archive(cereal::make_nvp("contour", params));
		}
		else {
			std::ofstream os(file, std::ios::binary);
			cereal::PortableBinaryOutputArchive archive(os);
			archive(cereal::make_nvp("contour", params));
		}
		params.setFile(file);
	}

}